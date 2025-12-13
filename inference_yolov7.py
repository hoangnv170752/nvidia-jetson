"""
YOLOv7 Inference Script
Supports image, video, and webcam inference
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import time


class YOLOv7Detector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, device=''):
        """
        Initialize YOLOv7 detector
        
        Args:
            model_path: Path to YOLOv7 .pt model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', '0', '1', etc.)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device == '':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading YOLOv7 model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = torch.load(model_path, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        
        # Get model info
        self.stride = int(self.model.stride.max())
        self.names = self.model.names if hasattr(self.model, 'names') else [f'class{i}' for i in range(1000)]
        
        print(f"Model loaded successfully. Classes: {len(self.names)}")
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize and pad image while maintaining aspect ratio"""
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, r, (dw, dh)
    
    def preprocess(self, img):
        """Preprocess image for inference"""
        img_processed, ratio, pad = self.letterbox(img, new_shape=(640, 640))
        img_processed = img_processed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_processed = np.ascontiguousarray(img_processed)
        img_processed = torch.from_numpy(img_processed).to(self.device)
        img_processed = img_processed.float() / 255.0
        
        if len(img_processed.shape) == 3:
            img_processed = img_processed[None]
        
        return img_processed, ratio, pad
    
    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """Perform Non-Maximum Suppression (NMS) on inference results"""
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        
        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            
            if not x.shape[0]:
                continue
            
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            
            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            
            # Apply NMS
            if x.shape[0]:
                c = x[:, 5:6] * 4096  # classes
                boxes, scores = x[:, :4] + c, x[:, 4]
                i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
                if i.shape[0] > max_det:
                    i = i[:max_det]
                output[xi] = x[i]
        
        return output
    
    def xywh2xyxy(self, x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y
    
    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """Rescale coords (xyxy) from img1_shape to img0_shape"""
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0]
            pad = ratio_pad[1]
        
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        
        # Clip bounding boxes
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])
        
        return coords
    
    def detect(self, img):
        """
        Run detection on image
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            detections: List of detections [x1, y1, x2, y2, conf, cls]
        """
        img0 = img.copy()
        img_processed, ratio, pad = self.preprocess(img)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img_processed)[0]
        
        # NMS
        pred = self.non_max_suppression(pred, self.conf_threshold, self.iou_threshold)
        
        # Process detections
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = self.scale_coords(img_processed.shape[2:], det[:, :4], img0.shape, (ratio, pad)).round()
                detections = det.cpu().numpy()
        
        return detections
    
    def draw_detections(self, img, detections):
        """Draw bounding boxes and labels on image"""
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            
            # Draw box
            color = self.get_color(cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{self.names[cls]} {conf:.2f}'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def get_color(self, idx):
        """Get color for class index"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
        return colors[idx % len(colors)]


def inference_image(detector, image_path, output_path=None, show=True):
    """Run inference on single image"""
    print(f"\nProcessing image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Detect
    start_time = time.time()
    detections = detector.detect(img)
    inference_time = time.time() - start_time
    
    # Draw results
    img_result = detector.draw_detections(img, detections)
    
    print(f"Detections: {len(detections)}, Inference time: {inference_time*1000:.2f}ms")
    
    # Save or show
    if output_path:
        cv2.imwrite(output_path, img_result)
        print(f"Saved result to {output_path}")
    
    if show:
        cv2.imshow('YOLOv7 Detection', img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def inference_video(detector, video_path, output_path=None, show=True):
    """Run inference on video with comprehensive metrics"""
    print(f"\nProcessing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Metrics tracking
    frame_count = 0
    total_time = 0
    total_detections = 0
    confidence_scores = []
    fps_history = []
    
    print("Processing frames... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - start_time
        total_time += inference_time
        frame_count += 1
        
        # Get detection metrics
        frame_detections = len(detections)
        total_detections += frame_detections
        
        # Collect confidence scores
        for det in detections:
            confidence_scores.append(det[4])  # confidence is at index 4 in YOLOv7
        
        # Calculate current FPS
        current_fps = 1/inference_time if inference_time > 0 else 0
        fps_history.append(current_fps)
        
        # Draw results
        frame_result = detector.draw_detections(frame, detections)
        
        # Add comprehensive metrics overlay
        y_offset = 30
        cv2.putText(frame_result, f"FPS: {current_fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame_result, f"Detections: {frame_detections}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame_result, f"Frame: {frame_count}/{total_frames}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if confidence_scores:
            avg_conf = sum(confidence_scores[-frame_detections:]) / max(1, frame_detections)
            y_offset += 30
            cv2.putText(frame_result, f"Avg Conf: {avg_conf:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progress bar
        progress = frame_count / total_frames
        bar_width = 300
        bar_height = 10
        bar_x, bar_y = 10, height - 30
        cv2.rectangle(frame_result, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame_result, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        cv2.putText(frame_result, f"Progress: {progress*100:.1f}%", (bar_x + bar_width + 10, bar_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write or show
        if writer:
            writer.write(frame_result)
        
        if show:
            cv2.imshow('YOLOv7 Detection', frame_result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print detailed progress
        if frame_count % 30 == 0:
            avg_fps_recent = sum(fps_history[-30:]) / min(30, len(fps_history))
            print(f"Frame {frame_count}/{total_frames} | FPS: {current_fps:.1f} (avg: {avg_fps_recent:.1f}) | Detections: {frame_detections} | Progress: {progress*100:.1f}%")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    print(f"\n{'='*50}")
    print(f"VIDEO PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {avg_detections_per_frame:.2f}")
    print(f"Average confidence score: {avg_confidence:.3f}")
    print(f"Min FPS: {min(fps_history):.1f}")
    print(f"Max FPS: {max(fps_history):.1f}")
    if output_path:
        print(f"Output saved to: {output_path}")
    print(f"{'='*50}")


def inference_webcam(detector, camera_id=0):
    """Run inference on webcam"""
    print(f"\nStarting webcam inference (camera {camera_id})")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Draw results
        frame_result = detector.draw_detections(frame, detections)
        
        # Add FPS info
        fps_text = f"FPS: {1/inference_time:.1f}"
        cv2.putText(frame_result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv7 Webcam', frame_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLOv7 Inference Script')
    parser.add_argument('--model', type=str, default='models/yolov7.pt', help='Path to YOLOv7 model')
    parser.add_argument('--source', type=str, default='0', help='Image/video path or webcam (0, 1, ...)')
    parser.add_argument('--output', type=str, default=None, help='Output path for result')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='', help='Device (cpu, cuda, 0, 1, ...)')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("\nAvailable models in 'models' folder:")
        models_dir = Path('models')
        if models_dir.exists():
            for model_file in models_dir.glob('*.pt'):
                print(f"  - {model_file}")
        else:
            print("Models folder not found. Please create 'models' folder and add .pt files.")
        return
    
    # Initialize detector
    detector = YOLOv7Detector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Determine source type
    source = args.source
    show = not args.no_show
    
    if source.isdigit():
        # Webcam
        inference_webcam(detector, int(source))
    elif Path(source).exists():
        if Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Image
            inference_image(detector, source, args.output, show)
        elif Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            # Video
            inference_video(detector, source, args.output, show)
        else:
            print(f"Error: Unsupported file type: {source}")
            print("Supported formats: Images (.jpg, .jpeg, .png, .bmp, .webp)")
            print("                   Videos (.mp4, .avi, .mov, .mkv, .wmv)")
    else:
        print(f"Error: Source not found: {source}")
        print("Please check the file path or use webcam (0, 1, 2, ...)")


if __name__ == '__main__':
    main()
