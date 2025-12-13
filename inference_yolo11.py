"""
YOLO11 Inference Script using Ultralytics
Supports image, video, and webcam inference
"""

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
import time


class YOLO11Detector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, device=''):
        """
        Initialize YOLO11 detector
        
        Args:
            model_path: Path to YOLO11 .pt model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', '0', '1', etc.)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Loading YOLO11 model from {model_path}")
        
        # Load model
        self.model = YOLO(model_path)
        
        # Set device
        if device:
            self.model.to(device)
        
        # Get model info
        self.names = self.model.names
        
        print(f"Model loaded successfully. Classes: {len(self.names)}")
        print(f"Using device: {self.model.device}")
    
    def detect(self, img):
        """
        Run detection on image
        
        Args:
            img: Input image (BGR format or path)
            
        Returns:
            results: YOLO results object
        """
        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        return results[0]
    
    def draw_detections(self, img, result):
        """Draw bounding boxes and labels on image"""
        # Use YOLO's built-in plotting
        annotated_img = result.plot()
        return annotated_img
    
    def get_detections_info(self, result):
        """Get detection information"""
        boxes = result.boxes
        detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class': cls,
                'class_name': self.names[cls]
            })
        
        return detections


def inference_image(detector, image_path, output_path=None, show=True):
    """Run inference on single image"""
    print(f"\nProcessing image: {image_path}")
    
    # Detect
    start_time = time.time()
    result = detector.detect(image_path)
    inference_time = time.time() - start_time
    
    # Get detections info
    detections = detector.get_detections_info(result)
    
    # Draw results
    img_result = detector.draw_detections(None, result)
    
    print(f"Detections: {len(detections)}, Inference time: {inference_time*1000:.2f}ms")
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    
    # Save or show
    if output_path:
        cv2.imwrite(output_path, img_result)
        print(f"Saved result to {output_path}")
    
    if show:
        cv2.imshow('YOLO11 Detection', img_result)
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
        result = detector.detect(frame)
        inference_time = time.time() - start_time
        total_time += inference_time
        frame_count += 1
        
        # Get detection metrics
        detections = detector.get_detections_info(result)
        frame_detections = len(detections)
        total_detections += frame_detections
        
        # Collect confidence scores
        for det in detections:
            confidence_scores.append(det['confidence'])
        
        # Calculate current FPS
        current_fps = 1/inference_time if inference_time > 0 else 0
        fps_history.append(current_fps)
        
        # Draw results
        frame_result = detector.draw_detections(frame, result)
        
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
            cv2.imshow('YOLO11 Detection', frame_result)
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
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect
        start_time = time.time()
        result = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Draw results
        frame_result = detector.draw_detections(frame, result)
        
        # Add FPS info
        fps_text = f"FPS: {1/inference_time:.1f}"
        cv2.putText(frame_result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('YOLO11 Webcam', frame_result)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


def inference_stream(detector, source):
    """Run inference using YOLO's built-in streaming"""
    print(f"\nStarting stream inference from: {source}")
    print("Press 'q' to quit")
    
    # Use YOLO's predict with stream=True for efficient processing
    results = detector.model.predict(
        source=source,
        conf=detector.conf_threshold,
        iou=detector.iou_threshold,
        stream=True,
        show=True,
        verbose=False
    )
    
    frame_count = 0
    for result in results:
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description='YOLO11 Inference Script')
    parser.add_argument('--model', type=str, default='models/yolo11s.pt', help='Path to YOLO11 model')
    parser.add_argument('--source', type=str, default='0', help='Image/video path or webcam (0, 1, ...)')
    parser.add_argument('--output', type=str, default=None, help='Output path for result')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='', help='Device (cpu, cuda, 0, 1, ...)')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    parser.add_argument('--stream', action='store_true', help='Use YOLO streaming mode (faster for video/webcam)')
    
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
    detector = YOLO11Detector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Determine source type
    source = args.source
    show = not args.no_show
    
    # Use streaming mode if requested (more efficient for video/webcam)
    if args.stream and not args.output:
        inference_stream(detector, source)
        return
    
    if source.isdigit():
        # Webcam
        inference_webcam(detector, int(source))
    elif Path(source).exists():
        if Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
            # Image
            inference_image(detector, source, args.output, show)
        elif Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            # Video
            inference_video(detector, source, args.output, show)
        else:
            print(f"Error: Unsupported file type: {source}")
            print("Supported formats: Images (.jpg, .jpeg, .png, .bmp, .webp, .tiff)")
            print("                   Videos (.mp4, .avi, .mov, .mkv, .wmv)")
    else:
        print(f"Error: Source not found: {source}")
        print("Please check the file path or use webcam (0, 1, 2, ...)")


if __name__ == '__main__':
    main()
