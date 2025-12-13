"""
YOLOv7 Inference Script using Official YOLOv7 Repository
Supports image, video, and webcam inference
"""

import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import subprocess
import threading

# Add YOLOv7 source to path
YOLOV7_PATH = Path(__file__).parent / 'yolov7-main'
if YOLOV7_PATH.exists():
    sys.path.insert(0, str(YOLOV7_PATH))
    
    # Import YOLOv7 modules
    try:
        from models.experimental import attempt_load
        from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
        from utils.plots import plot_one_box
        from utils.torch_utils import select_device, time_synchronized
        from utils.datasets import letterbox
        YOLOV7_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import YOLOv7 modules: {e}")
        print("Make sure yolov7-main folder exists and contains the YOLOv7 source code")
        YOLOV7_AVAILABLE = False
else:
    print(f"Warning: YOLOv7 source not found at {YOLOV7_PATH}")
    print("Please ensure yolov7-main folder exists in the same directory as this script")
    YOLOV7_AVAILABLE = False


def get_jetson_gpu_stats():
    """Get Jetson GPU utilization and memory usage"""
    try:
        # Try to get GPU stats using tegrastats (Jetson-specific)
        result = subprocess.run(['tegrastats', '--interval', '100'], 
                              capture_output=True, text=True, timeout=0.2)
        if result.returncode == 0:
            # Parse tegrastats output for GPU usage
            lines = result.stdout.strip().split('\n')
            if lines:
                line = lines[-1]  # Get last line
                if 'GR3D_FREQ' in line:
                    # Extract GPU frequency and usage
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'GR3D_FREQ' in part and i + 1 < len(parts):
                            gpu_freq = parts[i + 1].replace('%', '')
                            return {'gpu_util': gpu_freq, 'available': True}
    except:
        pass
    
    # Fallback to nvidia-ml-py if available
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'gpu_util': f"{gpu_util.gpu}%",
            'mem_used': f"{mem_info.used / 1024**2:.0f}MB",
            'mem_total': f"{mem_info.total / 1024**2:.0f}MB",
            'available': True
        }
    except:
        pass
    
    # Basic PyTorch GPU info
    if torch.cuda.is_available():
        try:
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            return {
                'mem_allocated': f"{mem_allocated:.0f}MB",
                'mem_reserved': f"{mem_reserved:.0f}MB",
                'available': True
            }
        except:
            pass
    
    return {'available': False}


class YOLOv7Detector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, device=''):
        """
        Initialize YOLOv7 detector optimized for Jetson Nano Orin GPU
        
        Args:
            model_path: Path to YOLOv7 .pt model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', '0', '1', etc.)
        """
        if not YOLOV7_AVAILABLE:
            raise ImportError("YOLOv7 modules not available. Please ensure yolov7-main folder exists.")
            
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Jetson Nano Orin GPU optimization
        if device == '' and torch.cuda.is_available():
            device = '0'  # Force GPU usage on Jetson
            print("Jetson GPU detected, forcing CUDA device 0")
        
        # Set device using YOLOv7's select_device function
        self.device = select_device(device)
        
        # Jetson-specific CUDA optimizations
        if self.device.type != 'cpu':
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
            print(f"CUDA optimizations enabled for Jetson")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print(f"Loading YOLOv7 model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model using YOLOv7's attempt_load function
        self.model = attempt_load(model_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.img_size = 640
        
        # Check image size
        self.img_size = check_img_size(self.img_size, s=self.stride)
        
        # Jetson optimization: Set model to half precision if GPU available
        if self.device.type != 'cpu':
            try:
                self.model.half()  # Convert to FP16 for better Jetson performance
                self.use_half = True
                print("Model converted to FP16 for Jetson optimization")
            except:
                self.use_half = False
                print("FP16 conversion failed, using FP32")
        else:
            self.use_half = False
        
        # Warm up the model
        if self.device.type != 'cpu':
            print("Warming up GPU...")
            dummy_input = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
            if self.use_half:
                dummy_input = dummy_input.half()
            with torch.no_grad():
                _ = self.model(dummy_input)
            print("GPU warm-up complete")
        
        print(f"Model loaded successfully. Classes: {len(self.names)}")
        print(f"Image size: {self.img_size}, Stride: {self.stride}")
        print(f"Half precision: {self.use_half}")
    
    def preprocess(self, img0):
        """Preprocess image for inference using YOLOv7's letterbox function with Jetson optimization"""
        # Letterbox
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).to(self.device, non_blocking=True)  # Non-blocking transfer for Jetson
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Convert to half precision if enabled
        if self.use_half:
            img = img.half()
            
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img
    
    def detect(self, img0):
        """
        Run detection on image using YOLOv7's official functions with Jetson GPU optimization
        
        Args:
            img0: Input image (BGR format)
            
        Returns:
            detections: List of detections [x1, y1, x2, y2, conf, cls]
        """
        # Preprocess
        img = self.preprocess(img0)
        
        # Inference with GPU optimization
        with torch.no_grad():
            if self.device.type != 'cpu':
                # Use CUDA stream for better performance on Jetson
                with torch.cuda.device(self.device):
                    pred = self.model(img, augment=False)[0]
            else:
                pred = self.model(img, augment=False)[0]
        
        # Apply NMS using YOLOv7's function
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=None, agnostic=False)
        
        # Process detections
        detections = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                detections = det.cpu().numpy()
        
        return detections
    
    def draw_detections(self, img, detections):
        """Draw bounding boxes and labels on image using YOLOv7's plot_one_box"""
        if YOLOV7_AVAILABLE:
            # Use YOLOv7's plotting function
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                label = f'{self.names[cls]} {conf:.2f}'
                plot_one_box([x1, y1, x2, y2], img, label=label, color=self.get_color(cls), line_thickness=2)
        else:
            # Fallback to basic drawing
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
        
        # Print detailed progress with GPU stats
        if frame_count % 30 == 0:
            avg_fps_recent = sum(fps_history[-30:]) / min(30, len(fps_history))
            gpu_stats = get_jetson_gpu_stats()
            gpu_info = ""
            if gpu_stats['available']:
                if 'gpu_util' in gpu_stats:
                    gpu_info = f" | GPU: {gpu_stats['gpu_util']}"
                if 'mem_allocated' in gpu_stats:
                    gpu_info += f" | GPU Mem: {gpu_stats['mem_allocated']}"
            print(f"Frame {frame_count}/{total_frames} | FPS: {current_fps:.1f} (avg: {avg_fps_recent:.1f}) | Detections: {frame_detections}{gpu_info} | Progress: {progress*100:.1f}%")
    
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
    parser = argparse.ArgumentParser(description='YOLOv7 Inference Script using Official Repository')
    parser.add_argument('--model', type=str, default='models/yolov7.pt', help='Path to YOLOv7 model')
    parser.add_argument('--source', type=str, default='0', help='Image/video path or webcam (0, 1, ...)')
    parser.add_argument('--output', type=str, default=None, help='Output path for result')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='', help='Device (cpu, cuda, 0, 1, ...)')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    
    args = parser.parse_args()
    
    # Check if YOLOv7 source is available
    if not YOLOV7_AVAILABLE:
        print("Error: YOLOv7 source code not found!")
        print(f"Please ensure 'yolov7-main' folder exists at: {YOLOV7_PATH}")
        print("\nTo fix this:")
        print("1. Download YOLOv7 from: https://github.com/WongKinYiu/yolov7")
        print("2. Extract to 'yolov7-main' folder in the same directory as this script")
        print("3. Install YOLOv7 requirements: pip install -r yolov7-main/requirements.txt")
        return
    
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
        print("\nYou can download YOLOv7 models from:")
        print("https://github.com/WongKinYiu/yolov7/releases")
        return
    
    try:
        # Initialize detector
        detector = YOLOv7Detector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing YOLOv7 detector: {e}")
        print("Make sure the model file is compatible with YOLOv7")
        return
    
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
