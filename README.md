# NVIDIA Jetson YOLO Inference

YOLO inference scripts optimized for NVIDIA Jetson devices. Supports YOLOv7 and YOLO11 models for real-time object detection on images, videos, and webcam streams.

## Features

- **YOLOv7**: Custom implementation with full control over inference pipeline
- **YOLO11**: Ultralytics-based implementation with streaming support
- **Multiple Input Sources**: Images, videos, webcam
- **Jetson Optimized**: GPU acceleration support for NVIDIA Jetson devices
- **Flexible Output**: Save results or display in real-time

## Installation

### Prerequisites
```bash
# For Jetson devices, ensure CUDA is properly installed
# Check CUDA installation
nvcc --version
```

### YOLOv7 Dependencies
```bash
pip install torch torchvision opencv-python numpy
```

### YOLO11 Dependencies
```bash
pip install ultralytics opencv-python
```

## Quick Start

### 1. Setup Models
Create a `models/` directory and place your YOLO model files:
```bash
mkdir models
# Place your .pt model files in the models/ directory
```

### 2. Run Inference

#### YOLO11 (Recommended)
```bash
# Webcam inference
python inference_yolo11.py --model models/yolo11s-detect.pt --source 0

# Video inference with real examples
python inference_yolo11.py --model models/yolo11s-1411.pt --source S018C001P008R001A081_rgb.avi --output result_081.mp4
python inference_yolo11.py --model models/yolo11s-detect.pt --source S018C001P008R001A082_rgb.avi --output result_082.mp4

# Image inference
python inference_yolo11.py --model models/yolo11s.pt --source image.jpg --output result.jpg

# Streaming mode (faster for video/webcam)
python inference_yolo11.py --model models/yolo11s-detect.pt --source 0 --stream
```

#### YOLOv7
```bash
# Webcam inference
python inference_yolov7.py --model models/yolov7.pt --source 0

# Video inference with real examples
python inference_yolov7.py --model models/yolov7.pt --source S018C001P008R001A083_rgb.avi --output result_083.mp4
python inference_yolov7.py --model models/yolov7.pt --source S018C001P008R001A084_rgb.avi --output result_084.mp4
```

## Usage Examples

### Basic Commands with Real Video Files
```bash
# Use specific confidence threshold with real video
python inference_yolo11.py --model models/yolo11s-detect.pt --source S018C001P008R001A085_rgb.avi --conf 0.5

# Force CPU usage
python inference_yolo11.py --model models/yolo11s-1411.pt --source S018C001P008R001A086_rgb.avi --device cpu

# No display (headless mode) - process multiple videos
python inference_yolo11.py --model models/yolo11s-detect.pt --source S018C001P008R001A087_rgb.avi --no-show --output result_087.mp4
python inference_yolo11.py --model models/yolo11s-detect.pt --source S018C001P008R001A088_rgb.avi --no-show --output result_088.mp4
```

### Advanced Configuration
```bash
# Process video with comprehensive metrics
python inference_yolo11.py \
    --model models/yolo11s-1411.pt \
    --source S018C001P008R001A081_rgb.avi \
    --output result_081_detailed.mp4 \
    --conf 0.25 \
    --iou 0.45 \
    --device 0 \
    --stream

# Batch processing multiple videos
python inference_yolov7.py --model models/yolov7.pt --source S018C001P008R001A082_rgb.avi --output yolov7_082.mp4
python inference_yolov7.py --model models/yolov7.pt --source S018C001P008R001A083_rgb.avi --output yolov7_083.mp4
```

## Parameters

| Parameter | Description | Default | YOLO11 | YOLOv7 |
|-----------|-------------|---------|--------|--------|
| `--model` | Path to model file (.pt) | `models/yolo11s.pt` or `models/yolov7.pt` | ✓ | ✓ |
| `--source` | Input source (image/video/webcam) | `0` | ✓ | ✓ |
| `--output` | Output file path | `None` | ✓ | ✓ |
| `--conf` | Confidence threshold | `0.25` | ✓ | ✓ |
| `--iou` | IOU threshold for NMS | `0.45` | ✓ | ✓ |
| `--device` | Device (cpu/cuda/0/1) | `''` (auto) | ✓ | ✓ |
| `--no-show` | Disable display | `False` | ✓ | ✓ |
| `--stream` | Use streaming mode | `False` | ✓ | ✗ |

## Supported Formats

### Input Formats
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`
- **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- **Webcam**: `0`, `1`, `2`, ... (camera index)

### Output Formats
- **Images**: Same as input format
- **Videos**: `.mp4` (H.264 codec)

## Video Inference Metrics

Both scripts now display comprehensive real-time metrics during video processing:

### On-Screen Display
- **FPS**: Current inference speed (frames per second)
- **Detections**: Number of objects detected in current frame
- **Frame Counter**: Current frame / Total frames
- **Average Confidence**: Mean confidence score of detections
- **Progress Bar**: Visual progress indicator

### Console Output
```
Frame 30/1500 | FPS: 12.3 (avg: 11.8) | Detections: 3 | Progress: 2.0%
Frame 60/1500 | FPS: 13.1 (avg: 12.1) | Detections: 2 | Progress: 4.0%
...
==================================================
VIDEO PROCESSING COMPLETE
==================================================
Total frames processed: 1500
Total processing time: 125.45s
Average FPS: 11.96
Total detections: 4523
Average detections per frame: 3.02
Average confidence score: 0.847
Min FPS: 8.2
Max FPS: 15.4
Output saved to: result_081.mp4
==================================================
```

### Example with Real Videos
```bash
# Process action recognition video with detailed metrics
python inference_yolo11.py --model models/yolo11s-1411.pt --source S018C001P008R001A081_rgb.avi --output detailed_081.mp4

# Compare performance between models
python inference_yolo11.py --model models/yolo11s-detect.pt --source S018C001P008R001A082_rgb.avi --output yolo11_082.mp4
python inference_yolov7.py --model models/yolov7.pt --source S018C001P008R001A082_rgb.avi --output yolov7_082.mp4
```

## Performance Tips

### For Jetson Devices
1. **Use GPU**: Leave `--device` empty for automatic GPU selection
2. **Streaming Mode**: Use `--stream` with YOLO11 for better video performance
3. **Model Selection**: Smaller models (yolo11n, yolo11s) for real-time performance
4. **Resolution**: Lower input resolution for higher FPS

### Memory Optimization
```bash
# Use smaller model
python inference_yolo11.py --model models/yolo11n.pt --source 0

# Reduce confidence threshold to filter detections
python inference_yolo11.py --model models/yolo11s.pt --source 0 --conf 0.5
```

## Troubleshooting

### Common Issues

**Model not found**
```
Error: Model file not found: models/yolo11s.pt
```
- Ensure model files are in the `models/` directory
- Check file path and permissions

**CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
- Use smaller model or reduce input resolution
- Try CPU inference: `--device cpu`

**Low FPS performance**
```
FPS: 2.3
```
- Verify GPU usage in output logs
- Use streaming mode: `--stream` (YOLO11 only)
- Try smaller model or lower resolution

**Camera not found**
```
Error: Could not open camera 0
```
- Check camera permissions
- Try different camera index: `--source 1`
- Verify camera is not used by other applications

## Development

### Repository Structure
```
nvidia-jetson/
├── models/                 # Place .pt model files here
├── inference_yolo11.py    # YOLO11 inference script
├── inference_yolov7.py    # YOLOv7 inference script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Jetson device
5. Submit a pull request

## License

This project is open source. Please check individual model licenses for commercial use.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for YOLO11 implementation
- [YOLOv7](https://github.com/WongKinYiu/yolov7) for the original YOLOv7 model
- NVIDIA for Jetson platform support
