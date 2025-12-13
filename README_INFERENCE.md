# YOLO Inference Scripts

Scripts để chạy inference với YOLOv7 và YOLO11 trên NVIDIA Jetson.

## Cài đặt

### YOLOv7
```bash
pip install torch torchvision opencv-python numpy
```

### YOLO11
```bash
pip install ultralytics opencv-python
```

## Cấu trúc thư mục

```
NVIDIA-Jetson/
├── models/
│   ├── yolov7.pt          # Đặt model YOLOv7 vào đây
│   ├── yolo11s.pt         # Đặt model YOLO11 vào đây
│   └── ...
├── inference_yolov7.py
├── inference_yolo11.py
└── README_INFERENCE.md
```

## Sử dụng

### YOLOv7

#### 1. Inference trên ảnh
```bash
python inference_yolov7.py --model models/yolov7.pt --source image.jpg --output result.jpg
```

#### 2. Inference trên video
```bash
python inference_yolov7.py --model models/yolov7.pt --source video.mp4 --output result.mp4
```

#### 3. Inference trên webcam
```bash
python inference_yolov7.py --model models/yolov7.pt --source 0
```

#### 4. Tùy chỉnh thông số
```bash
python inference_yolov7.py \
    --model models/yolov7.pt \
    --source image.jpg \
    --conf 0.5 \
    --iou 0.45 \
    --device cuda \
    --output result.jpg
```

### YOLO11

#### 1. Inference trên ảnh
```bash
python inference_yolo11.py --model models/yolo11s.pt --source image.jpg --output result.jpg
```

#### 2. Inference trên video
```bash
python inference_yolo11.py --model models/yolo11s.pt --source video.mp4 --output result.mp4
```

#### 3. Inference trên webcam
```bash
python inference_yolo11.py --model models/yolo11s.pt --source 0
```

#### 4. Sử dụng streaming mode (nhanh hơn cho video/webcam)
```bash
python inference_yolo11.py --model models/yolo11s.pt --source 0 --stream
```

#### 5. Tùy chỉnh thông số
```bash
python inference_yolo11.py \
    --model models/yolo11s-detect.pt \
    --source image.jpg \
    --conf 0.5 \
    --iou 0.45 \
    --device cuda \
    --output result.jpg
```

## Tham số

### Chung cho cả hai scripts

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--model` | Đường dẫn đến file model .pt | `models/yolov7.pt` hoặc `models/yolo11s.pt` |
| `--source` | Nguồn input (ảnh/video/webcam) | `0` (webcam) |
| `--output` | Đường dẫn lưu kết quả | `None` (không lưu) |
| `--conf` | Ngưỡng confidence | `0.25` |
| `--iou` | Ngưỡng IOU cho NMS | `0.45` |
| `--device` | Device để chạy (cpu/cuda/0/1) | `''` (tự động) |
| `--no-show` | Không hiển thị kết quả | `False` |

### Riêng cho YOLO11

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--stream` | Sử dụng streaming mode | `False` |

## Ví dụ với models hiện có

```bash
# YOLO11 - Model 1
python inference_yolo11.py --model models/yolo11s-1411.pt --source 0

# YOLO11 - Model 2
python inference_yolo11.py --model models/yolo11s-detect.pt --source test.jpg --output result.jpg
```

## Lưu ý

1. **YOLOv7**: Cần file model YOLOv7 chính thức (.pt). Script này implement inference từ đầu.
2. **YOLO11**: Sử dụng thư viện Ultralytics, đơn giản và tối ưu hơn.
3. **Performance**: YOLO11 thường nhanh hơn và dễ sử dụng hơn nhờ thư viện Ultralytics.
4. **Webcam**: Nhấn 'q' để thoát khi chạy webcam hoặc video.
5. **Device**: Trên Jetson, để trống `--device` để tự động chọn GPU.

## Troubleshooting

### Lỗi "Model file not found"
- Kiểm tra đường dẫn model có đúng không
- Đảm bảo file .pt đã được đặt trong thư mục `models/`

### Lỗi CUDA out of memory
- Giảm kích thước input hoặc sử dụng model nhỏ hơn
- Thử với `--device cpu`

### FPS thấp
- Đảm bảo đang sử dụng GPU: kiểm tra output "Using device"
- Với YOLO11, thử `--stream` mode cho video/webcam
- Giảm độ phân giải input
