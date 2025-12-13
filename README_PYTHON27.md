# YOLO Inference trên Python 2.7

## ⚠️ Lưu ý quan trọng

Python 2.7 đã **End of Life** từ 01/01/2020 và hầu hết các thư viện deep learning hiện đại **KHÔNG hỗ trợ** Python 2.7:

- **PyTorch**: Phiên bản cuối cùng hỗ trợ Python 2.7 là PyTorch 1.4.0 (2020)
- **Ultralytics YOLO**: KHÔNG hỗ trợ Python 2.7
- **YOLOv7**: KHÔNG hỗ trợ Python 2.7 (yêu cầu Python >= 3.7)
- **YOLO11**: KHÔNG hỗ trợ Python 2.7 (yêu cầu Python >= 3.8)

## Giải pháp thay thế

### Option 1: Nâng cấp Python (Khuyến nghị)

```bash
# Cài đặt Python 3.8+
sudo apt-get update
sudo apt-get install python3.8 python3-pip

# Tạo virtual environment
python3.8 -m venv yolo_env
source yolo_env/bin/activate

# Cài đặt dependencies
pip install ultralytics opencv-python torch torchvision
```

### Option 2: Sử dụng YOLOv3/YOLOv4 với Darknet (C++)

Nếu bắt buộc phải dùng Python 2.7, bạn cần sử dụng phiên bản YOLO cũ hơn với Darknet framework:

#### Cài đặt Darknet

```bash
# Clone Darknet repository
git clone https://github.com/AlexeyAB/darknet
cd darknet

# Compile (chỉnh sửa Makefile nếu cần GPU)
make

# Download YOLOv4 weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

#### Script Python 2.7 cho YOLOv3/v4

Tạo file `inference_yolov4_py27.py`:

```python
# -*- coding: utf-8 -*-
"""
YOLOv4 Inference Script for Python 2.7
Requires: OpenCV 3.x compiled with Python 2.7 support
"""

import cv2
import numpy as np
import argparse
import time


class YOLOv4Detector:
    def __init__(self, weights_path, config_path, names_path, conf_threshold=0.25, nms_threshold=0.45):
        """
        Initialize YOLOv4 detector using OpenCV DNN module
        
        Args:
            weights_path: Path to .weights file
            config_path: Path to .cfg file
            names_path: Path to .names file
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        print("Loading YOLOv4 model...")
        
        # Load network
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Try to use GPU if available
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA backend")
        except:
            print("Using CPU backend")
        
        # Load class names
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        print("Model loaded successfully. Classes: {}".format(len(self.classes)))
    
    def detect(self, image):
        """
        Run detection on image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            boxes, confidences, class_ids
        """
        height, width = image.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])
        
        return final_boxes, final_confidences, final_class_ids
    
    def draw_detections(self, image, boxes, confidences, class_ids):
        """Draw bounding boxes on image"""
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}".format(label, confidence)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image


def inference_image(detector, image_path, output_path=None, show=True):
    """Run inference on image"""
    print("\nProcessing image: {}".format(image_path))
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image {}".format(image_path))
        return
    
    # Detect
    start_time = time.time()
    boxes, confidences, class_ids = detector.detect(image)
    inference_time = time.time() - start_time
    
    # Draw results
    result_image = detector.draw_detections(image.copy(), boxes, confidences, class_ids)
    
    print("Detections: {}, Inference time: {:.2f}ms".format(len(boxes), inference_time * 1000))
    
    # Save or show
    if output_path:
        cv2.imwrite(output_path, result_image)
        print("Saved result to {}".format(output_path))
    
    if show:
        cv2.imshow('YOLOv4 Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def inference_video(detector, video_path, output_path=None, show=True):
    """Run inference on video"""
    print("\nProcessing video: {}".format(video_path))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video {}".format(video_path))
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        start_time = time.time()
        boxes, confidences, class_ids = detector.detect(frame)
        inference_time = time.time() - start_time
        total_time += inference_time
        frame_count += 1
        
        # Draw results
        result_frame = detector.draw_detections(frame, boxes, confidences, class_ids)
        
        # Add FPS info
        fps_text = "FPS: {:.1f}".format(1.0 / inference_time if inference_time > 0 else 0)
        cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write or show
        if writer:
            writer.write(result_frame)
        
        if show:
            cv2.imshow('YOLOv4 Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\nProcessed {} frames".format(frame_count))
    print("Average FPS: {:.2f}".format(avg_fps))
    if output_path:
        print("Saved result to {}".format(output_path))


def inference_webcam(detector, camera_id=0):
    """Run inference on webcam"""
    print("\nStarting webcam inference (camera {})".format(camera_id))
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera {}".format(camera_id))
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        start_time = time.time()
        boxes, confidences, class_ids = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Draw results
        result_frame = detector.draw_detections(frame, boxes, confidences, class_ids)
        
        # Add FPS info
        fps_text = "FPS: {:.1f}".format(1.0 / inference_time if inference_time > 0 else 0)
        cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv4 Webcam', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLOv4 Inference Script for Python 2.7')
    parser.add_argument('--weights', type=str, default='yolov4.weights', help='Path to weights file')
    parser.add_argument('--config', type=str, default='yolov4.cfg', help='Path to config file')
    parser.add_argument('--names', type=str, default='coco.names', help='Path to names file')
    parser.add_argument('--source', type=str, default='0', help='Image/video path or webcam (0, 1, ...)')
    parser.add_argument('--output', type=str, default=None, help='Output path for result')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS threshold')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOv4Detector(
        weights_path=args.weights,
        config_path=args.config,
        names_path=args.names,
        conf_threshold=args.conf,
        nms_threshold=args.nms
    )
    
    # Determine source type
    source = args.source
    show = not args.no_show
    
    if source.isdigit():
        # Webcam
        inference_webcam(detector, int(source))
    elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Image
        inference_image(detector, source, args.output, show)
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video
        inference_video(detector, source, args.output, show)
    else:
        print("Error: Unsupported source type: {}".format(source))


if __name__ == '__main__':
    main()
```

#### Cài đặt OpenCV cho Python 2.7

```bash
# Option 1: Từ package manager (nếu có)
sudo apt-get install python-opencv

# Option 2: Compile từ source (khuyến nghị cho Jetson)
# Download OpenCV 3.4.x (phiên bản cuối hỗ trợ Python 2.7)
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.16.zip
unzip opencv.zip
cd opencv-3.4.16
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D PYTHON2_EXECUTABLE=/usr/bin/python2.7 \
      -D PYTHON2_INCLUDE_DIR=/usr/include/python2.7 \
      -D PYTHON2_LIBRARY=/usr/lib/python2.7/config/libpython2.7.so \
      ..

make -j4
sudo make install
```

#### Sử dụng

```bash
# Inference trên ảnh
python2.7 inference_yolov4_py27.py \
    --weights yolov4.weights \
    --config yolov4.cfg \
    --names coco.names \
    --source image.jpg \
    --output result.jpg

# Inference trên webcam
python2.7 inference_yolov4_py27.py \
    --weights yolov4.weights \
    --config yolov4.cfg \
    --names coco.names \
    --source 0

# Inference trên video
python2.7 inference_yolov4_py27.py \
    --weights yolov4.weights \
    --config yolov4.cfg \
    --names coco.names \
    --source video.mp4 \
    --output result.avi
```

### Option 3: Sử dụng TensorRT (Khuyến nghị cho Jetson)

Nếu bạn đang dùng NVIDIA Jetson, TensorRT là lựa chọn tốt nhất:

```bash
# Convert YOLO model sang TensorRT engine
# Sử dụng Python 3 để convert
python3 -m pip install tensorrt

# Sau đó có thể inference từ Python 2.7 qua ctypes
```

## Khuyến nghị cuối cùng

**Nên nâng cấp lên Python 3.8+** vì:

1. ✅ Hỗ trợ đầy đủ YOLOv7, YOLO11
2. ✅ Performance tốt hơn
3. ✅ Bảo mật tốt hơn
4. ✅ Cộng đồng hỗ trợ
5. ✅ Tương thích với các thư viện mới

Nếu hệ thống yêu cầu Python 2.7, hãy cân nhắc:
- Chạy inference service riêng với Python 3
- Giao tiếp qua REST API hoặc gRPC
- Code Python 2.7 gọi service này

## Liên hệ

Nếu cần hỗ trợ thêm về việc migrate lên Python 3 hoặc setup inference service, hãy cho tôi biết!
