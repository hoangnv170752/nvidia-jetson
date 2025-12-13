# Installation Guide for Python 3.8.10

This guide provides step-by-step instructions for installing YOLO inference dependencies on Python 3.8.10.

## Method 1: Install Updated Requirements (Recommended)

```bash
# Update pip first
python3 -m pip install --upgrade pip

# Install requirements with updated versions
pip install -r requirements.txt
```

## Method 2: Manual Installation (If Method 1 fails)

### Step 1: Install Core Dependencies
```bash
# Install numpy first (required by other packages)
pip install "numpy>=1.19.0,<1.25.0"

# Install OpenCV
pip install "opencv-python>=4.5.0,<4.9.0"

# Install Pillow
pip install "Pillow>=8.0.0,<10.0.0"
```

### Step 2: Install PyTorch for YOLOv7
```bash
# For CPU only
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.6 (if you have GPU)
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu116

# For CUDA 11.7 (if you have GPU)
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
```

### Step 3: Install Ultralytics for YOLO11
```bash
pip install "ultralytics>=8.0.0,<8.1.0"
```

### Step 4: Install Additional Dependencies
```bash
pip install "matplotlib>=3.3.0,<3.8.0"
pip install "PyYAML>=5.4.0,<7.0.0"
pip install "requests>=2.25.0,<3.0.0"
pip install "scipy>=1.6.0,<1.11.0"
pip install "tqdm>=4.60.0,<5.0.0"
```

## Method 3: Conda Installation (Alternative)

If you're using Conda, you can create a new environment:

```bash
# Create new environment
conda create -n yolo-jetson python=3.8.10

# Activate environment
conda activate yolo-jetson

# Install packages via conda
conda install numpy=1.24.3 opencv matplotlib pyyaml requests scipy tqdm

# Install PyTorch
conda install pytorch=1.13.1 torchvision=0.14.1 cpuonly -c pytorch

# Install ultralytics via pip
pip install ultralytics==8.0.196
```

## Troubleshooting

### Error: "No matching distribution found"
```bash
# Update pip and setuptools
python3 -m pip install --upgrade pip setuptools wheel

# Try installing with --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

### Error: "Microsoft Visual C++ 14.0 is required" (Windows)
```bash
# Install packages one by one
pip install numpy opencv-python
pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch torchvision
pip install ultralytics
```

### Error: "Failed building wheel"
```bash
# Install pre-compiled wheels
pip install --only-binary=all -r requirements.txt
```

## Verification

After installation, verify everything works:

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
```

## Test Inference

```bash
# Test YOLO11
python3 inference_yolo11.py --model models/yolo11s-detect.pt --source 0

# Test YOLOv7 (if you have a YOLOv7 model)
python3 inference_yolov7.py --model models/yolov7.pt --source 0
```

## Notes for Jetson Nano

If you're on Jetson Nano, consider these optimizations:

1. **Use JetPack PyTorch**: Install PyTorch from NVIDIA's pre-built wheels
2. **Reduce Model Size**: Use smaller models (yolo11n, yolo11s)
3. **Optimize OpenCV**: Use OpenCV compiled with CUDA support

### Jetson-specific Installation
```bash
# Install PyTorch for Jetson (if available)
wget https://nvidia.box.com/shared/static/rehpfc4dwsxuhpv4jgqou6q8rm55jbsz.whl -O torch-1.13.0-cp38-cp38-linux_aarch64.whl
pip install torch-1.13.0-cp38-cp38-linux_aarch64.whl

# Install torchvision
pip install torchvision==0.14.0
```
