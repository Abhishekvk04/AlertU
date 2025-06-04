# Installation Requirements and Setup Guide

## Required Dependencies

### Core Libraries
```bash
pip install opencv-python
pip install numpy
pip install scipy
pip install pygame
```

### For Dlib Version
```bash
pip install dlib
```

### For MediaPipe Version
```bash
pip install mediapipe
```

## Additional Requirements for Dlib

### Download Shape Predictor File
For the dlib version, you need to download the facial landmark predictor:

1. **Download the shape predictor file:**
   - Go to: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract the .bz2 file to get `shape_predictor_68_face_landmarks.dat`
   - Place this file in the same directory as your Python script

2. **Alternative download methods:**
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

## Complete Installation Commands

### Option 1: All at once
```bash
pip install opencv-python numpy scipy pygame dlib mediapipe
```

### Option 2: Step by step
```bash
# Basic requirements
pip install opencv-python
pip install numpy
pip install scipy
pip install pygame

# Choose one or both detection methods
pip install dlib          # For dlib-based detection
pip install mediapipe     # For MediaPipe-based detection
```

## System Requirements

### Minimum Requirements
- Python 3.7+
- Webcam or USB camera
- 4GB RAM
- Dual-core processor

### Recommended Requirements
- Python 3.8+
- HD Webcam
- 8GB RAM
- Quad-core processor
- Good lighting conditions

## Troubleshooting Installation

### Dlib Installation Issues

#### Windows
```bash
# If dlib installation fails, try:
pip install cmake
pip install dlib

# Or use conda:
conda install -c conda-forge dlib
```

#### macOS
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake
pip install dlib
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib
```

### OpenCV Issues
```bash
# If OpenCV installation fails:
pip uninstall opencv-python
pip install opencv-python-headless
```

### Camera Access Issues

#### Linux
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Logout and login again
```

#### Windows
- Ensure camera privacy settings allow desktop apps to access camera
- Check Device Manager for camera drivers

#### macOS
- Grant camera permissions in System Preferences > Security & Privacy > Camera

## Project Structure
```
drowsiness_detection/
│
├── drowsiness_detector_dlib.py      # Dlib-based implementation
├── drowsiness_detector_mediapipe.py # MediaPipe-based implementation
├── shape_predictor_68_face_landmarks.dat  # Required for dlib
├── alarm.wav                        # Optional alarm sound file
├── requirements.txt                 # Dependencies list
└── README.md                       # This file
```

## Quick Test

### Test Camera Access
```python
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera is working!")
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()
else:
    print("Camera not found!")
```

### Test Dlib Installation
```python
import dlib
print("Dlib version:", dlib.version)
```

### Test MediaPipe Installation
```python
import mediapipe as mp
print("MediaPipe version:", mp.__version__)
```