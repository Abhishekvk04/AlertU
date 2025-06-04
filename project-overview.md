# Complete Drowsiness Detection Project Overview

## Project Description

This project implements a real-time drowsiness detection system using computer vision techniques. The system monitors a person's eyes through a camera feed and triggers an alert when signs of drowsiness are detected. Two different implementations are provided: one using Dlib and another using MediaPipe.

## How It Works

### Eye Aspect Ratio (EAR) Method

The system uses the Eye Aspect Ratio (EAR) algorithm to detect drowsiness:

1. **Face Detection**: Detect faces in the video frame
2. **Landmark Detection**: Locate facial landmarks, specifically eye landmarks
3. **EAR Calculation**: Calculate the ratio between eye height and width
4. **Drowsiness Detection**: Monitor EAR values over consecutive frames
5. **Alert System**: Trigger alarms when drowsiness is detected

### EAR Formula

```
EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
```

Where P1-P6 are specific eye landmark points:
- P1, P4: Horizontal eye corners
- P2, P3: Upper eyelid points  
- P5, P6: Lower eyelid points

### EAR Thresholds

- **Normal (Alert)**: EAR > 0.25 (typically 0.3-0.4)
- **Drowsy**: EAR < 0.25 (typically 0.2-0.3)
- **Consecutive Frames**: 20 frames below threshold triggers alarm

## Two Implementation Approaches

### 1. Dlib Implementation (`drowsiness_detector_dlib.py`)

**Advantages:**
- Higher accuracy (95-98%)
- Robust facial landmark detection
- Excellent documentation and community support
- Works well in various lighting conditions

**Disadvantages:**
- More complex installation (requires CMake)
- Larger model size (99.7 MB)
- Requires separate shape predictor file download
- Slightly lower FPS performance

**Key Features:**
- Uses HOG + Linear SVM for face detection
- 68-point facial landmark model
- Proven stability and reliability

### 2. MediaPipe Implementation (`drowsiness_detector_mediapipe.py`)

**Advantages:**
- Easy installation (simple pip install)
- Higher FPS performance (25-30 FPS)
- Lower resource usage
- Smaller model size (2.6 MB)
- Better cross-platform support

**Disadvantages:**
- Slightly lower accuracy (92-96%)
- Less robust in challenging conditions
- Newer technology with smaller community

**Key Features:**
- Uses BlazeFace for face detection
- 468-point facial landmark model
- Optimized for real-time applications

## Project Structure

```
drowsiness_detection/
│
├── Core Implementation Files
│   ├── drowsiness_detector_dlib.py      # Dlib-based version
│   └── drowsiness_detector_mediapipe.py # MediaPipe-based version
│
├── Documentation
│   ├── installation-guide.md            # Setup instructions
│   ├── drowsiness-detector-dlib.md      # Dlib implementation guide
│   └── drowsiness-detector-mediapipe.md # MediaPipe implementation guide
│
├── Data Files (for Dlib)
│   └── shape_predictor_68_face_landmarks.dat # Download required
│
├── Optional Assets
│   ├── alarm.wav                        # Custom alarm sound
│   └── drowsiness_detection_comparison.csv # Performance comparison
│
└── Generated Assets
    ├── drowsiness_detection_comparison.png # Performance chart
    └── ear_diagram.png                  # EAR calculation diagram
```

## Quick Start Guide

### Step 1: Install Dependencies

**For MediaPipe version (Recommended for beginners):**
```bash
pip install opencv-python mediapipe numpy scipy pygame
```

**For Dlib version:**
```bash
pip install opencv-python dlib numpy scipy pygame
# Download shape_predictor_68_face_landmarks.dat from http://dlib.net/files/
```

### Step 2: Run the Application

**MediaPipe version:**
```bash
python drowsiness_detector_mediapipe.py
```

**Dlib version:**
```bash
python drowsiness_detector_dlib.py
```

### Step 3: Control the Application

- **'q'**: Quit the application
- **'s'**: Stop the alarm
- **'r'**: Reset the drowsiness counter (MediaPipe version)

## Key Features

### Real-time Detection
- Live camera feed processing
- Immediate drowsiness alerts
- Visual feedback with eye landmark display

### Configurable Parameters
- Adjustable EAR threshold (default: 0.25)
- Customizable consecutive frame count (default: 20)
- Multiple alert mechanisms

### Alert System
- Visual alerts on screen
- Audio alarms (using pygame)
- Console warnings
- System beep fallback

### Performance Monitoring
- Real-time EAR value display
- Frame counter visualization
- Threshold status indication

## Customization Options

### Adjusting Sensitivity
```python
# In the constructor
self.EAR_THRESHOLD = 0.22  # Lower = more sensitive
self.CONSEC_FRAMES = 15    # Lower = faster detection
```

### Custom Alarm Sounds
1. Place your alarm file as `alarm.wav` in the project directory
2. Supported formats: WAV, MP3 (depending on pygame installation)

### Camera Selection
```python
# Change camera index (0 = default, 1 = external camera)
cap = cv2.VideoCapture(1)
```

## Troubleshooting

### Common Issues

1. **Camera not found**
   - Check camera permissions
   - Try different camera indices (0, 1, 2)
   - Ensure no other applications are using the camera

2. **Dlib installation fails**
   - Install CMake first: `pip install cmake`
   - Use conda: `conda install -c conda-forge dlib`

3. **Shape predictor file not found**
   - Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract and place in project directory

4. **Low FPS performance**
   - Try MediaPipe implementation
   - Reduce camera resolution
   - Close other applications

### Performance Optimization

1. **Improve FPS**
   - Use MediaPipe implementation
   - Implement threading for camera capture
   - Reduce frame resolution

2. **Improve Accuracy**
   - Use Dlib implementation
   - Ensure good lighting conditions
   - Position camera at eye level

## Use Cases

### Personal Use
- Monitor alertness during long work sessions
- Stay alert while driving
- Academic research on fatigue detection

### Commercial Applications
- Fleet management systems
- Industrial safety monitoring
- Medical patient monitoring
- Security and surveillance systems

## Future Enhancements

### Possible Improvements
1. **Multi-person detection**
2. **Head pose estimation**
3. **Yawn detection**
4. **Heart rate monitoring**
5. **Mobile app development**
6. **Cloud-based analytics**

### Advanced Features
1. **Machine learning model training**
2. **Data logging and analytics**
3. **Integration with IoT devices**
4. **Remote monitoring capabilities**

## Technical Specifications

### System Requirements
- **Operating System**: Windows, macOS, Linux
- **Python**: 3.7 or higher
- **Camera**: Any USB webcam or built-in camera
- **RAM**: Minimum 4GB (8GB recommended)
- **Processor**: Dual-core (Quad-core recommended)

### Performance Metrics
- **Dlib**: 15-20 FPS, 95-98% accuracy
- **MediaPipe**: 25-30 FPS, 92-96% accuracy
- **Latency**: < 100ms detection time
- **Resource Usage**: Low to medium CPU usage

## Conclusion

This drowsiness detection system provides a robust, real-time solution for monitoring alertness using computer vision. The dual implementation approach allows users to choose between accuracy (Dlib) and performance (MediaPipe) based on their specific needs. The system is designed to be easily customizable and can be integrated into larger safety systems.

Choose **MediaPipe** for:
- Easy setup and deployment
- Real-time applications
- Resource-constrained environments
- Cross-platform compatibility

Choose **Dlib** for:
- Maximum accuracy
- Research applications
- Robust performance in challenging conditions
- When model size is not a concern