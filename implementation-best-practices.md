# Implementation Best Practices and Optimization Tips

## Code Optimization Strategies

### 1. Frame Processing Optimization

#### Reduce Frame Size
```python
# Resize frame for faster processing
def resize_frame(frame, width=640):
    height = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, height))

# In main loop:
frame = resize_frame(frame, 480)  # Smaller = faster
```

#### Skip Frame Processing
```python
# Process every nth frame to improve performance
frame_skip = 2
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % frame_skip == 0:
        # Process frame for drowsiness detection
        pass
    
    # Always display the frame
    cv2.imshow('Detection', frame)
```

### 2. Threading for Camera Capture

```python
import threading
from queue import Queue

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = Queue()
        self.running = True
        
    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        return self
        
    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if not self.q.empty():
                self.q.get()
            self.q.put(frame)
            
    def read(self):
        return self.q.get()
        
    def stop(self):
        self.running = False
        self.thread.join()
```

### 3. Memory Management

```python
# Efficient numpy operations
def calculate_ear_optimized(eye_landmarks):
    eye_landmarks = np.array(eye_landmarks, dtype=np.float32)
    
    # Vectorized distance calculations
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
```

## Advanced Features Implementation

### 1. Multi-Person Detection

```python
def detect_multiple_faces(self):
    """Detect drowsiness for multiple people"""
    faces_data = []
    
    if results.multi_face_landmarks:
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Process each face separately
            person_data = {
                'id': idx,
                'ear': self.calculate_average_ear(face_landmarks),
                'drowsy': False,
                'frame_counter': self.frame_counters.get(idx, 0)
            }
            
            # Update drowsiness status
            if person_data['ear'] < self.EAR_THRESHOLD:
                person_data['frame_counter'] += 1
                if person_data['frame_counter'] >= self.CONSEC_FRAMES:
                    person_data['drowsy'] = True
            else:
                person_data['frame_counter'] = 0
                
            self.frame_counters[idx] = person_data['frame_counter']
            faces_data.append(person_data)
    
    return faces_data
```

### 2. Data Logging and Analytics

```python
import json
from datetime import datetime

class DrowsinessLogger:
    def __init__(self, log_file='drowsiness_log.json'):
        self.log_file = log_file
        self.session_data = []
        
    def log_detection(self, ear_value, is_drowsy, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        log_entry = {
            'timestamp': timestamp,
            'ear_value': ear_value,
            'is_drowsy': is_drowsy,
            'session_id': self.session_id
        }
        
        self.session_data.append(log_entry)
        
    def save_session(self):
        with open(self.log_file, 'a') as f:
            for entry in self.session_data:
                f.write(json.dumps(entry) + '\n')
```

### 3. Adaptive Thresholding

```python
class AdaptiveThreshold:
    def __init__(self, initial_threshold=0.25):
        self.threshold = initial_threshold
        self.ear_history = []
        self.max_history = 100
        
    def update_threshold(self, ear_value):
        self.ear_history.append(ear_value)
        
        if len(self.ear_history) > self.max_history:
            self.ear_history.pop(0)
            
        # Calculate adaptive threshold based on recent history
        if len(self.ear_history) >= 30:
            mean_ear = np.mean(self.ear_history)
            std_ear = np.std(self.ear_history)
            self.threshold = mean_ear - (2 * std_ear)
            
        return self.threshold
```

## Performance Monitoring

### 1. FPS Counter

```python
class FPSCounter:
    def __init__(self):
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
            
    def get_fps(self):
        return self.fps
```

### 2. Resource Usage Monitoring

```python
import psutil

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        
    def get_cpu_usage(self):
        return self.process.cpu_percent()
        
    def get_memory_usage(self):
        return self.process.memory_info().rss / 1024 / 1024  # MB
        
    def log_resources(self):
        cpu = self.get_cpu_usage()
        memory = self.get_memory_usage()
        print(f"CPU: {cpu:.1f}% | Memory: {memory:.1f}MB")
```

## Error Handling and Robustness

### 1. Camera Error Handling

```python
def initialize_camera_with_retry(self, max_retries=3):
    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Test camera with a frame read
                ret, frame = cap.read()
                if ret:
                    print(f"Camera initialized successfully on attempt {attempt + 1}")
                    return cap
                else:
                    cap.release()
                    
        except Exception as e:
            print(f"Camera initialization attempt {attempt + 1} failed: {e}")
            
        time.sleep(1)  # Wait before retry
        
    raise Exception("Failed to initialize camera after maximum retries")
```

### 2. Model Loading Error Handling

```python
def load_model_safely(self, model_path):
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return self.download_model(model_path)
            
        # Validate model file
        file_size = os.path.getsize(model_path)
        if file_size < 1000000:  # Less than 1MB is suspicious
            print("Model file appears corrupted, re-downloading...")
            return self.download_model(model_path)
            
        return dlib.shape_predictor(model_path)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
```

## Configuration Management

### 1. Configuration File

```python
# config.yaml
camera:
  source: 0
  width: 640
  height: 480
  fps: 30

detection:
  ear_threshold: 0.25
  consecutive_frames: 20
  enable_adaptive_threshold: false

alerts:
  enable_audio: true
  enable_visual: true
  alarm_file: "alarm.wav"
  volume: 0.8

logging:
  enable_logging: true
  log_file: "drowsiness_log.json"
  log_level: "INFO"
```

### 2. Configuration Loader

```python
import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get(self, path, default=None):
        """Get config value using dot notation: 'camera.source'"""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
```

## Deployment Considerations

### 1. Executable Creation

```python
# Create executable with PyInstaller
# requirements: pip install pyinstaller

# Command to create executable:
# pyinstaller --onefile --add-data "shape_predictor_68_face_landmarks.dat;." drowsiness_detector.py
```

### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "drowsiness_detector_mediapipe.py"]
```

## Testing and Validation

### 1. Unit Testing

```python
import unittest

class TestDrowsinessDetection(unittest.TestCase):
    def setUp(self):
        self.detector = DrowsinessDetectorMediaPipe()
        
    def test_ear_calculation(self):
        # Test with known eye landmarks
        test_landmarks = [
            [0, 0], [1, -0.5], [2, -0.5],
            [3, 0], [2, 0.5], [1, 0.5]
        ]
        
        ear = self.detector.calculate_ear(test_landmarks)
        self.assertGreater(ear, 0)
        self.assertLess(ear, 1)
        
    def test_threshold_detection(self):
        # Test drowsiness threshold logic
        drowsy_ear = 0.2
        alert_ear = 0.35
        
        self.assertTrue(drowsy_ear < self.detector.EAR_THRESHOLD)
        self.assertFalse(alert_ear < self.detector.EAR_THRESHOLD)

if __name__ == '__main__':
    unittest.main()
```

### 2. Performance Benchmarking

```python
def benchmark_detection_speed(detector, num_frames=100):
    """Benchmark detection speed"""
    times = []
    
    for _ in range(num_frames):
        # Generate dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        # Process frame (mock detection)
        detector.process_frame(frame)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Average processing time: {avg_time:.4f}s")
    print(f"Estimated FPS: {fps:.2f}")
    
    return fps
```

## Production Deployment Tips

1. **Use MediaPipe for production** - Better performance and easier deployment
2. **Implement proper error handling** - Handle camera disconnections gracefully
3. **Add configuration files** - Make system easily configurable
4. **Include logging** - Log important events and errors
5. **Monitor performance** - Track FPS and resource usage
6. **Test thoroughly** - Test with different lighting conditions and camera angles
7. **Provide user documentation** - Include setup and usage instructions
8. **Consider edge cases** - Handle multiple faces, no faces, poor lighting
9. **Implement security measures** - If deploying as a service
10. **Plan for updates** - Design system for easy model and code updates