# Drowsiness Detection using MediaPipe

# ```python
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
import time

# Initialize pygame mixer for alarm
pygame.mixer.init()

class DrowsinessDetectorMediaPipe:
    def __init__(self):
        # Eye Aspect Ratio (EAR) threshold and consecutive frame count
        self.EAR_THRESHOLD = 0.25
        self.CONSEC_FRAMES = 20
        
        # Initialize counters
        self.frame_counter = 0
        self.alarm_on = False
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define eye landmark indices for MediaPipe (468 landmarks model)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Simplified eye indices for EAR calculation (6 points per eye)
        self.LEFT_EYE_EAR = [33, 159, 158, 133, 153, 145]  # Left eye landmarks
        self.RIGHT_EYE_EAR = [362, 380, 374, 263, 386, 385]  # Right eye landmarks
        
        # Load alarm sound
        self.load_alarm_sound()
        
    def load_alarm_sound(self):
        """Load or create alarm sound"""
        try:
            pygame.mixer.music.load("alarm.wav")
        except:
            print("Alarm sound file not found. Using system beep.")
            self.use_system_beep = True
            
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR) for MediaPipe landmarks"""
        # Calculate vertical distances
        A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Calculate horizontal distance
        C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_landmarks(self, landmarks, eye_indices, image_width, image_height):
        """Extract eye landmarks from MediaPipe face landmarks"""
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            eye_points.append([x, y])
        return eye_points
    
    def draw_eye_landmarks(self, frame, eye_landmarks):
        """Draw eye landmarks on frame"""
        eye_landmarks = np.array(eye_landmarks, dtype=np.int32)
        
        # Draw points
        for (x, y) in eye_landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
        # Draw eye contour
        cv2.polylines(frame, [eye_landmarks], True, (0, 255, 0), 1)
    
    def trigger_alarm(self):
        """Trigger drowsiness alarm"""
        if not self.alarm_on:
            self.alarm_on = True
            try:
                pygame.mixer.music.play(-1)  # Play alarm continuously
            except:
                # Fallback to system beep and console output
                print("\a" * 3)  # System beep
                print("🚨 WAKE UP! DROWSINESS DETECTED! 🚨")
                
    def stop_alarm(self):
        """Stop drowsiness alarm"""
        if self.alarm_on:
            self.alarm_on = False
            try:
                pygame.mixer.music.stop()
            except:
                pass
    
    def detect_drowsiness(self):
        """Main drowsiness detection loop"""
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        print("MediaPipe Drowsiness Detection Started...")
        print("Press 'q' to quit")
        print("Press 's' to stop alarm")
        print("Press 'r' to reset counter")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get frame dimensions
            height, width, _ = frame.shape
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract eye landmarks
                    left_eye = self.extract_eye_landmarks(
                        face_landmarks.landmark, self.LEFT_EYE_EAR, width, height
                    )
                    right_eye = self.extract_eye_landmarks(
                        face_landmarks.landmark, self.RIGHT_EYE_EAR, width, height
                    )
                    
                    # Calculate EAR for both eyes
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)
                    
                    # Average EAR
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Draw eye landmarks
                    self.draw_eye_landmarks(frame, left_eye)
                    self.draw_eye_landmarks(frame, right_eye)
                    
                    # Check for drowsiness
                    if avg_ear < self.EAR_THRESHOLD:
                        self.frame_counter += 1
                        
                        # Display warning
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Trigger alarm if drowsy for consecutive frames
                        if self.frame_counter >= self.CONSEC_FRAMES:
                            self.trigger_alarm()
                            cv2.putText(frame, "WAKE UP! ALARM ON", (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Draw warning rectangle
                            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)
                    else:
                        self.frame_counter = max(0, self.frame_counter - 1)
                        if self.frame_counter == 0:
                            self.stop_alarm()
                        
                        cv2.putText(frame, "ALERT", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display EAR value and frame counter
                    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Frames: {self.frame_counter}/{self.CONSEC_FRAMES}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Display threshold
                    cv2.putText(frame, f"Threshold: {self.EAR_THRESHOLD}", (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow('MediaPipe Drowsiness Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.stop_alarm()
            elif key == ord('r'):
                self.frame_counter = 0
                self.stop_alarm()
                print("Counter reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

def main():
    """Main function"""
    detector = DrowsinessDetectorMediaPipe()
    
    try:
        detector.detect_drowsiness()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A working camera")
        print("2. Installed required dependencies: pip install mediapipe opencv-python pygame scipy")

if __name__ == "__main__":
    main()
# ```