# Create a complete drowsiness detection project using dlib approach
import os

# Create the main drowsiness detection script using dlib
dlib_script = '''import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
import time

# Initialize pygame mixer for alarm
pygame.mixer.init()

class DrowsinessDetectorDlib:
    def __init__(self):
        # Eye Aspect Ratio (EAR) threshold and consecutive frame count
        self.EAR_THRESHOLD = 0.25
        self.CONSEC_FRAMES = 20
        
        # Initialize counters
        self.frame_counter = 0
        self.alarm_on = False
        
        # Initialize dlib face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Define facial landmark indices for left and right eye
        self.LEFT_EYE_START = 42
        self.LEFT_EYE_END = 47
        self.RIGHT_EYE_START = 36
        self.RIGHT_EYE_END = 41
        
        # Load alarm sound (create a simple beep if file doesn't exist)
        self.load_alarm_sound()
        
    def load_alarm_sound(self):
        """Load or create alarm sound"""
        try:
            # Try to load existing alarm sound
            pygame.mixer.music.load("alarm.wav")
        except:
            # Create a simple beep sound if alarm.wav doesn't exist
            print("Alarm sound file not found. Using system beep.")
            self.use_system_beep = True
            
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Compute euclidean distances between vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute euclidean distance between horizontal eye landmarks
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_landmarks(self, landmarks, start, end):
        """Extract eye landmarks from facial landmarks"""
        eye_points = []
        for i in range(start, end + 1):
            eye_points.append((landmarks.part(i).x, landmarks.part(i).y))
        return np.array(eye_points)
    
    def draw_eye_landmarks(self, frame, eye_landmarks):
        """Draw eye landmarks on frame"""
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
                # Fallback to system beep
                print("\\a" * 5)  # System beep
                
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
            
        print("Drowsiness Detection Started...")
        print("Press 'q' to quit")
        print("Press 's' to stop alarm")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            for face in faces:
                # Get facial landmarks
                landmarks = self.predictor(gray, face)
                
                # Extract left and right eye landmarks
                left_eye = self.extract_eye_landmarks(landmarks, self.LEFT_EYE_START, self.LEFT_EYE_END)
                right_eye = self.extract_eye_landmarks(landmarks, self.RIGHT_EYE_START, self.RIGHT_EYE_END)
                
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
                else:
                    self.frame_counter = 0
                    self.stop_alarm()
                    cv2.putText(frame, "ALERT", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display EAR value
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Display frame counter
                cv2.putText(frame, f"Frames: {self.frame_counter}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Drowsiness Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.stop_alarm()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

def main():
    """Main function"""
    detector = DrowsinessDetectorDlib()
    
    try:
        detector.detect_drowsiness()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A working camera")
        print("2. Downloaded shape_predictor_68_face_landmarks.dat file")
        print("3. Installed required dependencies")

if __name__ == "__main__":
    main()
'''

# Save the dlib-based script
with open("drowsiness_detector_dlib.py", "w") as f:
    f.write(dlib_script)

print("Created drowsiness_detector_dlib.py")