# Create the dlib-based drowsiness detection script
dlib_script = """import cv2
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
        
    def calculate_ear(self, eye_landmarks):
        # Calculate Eye Aspect Ratio (EAR)
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_landmarks(self, landmarks, start, end):
        eye_points = []
        for i in range(start, end + 1):
            eye_points.append((landmarks.part(i).x, landmarks.part(i).y))
        return np.array(eye_points)
    
    def detect_drowsiness(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        print("Drowsiness Detection Started...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            for face in faces:
                landmarks = self.predictor(gray, face)
                
                left_eye = self.extract_eye_landmarks(landmarks, self.LEFT_EYE_START, self.LEFT_EYE_END)
                right_eye = self.extract_eye_landmarks(landmarks, self.RIGHT_EYE_START, self.RIGHT_EYE_END)
                
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear < self.EAR_THRESHOLD:
                    self.frame_counter += 1
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if self.frame_counter >= self.CONSEC_FRAMES:
                        cv2.putText(frame, "WAKE UP! ALARM ON", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print("DROWSINESS DETECTED! WAKE UP!")
                else:
                    self.frame_counter = 0
                    cv2.putText(frame, "ALERT", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow('Drowsiness Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = DrowsinessDetectorDlib()
    detector.detect_drowsiness()

if __name__ == "__main__":
    main()
"""

# Save to file
with open("drowsiness_detector_dlib.py", "w") as f:
    f.write(dlib_script)

print("✓ Created drowsiness_detector_dlib.py")
print("File size:", len(dlib_script), "characters")