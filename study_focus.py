import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import csv
from datetime import datetime
from scipy.spatial import distance as dist
import os

class StudyFocusAssistant:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.display.init()

        # Detection thresholds (adjust these based on testing)
        self.EAR_THRESH = 0.25  # Eye Aspect Ratio
        self.MAR_THRESH = 0.60  # Mouth Aspect Ratio
        self.HEAD_TILT_THRESH = 25  # Degrees
        
        # State tracking
        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.head_tilt_counter = 0
        self.CONSEC_FRAMES = 45  # 1.5 seconds at 30 FPS (for 3 seconds use 90)
        self.YAWN_FRAMES = 15   # Frames for yawn detection
        self.HEAD_FRAMES = 30   # Frames for head tilt detection
        self.alarm_active = False
        self.last_alarm_time = 0
        self.ALARM_COOLDOWN = 3  # 3 seconds between alarms
        
        # MediaPipe setup with correct eye indices
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Correct MediaPipe 468-point indices
        self.L_EYE = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
        self.R_EYE = [362, 385, 386, 263, 373, 374] # Right eye landmarks
        self.MOUTH = [61, 291, 39, 181]             # Mouth landmarks
        
        # Color scheme
        self.COLORS = {
            'alert': (50, 205, 50),     # Green
            'warning': (255, 165, 0),   # Orange
            'drowsy': (255, 0, 0),      # Red
            'text': (245, 245, 245),    # White
            'eye_tracking': (0, 255, 0) # Green for eye points
        }

        # Initialize metrics with proper tracking
        self.metrics = {
            'eye_closures': [],
            'yawns': [],
            'head_movements': [],
            'total_frames': 0,
            'drowsy_episodes': 0,
            'session_start': None
        }

        # Session configuration
        self.session_config = {
            'duration': 0,
            'start_time': None,
            'paused': False,
            'active': False
        }

        # Create a simple beep sound programmatically
        self.create_beep_sound()

    def create_beep_sound(self):
        """Create a simple beep sound using pygame"""
        try:
            # Create a simple sine wave beep
            sample_rate = 22050
            duration = 0.5  # seconds
            frequency = 800  # Hz
            
            frames = int(duration * sample_rate)
            arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
            arr = (arr * 32767).astype(np.int16)
            arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
            
            self.beep_sound = pygame.sndarray.make_sound(arr)
        except:
            self.beep_sound = None
            print("Warning: Could not create beep sound")

    def calculate_ear(self, eye_points):
        """Improved EAR calculation with 6-point detection"""
        try:
            A = dist.euclidean(eye_points[1], eye_points[5])
            B = dist.euclidean(eye_points[2], eye_points[4])
            C = dist.euclidean(eye_points[0], eye_points[3])
            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0.3  # Default value if calculation fails

    def calculate_mar(self, mouth_points):
        """Improved MAR calculation"""
        try:
            vertical = dist.euclidean(mouth_points[2], mouth_points[3])
            horizontal = dist.euclidean(mouth_points[0], mouth_points[1])
            return vertical / horizontal
        except:
            return 0.5  # Default value if calculation fails

    def get_head_pose(self, face_landmarks, frame_shape):
        """Simplified head pose using facial geometry"""
        try:
            # Use simple landmark relationships for tilt detection
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            
            # Convert to pixel coordinates
            left_eye_pos = (left_eye.x * frame_shape[1], left_eye.y * frame_shape[0])
            right_eye_pos = (right_eye.x * frame_shape[1], right_eye.y * frame_shape[0])
            
            # Calculate tilt angle
            dx = right_eye_pos[0] - left_eye_pos[0]
            dy = right_eye_pos[1] - left_eye_pos[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Normalize angle to -180 to 180 range
            if angle > 180:
                angle -= 360
            elif angle < -180:
                angle += 360
                
            return (0, 0, angle)
            
        except:
            return (0, 0, 0)

    def process_frame(self, frame):
        # Resize frame for larger display
        frame = cv2.resize(frame, (1024, 768))  # Increased from default camera resolution
        
        frame.flags.writeable = False
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True
        
        if results.multi_face_landmarks:
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) 
                        for lm in results.multi_face_landmarks[0].landmark]
            
            # Draw eye tracking points
            for idx in self.L_EYE + self.R_EYE:
                if idx < len(landmarks):
                    x, y = landmarks[idx]
                    cv2.circle(frame, (x, y), 2, self.COLORS['eye_tracking'], -1)
            
            # Calculate metrics
            left_ear = self.calculate_ear([landmarks[i] for i in self.L_EYE if i < len(landmarks)])
            right_ear = self.calculate_ear([landmarks[i] for i in self.R_EYE if i < len(landmarks)])
            avg_ear = (left_ear + right_ear) / 2.0
            
            mar = self.calculate_mar([landmarks[i] for i in self.MOUTH if i < len(landmarks)])
            head_angles = self.get_head_pose(results.multi_face_landmarks[0].landmark, frame.shape)
            
            # Update detection state
            self.update_metrics(avg_ear, mar, head_angles)
            
            # Draw real-time feedback
            frame = self.draw_hud(frame, avg_ear, mar, head_angles)
        else:
            # No face detected - reset counters
            self.drowsy_counter = max(0, self.drowsy_counter - 3)
            self.yawn_counter = max(0, self.yawn_counter - 2)
            self.head_tilt_counter = max(0, self.head_tilt_counter - 2)
            
        return frame

    def update_metrics(self, ear, mar, head_angles):
        self.metrics['total_frames'] += 1
        current_time = time.time()
        
        # Individual indicators
        eye_drowsy = ear < self.EAR_THRESH
        mouth_drowsy = mar > self.MAR_THRESH
        head_drowsy = abs(head_angles[2]) > self.HEAD_TILT_THRESH
        
        # Eye closure detection
        if eye_drowsy:
            self.drowsy_counter += 1
            if self.drowsy_counter >= self.CONSEC_FRAMES:
                self.record_event('eye_closure', current_time)
                self.trigger_alarm('drowsiness')
                self.drowsy_counter = 0  # Reset after detection
        else:
            self.drowsy_counter = max(0, self.drowsy_counter - 2)
        
        # Yawn detection
        if mouth_drowsy:
            self.yawn_counter += 1
            if self.yawn_counter >= self.YAWN_FRAMES:
                self.record_event('yawn', current_time)
                self.yawn_counter = 0  # Reset after detection
        else:
            self.yawn_counter = max(0, self.yawn_counter - 1)
        
        # Head tilt detection
        if head_drowsy:
            self.head_tilt_counter += 1
            if self.head_tilt_counter >= self.HEAD_FRAMES:
                self.record_event('head_movement', current_time)
                self.head_tilt_counter = 0  # Reset after detection
        else:
            self.head_tilt_counter = max(0, self.head_tilt_counter - 1)

    def record_event(self, event_type, timestamp):
        """Record detected events for reporting"""
        if event_type == 'eye_closure':
            self.metrics['eye_closures'].append(timestamp)
            self.metrics['drowsy_episodes'] += 1
        elif event_type == 'yawn':
            self.metrics['yawns'].append(timestamp)
        elif event_type == 'head_movement':
            self.metrics['head_movements'].append(timestamp)

    def trigger_alarm(self, alarm_type):
        """Trigger alarm with cooldown to prevent spam"""
        current_time = time.time()
        if current_time - self.last_alarm_time > self.ALARM_COOLDOWN:
            self.alarm_active = True
            self.last_alarm_time = current_time
            
            # Visual alert
            print(f"\n🚨 ALERT: {alarm_type.upper()} DETECTED! 🚨")
            
            # Sound alert
            try:
                if self.beep_sound:
                    self.beep_sound.play()
                else:
                    # Fallback to system beep
                    print("\a" * 3)
            except:
                print("\a" * 3)  # System beep fallback

    def draw_hud(self, frame, ear, mar, head_angles):
        h, w = frame.shape[:2]
        
        pitch, yaw, roll = head_angles
        
        # Status calculations
        eye_status = "CLOSED" if ear < self.EAR_THRESH else "OPEN"
        yawn_status = "YAWNING" if mar > self.MAR_THRESH else "NORMAL"
        head_status = "TILTED" if abs(roll) > self.HEAD_TILT_THRESH else "STABLE"
        
        # Multi-factor assessment
        drowsy_count = sum([
            ear < self.EAR_THRESH,
            mar > self.MAR_THRESH,
            abs(roll) > self.HEAD_TILT_THRESH
        ])
        
        overall_status = "DROWSY" if drowsy_count >= 1 else "ALERT"
        status_color = self.COLORS['drowsy'] if overall_status == "DROWSY" else self.COLORS['alert']
        
        # Draw metrics panel with better visibility
        cv2.rectangle(frame, (10, 10), (600, 240), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (600, 240), (255, 255, 255), 2)
        
        metrics = [
            ("Eye Status:", f"{eye_status} (EAR: {ear:.3f})"),
            ("Drowsy Counter:", f"{self.drowsy_counter}/{self.CONSEC_FRAMES}"),
            ("Yawn Status:", f"{yawn_status} (MAR: {mar:.3f})"),
            ("Yawn Counter:", f"{self.yawn_counter}/{self.YAWN_FRAMES}"),
            ("Head Tilt:", f"{roll:.1f}° ({head_status})"),
            ("Head Counter:", f"{self.head_tilt_counter}/{self.HEAD_FRAMES}"),
            ("Overall Status:", f"{overall_status}"),
            ("Events:", f"Eyes:{len(self.metrics['eye_closures'])} Yawns:{len(self.metrics['yawns'])} Head:{len(self.metrics['head_movements'])}")
        ]
        
        for i, (label, value) in enumerate(metrics):
            color = status_color if "Overall Status:" in label else self.COLORS['text']
            cv2.putText(frame, f"{label} {value}", (20, 40 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Session timer
        if self.session_config['active']:
            elapsed = time.time() - self.session_config['start_time']
            remaining = max(0, self.session_config['duration'] - elapsed)
            timer_text = f"Time: {int(remaining//60)}:{int(remaining%60):02d}"
            cv2.putText(frame, timer_text, (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        
        # Alarm indicator
        if self.alarm_active:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
            cv2.putText(frame, "DROWSINESS ALERT!", (w//2-200, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            # Reset alarm after a short time
            if time.time() - self.last_alarm_time > 2:
                self.alarm_active = False
        
        return frame

    def session_setup_screen(self):
        """Interactive session setup using pygame events"""
        pygame.display.set_caption("Study Focus Setup")
        duration_options = [1, 15, 25, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]
        selected_index = 0  # Start with 1 minute for testing
        
        cap = cv2.VideoCapture(0)
        # Set camera resolution for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        
        print("=== Study Focus Assistant Setup ===")
        print("Use UP/DOWN arrows to select duration")
        print("Press ENTER to start session")
        print("Press ESC to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize frame for larger display
            frame = cv2.resize(frame, (1024, 768))
                
            # Draw setup interface
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (50, 50), (w-50, h-50), (0, 0, 0), -1)
            cv2.rectangle(frame, (50, 50), (w-50, h-50), (255, 255, 255), 2)
            
            # Title
            cv2.putText(frame, "Study Focus Assistant", (60, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLORS['text'], 2)
            
            cv2.putText(frame, "Select Session Duration:", (60, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['text'], 2)
            
            # Duration options
            for i, duration in enumerate(duration_options):
                y_pos = 160 + i * 35
                color = self.COLORS['alert'] if i == selected_index else self.COLORS['text']
                prefix = "→ " if i == selected_index else "  "
                cv2.putText(frame, f"{prefix}{duration} minutes", (80, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Instructions
            cv2.putText(frame, "UP/DOWN: Navigate | ENTER: Start | ESC: Exit", 
                       (60, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Study Focus Assistant - Setup', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == 13:  # ENTER
                cap.release()
                cv2.destroyAllWindows()
                return duration_options[selected_index]
            elif key == 82 or key == ord('w'):  # UP arrow or W
                selected_index = max(0, selected_index - 1)
            elif key == 84 or key == ord('s'):  # DOWN arrow or S
                selected_index = min(len(duration_options)-1, selected_index + 1)

    def start_session(self, duration):
        self.session_config = {
            'duration': duration * 60,
            'start_time': time.time(),
            'paused': False,
            'active': True
        }
        
        # Initialize session metrics
        self.metrics['session_start'] = time.time()
        self.metrics['eye_closures'] = []
        self.metrics['yawns'] = []
        self.metrics['head_movements'] = []
        self.metrics['drowsy_episodes'] = 0
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera resolution for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
            
        clock = pygame.time.Clock()
        
        print(f"\n🎯 Starting {duration}-minute focus session...")
        print("Press 'q' to quit, 'p' to pause/unpause")
        print("Close your eyes for 3+ seconds to test drowsiness detection")
        
        frame_count = 0
        while self.session_active():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize frame for larger display
            if not self.session_config['paused']:
                frame = self.process_frame(frame)
            else:
                # Resize frame even when paused
                frame = cv2.resize(frame, (1024, 768))
                # Show paused message
                h, w = frame.shape[:2]
                cv2.putText(frame, "PAUSED - Press 'p' to resume", 
                           (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            cv2.imshow('Study Focus Assistant', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nSession stopped by user")
                break
            elif key == ord('p'):
                self.session_config['paused'] = not self.session_config['paused']
                status = "PAUSED" if self.session_config['paused'] else "RESUMED"
                print(f"Session {status}")
                
            clock.tick(30)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.session_active():
            print("\n✅ Session completed successfully!")
        
        self.generate_report()

    def session_active(self):
        if not self.session_config['active']:
            return False
        elapsed = time.time() - self.session_config['start_time']
        return elapsed < self.session_config['duration']

    def generate_report(self):
        """Generate detailed session report"""
        session_duration = self.session_config['duration'] / 60  # Convert to minutes
        
        print("\n" + "="*60)
        print("📊 STUDY FOCUS SESSION REPORT")
        print("="*60)
        print(f"📅 Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Planned Duration: {session_duration:.0f} minutes")
        
        if self.metrics['session_start']:
            actual_duration = (time.time() - self.metrics['session_start']) / 60
            print(f"⏱️  Actual Duration: {actual_duration:.1f} minutes")
        
        print(f"🎯 Total Frames Processed: {self.metrics['total_frames']}")
        print("\n📈 DETECTION SUMMARY:")
        print(f"👁️  Eye Closure Episodes: {len(self.metrics['eye_closures'])}")
        print(f"🥱 Yawning Episodes: {len(self.metrics['yawns'])}")
        print(f"📐 Head Movement Episodes: {len(self.metrics['head_movements'])}")
        print(f"😴 Total Drowsy Episodes: {self.metrics['drowsy_episodes']}")
        
        # Calculate focus score
        total_events = len(self.metrics['eye_closures']) + len(self.metrics['yawns']) + len(self.metrics['head_movements'])
        if session_duration > 0:
            events_per_minute = total_events / session_duration
            focus_score = max(0, 100 - (events_per_minute * 10))
        else:
            focus_score = 100
            
        print(f"\n🏆 FOCUS SCORE: {focus_score:.1f}/100")
        
        if focus_score >= 90:
            print("🌟 Excellent focus! Keep up the great work!")
        elif focus_score >= 70:
            print("👍 Good focus session with room for improvement")
        elif focus_score >= 50:
            print("⚠️  Moderate focus - consider taking breaks")
        else:
            print("😴 Low focus detected - ensure you're well-rested")
        
        print("="*60)
        
        # Save report to CSV
        self.save_report_csv()

    def save_report_csv(self):
        """Save session report to CSV file"""
        os.makedirs('Reports', exist_ok=True)
        try:
            filename = f"Reports/focus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Session Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['Duration (minutes)', self.session_config['duration'] / 60])
                writer.writerow(['Eye Closures', len(self.metrics['eye_closures'])])
                writer.writerow(['Yawns', len(self.metrics['yawns'])])
                writer.writerow(['Head Movements', len(self.metrics['head_movements'])])
                writer.writerow(['Total Drowsy Episodes', self.metrics['drowsy_episodes']])
                writer.writerow(['Total Frames', self.metrics['total_frames']])
            print(f"📝 Report saved to: {filename}")
        except Exception as e:
            print(f"❌ Could not save report: {e}")

if __name__ == "__main__":
    assistant = StudyFocusAssistant()
    duration = assistant.session_setup_screen()
    if duration:
        assistant.start_session(duration)
    pygame.quit()