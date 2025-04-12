import cv2
import mediapipe as mp
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Configuration
MODEL_PATH = './models/pose/pose_landmarker_full.task'  # Update with your model path
CAMERA_ID = 0  # Usually 0 for built-in webcam
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Define gesture recognition class
class GestureRecognizer:
    def __init__(self):
        self.last_gesture = None
        self.gesture_timestamps = {}
        self.cooldown = 1.0  # Cooldown in seconds between gesture detections
        
    def detect_punch(self, pose_landmarks):
        """Detect a punch gesture - right arm extended forward"""
        if not pose_landmarks:
            return False
            
        # Get relevant landmarks
        right_shoulder = pose_landmarks[12]  # Right shoulder
        right_elbow = pose_landmarks[14]     # Right elbow
        right_wrist = pose_landmarks[16]     # Right wrist
        
        # Check if arm is extended (elbow is extended)
        elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Check if arm is forward (z-coordinate is negative, meaning towards camera)
        arm_forward = right_wrist.z < right_elbow.z
        
        return elbow_angle > 150 and arm_forward
    
    def detect_kick(self, pose_landmarks):
        """Detect a kick gesture - leg extended forward"""
        if not pose_landmarks:
            return False
            
        # Get relevant landmarks
        right_hip = pose_landmarks[24]      # Right hip
        right_knee = pose_landmarks[26]     # Right knee
        right_ankle = pose_landmarks[28]    # Right ankle
        
        # Check if leg is extended
        knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        # Check if leg is raised (ankle is higher than knee)
        leg_raised = right_ankle.y < right_knee.y
        
        return knee_angle > 150 and leg_raised
    
    def detect_block(self, pose_landmarks):
        """Detect a block gesture - arms in front of body forming protection"""
        if not pose_landmarks:
            return False
            
        # Get relevant landmarks
        left_shoulder = pose_landmarks[11]   # Left shoulder
        left_elbow = pose_landmarks[13]      # Left elbow
        left_wrist = pose_landmarks[15]      # Left wrist
        right_shoulder = pose_landmarks[12]  # Right shoulder
        right_elbow = pose_landmarks[14]     # Right elbow
        right_wrist = pose_landmarks[16]     # Right wrist
        
        # Check if both arms are bent
        left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Check if arms are in front of body
        arms_forward = (left_wrist.z < left_elbow.z) and (right_wrist.z < right_elbow.z)
        
        # Check if wrists are at approximate shoulder height
        wrists_at_height = (abs(left_wrist.y - left_shoulder.y) < 0.15 and 
                            abs(right_wrist.y - right_shoulder.y) < 0.15)
        
        return (left_elbow_angle < 120 and right_elbow_angle < 120 and 
                arms_forward and wrists_at_height)
    
    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points (in degrees)"""
        # a: first point, b: middle point (vertex), c: end point
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        
        # Normalize vectors
        ba_norm = ba / np.linalg.norm(ba)
        bc_norm = bc / np.linalg.norm(bc)
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba_norm, bc_norm)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def detect_gesture(self, pose_landmarks):
        """Detect a gesture from the given pose landmarks"""
        current_time = time.time()
        
        # Check for gestures
        if self.detect_punch(pose_landmarks):
            gesture = "PUNCH"
        elif self.detect_kick(pose_landmarks):
            gesture = "KICK"
        elif self.detect_block(pose_landmarks):
            gesture = "BLOCK"
        else:
            gesture = None
            
        # Apply cooldown
        if gesture and (gesture != self.last_gesture or 
                        current_time - self.gesture_timestamps.get(gesture, 0) > self.cooldown):
            self.last_gesture = gesture
            self.gesture_timestamps[gesture] = current_time
            return gesture
            
        return None


# Callback function for the MediaPipe Pose Landmarker
class PoseResultCallback:
    def __init__(self):
        self.result = None
        self.gesture_recognizer = GestureRecognizer()
        self.detected_gesture = None
        
    def __call__(self, result, output_image, timestamp_ms):
        self.result = result
        
        # Process gesture detection if pose landmarks are available
        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            self.detected_gesture = self.gesture_recognizer.detect_gesture(result.pose_landmarks[0])
        else:
            self.detected_gesture = None


# Main function
def main():
    print("Starting Pose Recognition Game...")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"MediaPipe version: {mp.__version__}")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please download the model file and update the MODEL_PATH variable.")
        print("You can download the pose landmarker model from:")
        print("https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
        return
    else:
        print(f"Found model at: {MODEL_PATH}")
    
    # Set up MediaPipe Pose Landmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Create callback
    pose_callback = PoseResultCallback()
    
    # Create options for the pose landmarker
    print("Creating pose landmarker options...")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        num_poses=1,
        result_callback=pose_callback
    )
    
    # Initialize video capture
    print("Initializing video capture...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open video capture device {CAMERA_ID}")
        return
    
    # Create pose landmarker
    print("Creating pose landmarker...")
    with PoseLandmarker.create_from_options(options) as landmarker:
        print("Pose landmarker created successfully.")
        
        # Variables for FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Game state
        score = 0
        game_message = ""
        message_time = 0
        message_duration = 2.0  # Message display duration in seconds
        
        print("Starting main loop. Press 'q' to quit.")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break
                
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
                
            # Convert the image to RGB and process it
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Get timestamp for the frame
            frame_timestamp_ms = int(time.time() * 1000)
            
            # Process the frame asynchronously
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            # Draw landmarks if result is available
            if pose_callback.result and pose_callback.result.pose_landmarks:
                # Draw pose landmarks using the Mediapipe solution draw method
                for idx, landmarks in enumerate(pose_callback.result.pose_landmarks):
                    # Convert landmarks to proto format for drawing
                    landmark_list = landmark_pb2.NormalizedLandmarkList()
                    for landmark in landmarks:
                        landmark_proto = landmark_pb2.NormalizedLandmark()
                        landmark_proto.x = landmark.x
                        landmark_proto.y = landmark.y
                        landmark_proto.z = landmark.z
                        if hasattr(landmark, 'visibility'):
                            landmark_proto.visibility = landmark.visibility
                        landmark_list.landmark.append(landmark_proto)
                    
                    # Draw the landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        landmark_list,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # Check for detected gestures
                if pose_callback.detected_gesture:
                    score += 10
                    game_message = f"{pose_callback.detected_gesture} detected! +10 points"
                    message_time = time.time()
                    print(f"Detected: {pose_callback.detected_gesture}")
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Display FPS and score
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Score: {score}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display game message
            if time.time() - message_time < message_duration:
                cv2.putText(image, game_message, (image.shape[1]//2 - 200, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display instructions
            cv2.putText(image, "Gestures: Punch, Kick, Block", (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the image
            cv2.imshow('Pose Recognition Game', image)
            
            # Exit on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("Application ended.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()