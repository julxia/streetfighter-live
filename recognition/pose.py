import cv2
import mediapipe as mp
import numpy as np
import os
import time
from mediapipe.framework.formats import landmark_pb2
from threading import Thread

from recognition.model_types import ModelOutput


class GestureRecognizer:
    """Recognizes gestures based on pose landmarks"""

    def __init__(self):
        self.last_gesture = None
        self.gesture_timestamps = {}
        self.cooldown = 1.0  # Cooldown in seconds between gesture detections

    def detect_punch(self, pose_landmarks):
        """Detect a punch gesture - right arm extended forward"""
        if not pose_landmarks:
            return False, 0.0

        # Get relevant landmarks
        right_shoulder = pose_landmarks[12]  # Right shoulder
        right_elbow = pose_landmarks[14]  # Right elbow
        right_wrist = pose_landmarks[16]  # Right wrist

        # Check if arm is extended (elbow is extended)
        elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Check if arm is forward (z-coordinate is negative, meaning towards camera)
        arm_forward = right_wrist.z < right_elbow.z

        # Calculate confidence based on how extended the arm is and how forward it is
        angle_confidence = min(
            1.0, max(0.0, (elbow_angle - 130) / 50)
        )  # Normalize to 0-1 range
        z_diff = right_elbow.z - right_wrist.z
        forward_confidence = min(1.0, max(0.0, z_diff * 10))  # Scale z difference

        # Combined confidence
        confidence = angle_confidence * 0.7 + forward_confidence * 0.3

        return elbow_angle > 150 and arm_forward, confidence

    def detect_kick(self, pose_landmarks):
        """Detect a kick gesture - leg extended forward"""
        if not pose_landmarks:
            return False, 0.0

        # Get relevant landmarks
        right_hip = pose_landmarks[24]  # Right hip
        right_knee = pose_landmarks[26]  # Right knee
        right_ankle = pose_landmarks[28]  # Right ankle

        # Check if leg is extended
        knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)

        # Check if leg is raised (ankle is higher than knee)
        leg_raised = right_ankle.y < right_knee.y
        height_diff = right_knee.y - right_ankle.y

        # Calculate confidence based on how extended the leg is and how high it's raised
        angle_confidence = min(
            1.0, max(0.0, (knee_angle - 130) / 50)
        )  # Normalize to 0-1 range
        height_confidence = min(
            1.0, max(0.0, height_diff * 10)
        )  # Scale height difference

        # Combined confidence
        confidence = angle_confidence * 0.6 + height_confidence * 0.4

        return knee_angle > 150 and leg_raised, confidence

    def detect_block(self, pose_landmarks):
        """Detect a block gesture - arms in front of body forming protection"""
        if not pose_landmarks:
            return False, 0.0

        # Get relevant landmarks
        left_shoulder = pose_landmarks[11]  # Left shoulder
        left_elbow = pose_landmarks[13]  # Left elbow
        left_wrist = pose_landmarks[15]  # Left wrist
        right_shoulder = pose_landmarks[12]  # Right shoulder
        right_elbow = pose_landmarks[14]  # Right elbow
        right_wrist = pose_landmarks[16]  # Right wrist

        # Check if both arms are bent
        left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self._calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )

        # Check if arms are in front of body
        arms_forward = (left_wrist.z < left_elbow.z) and (right_wrist.z < right_elbow.z)

        # Check if wrists are at approximate shoulder height
        wrists_at_height = (
            abs(left_wrist.y - left_shoulder.y) < 0.15
            and abs(right_wrist.y - right_shoulder.y) < 0.15
        )

        # Calculate confidence based on angles and positions
        left_angle_confidence = min(1.0, max(0.0, (120 - left_elbow_angle) / 60))
        right_angle_confidence = min(1.0, max(0.0, (120 - right_elbow_angle) / 60))

        height_diff_left = abs(left_wrist.y - left_shoulder.y)
        height_diff_right = abs(right_wrist.y - right_shoulder.y)
        height_confidence = min(
            1.0, max(0.0, (0.15 - (height_diff_left + height_diff_right) / 2) / 0.15)
        )

        # Combined confidence
        confidence = (
            left_angle_confidence * 0.3
            + right_angle_confidence * 0.3
            + height_confidence * 0.4
        )

        is_block = (
            left_elbow_angle < 120
            and right_elbow_angle < 120
            and arms_forward
            and wrists_at_height
        )

        return is_block, confidence

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

        # Check for gestures with confidence
        is_punch, punch_confidence = self.detect_punch(pose_landmarks)
        is_kick, kick_confidence = self.detect_kick(pose_landmarks)
        is_block, block_confidence = self.detect_block(pose_landmarks)

        # Choose the gesture with highest confidence
        gesture = None
        confidence = 0.0

        if is_punch and punch_confidence > confidence:
            gesture = "PUNCH"
            confidence = punch_confidence

        if is_kick and kick_confidence > confidence:
            gesture = "KICK"
            confidence = kick_confidence

        if is_block and block_confidence > confidence:
            gesture = "BLOCK"
            confidence = block_confidence

        # Apply cooldown
        if gesture and (
            gesture != self.last_gesture
            or current_time - self.gesture_timestamps.get(gesture, 0) > self.cooldown
        ):
            self.last_gesture = gesture
            self.gesture_timestamps[gesture] = current_time
            return gesture, confidence

        return None, 0.0


class PoseResultCallback:
    """Callback class for MediaPipe Pose Landmarker"""

    def __init__(self):
        self.result = None
        self.gesture_recognizer = GestureRecognizer()
        self.detected_gesture = None
        self.gesture_confidence = 0.0

    def __call__(self, result, output_image, timestamp_ms):
        self.result = result

        # Process gesture detection if pose landmarks are available
        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            self.detected_gesture, self.gesture_confidence = (
                self.gesture_recognizer.detect_gesture(result.pose_landmarks[0])
            )
        else:
            self.detected_gesture = None
            self.gesture_confidence = 0.0

    def get_output(self) -> ModelOutput:
        """Return the gesture data as a JSON string"""
        if self.detected_gesture:
            data = {
                "output": self.detected_gesture.lower(),
                "confidence": round(self.gesture_confidence, 2),
            }
        else:
            data = {"output": None, "confidence": 0.0}
        return ModelOutput(output=data["output"], confidence=data["confidence"])


class PoseRecognition:
    """API for pose recognition to be used in the main game logic"""

    def __init__(
        self,
        model_path="./models/pose/pose_landmarker_full.task",
        camera_id=0,
        detection_confidence=0.5,
        tracking_confidence=0.5,
        display_camera=True,
        display_landmarks=True,
    ):
        """
        Initialize the pose recognition system

        Args:
            model_path: Path to the MediaPipe pose landmarker model
            camera_id: Camera device ID (usually 0 for built-in webcam)
            detection_confidence: Minimum confidence for pose detection
            tracking_confidence: Minimum confidence for pose tracking
            display_camera: Whether to display the camera feed
            display_landmarks: Whether to display pose landmarks
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.display_camera = display_camera
        self.display_landmarks = display_landmarks

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Initialize MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Set up MediaPipe classes
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # Initialize callback and video capture
        self.pose_callback = PoseResultCallback()
        self.cap = None
        self.landmarker = None

        # Game state
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None

        # Threading
        self.thread = None

    def initialize(self):
        """Initialize the video capture and pose landmarker"""
        # Create options for the pose landmarker
        options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            min_pose_detection_confidence=self.detection_confidence,
            min_pose_presence_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
            num_poses=1,
            result_callback=self.pose_callback,
        )

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video capture device {self.camera_id}")

        # Create pose landmarker
        self.landmarker = self.PoseLandmarker.create_from_options(options)
        self.start_time = time.time()
        self.is_running = True

        return True

    def start(self, threaded=True):
        """Start the pose recognition process"""
        if not self.landmarker:
            self.initialize()

        if threaded:
            self.thread = Thread(target=self._run_loop)
            self.thread.daemon = True
            self.thread.start()
        else:
            self._run_loop()

    def stop(self):
        """Stop the pose recognition process"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)

        if self.cap:
            self.cap.release()

        if self.landmarker:
            self.landmarker.close()

        if self.display_camera:
            cv2.destroyAllWindows()

    def _run_loop(self):
        """Main processing loop"""
        while self.is_running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            self.current_frame = image.copy()

            # Convert the image to RGB and process it
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Get timestamp for the frame
            frame_timestamp_ms = int(time.time() * 1000)

            # Process the frame asynchronously
            self.landmarker.detect_async(mp_image, frame_timestamp_ms)

            # Draw landmarks if result is available
            if (
                self.pose_callback.result
                and self.pose_callback.result.pose_landmarks
                and self.display_landmarks
            ):
                # Draw pose landmarks using the Mediapipe solution draw method
                for idx, landmarks in enumerate(
                    self.pose_callback.result.pose_landmarks
                ):
                    # Convert landmarks to proto format for drawing
                    landmark_list = landmark_pb2.NormalizedLandmarkList()
                    for landmark in landmarks:
                        landmark_proto = landmark_pb2.NormalizedLandmark()
                        landmark_proto.x = landmark.x
                        landmark_proto.y = landmark.y
                        landmark_proto.z = landmark.z
                        if hasattr(landmark, "visibility"):
                            landmark_proto.visibility = landmark.visibility
                        landmark_list.landmark.append(landmark_proto)

                    # Draw the landmarks
                    self.mp_drawing.draw_landmarks(
                        image,
                        landmark_list,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

            # Calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()

            # Display FPS and gesture info
            if self.display_camera:
                cv2.putText(
                    image,
                    f"FPS: {self.fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Display current gesture and confidence
                if self.pose_callback.detected_gesture:
                    gesture_text = f"{self.pose_callback.detected_gesture}: {self.pose_callback.gesture_confidence:.2f}"
                    cv2.putText(
                        image,
                        gesture_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                # Display instructions
                cv2.putText(
                    image,
                    "Gestures: Punch, Kick, Block",
                    (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Show the image
                cv2.imshow("Pose Recognition", image)

                # Exit on 'q' key press
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    self.is_running = False
                    break

            self.processed_frame = image.copy()

    def get_current_frame(self):
        """Get the current camera frame"""
        return self.current_frame

    def get_processed_frame(self):
        """Get the current processed frame with landmarks"""
        return self.processed_frame

    def get_detected_gesture(self):
        """Get the currently detected gesture"""
        return self.pose_callback.detected_gesture

    def get_gesture_confidence(self):
        """Get the confidence level of the currently detected gesture"""
        return self.pose_callback.gesture_confidence

    def get_output(self) -> ModelOutput:
        """Get the gesture data as a dictionary"""
        return self.pose_callback.get_output()

    def get_pose_landmarks(self):
        """Get the current pose landmarks"""
        if self.pose_callback.result and self.pose_callback.result.pose_landmarks:
            return self.pose_callback.result.pose_landmarks[0]
        return None

    def get_fps(self):
        """Get the current FPS"""
        return self.fps

    def is_initialized(self):
        """Check if the pose recognition system is initialized"""
        return self.landmarker is not None

    def is_detecting_pose(self):
        """Check if the system is currently detecting a pose"""
        return (
            self.pose_callback.result is not None
            and len(self.pose_callback.result.pose_landmarks) > 0
        )


# Example usage
def example_usage():
    """Example of how to use the PoseRecognition API"""
    # Initialize pose recognition
    pose_system = PoseRecognition(
        model_path="./models/pose/pose_landmarker_full.task", display_camera=True
    )

    try:
        print("Starting pose recognition...")
        print("Pose System initialized?", pose_system.is_initialized())

        # Start pose recognition in a separate thread
        pose_system.start(threaded=True)

        print("Pose System initialized?", pose_system.is_initialized())

        # Game loop
        score = 0
        last_gesture = None

        while True:
            # Get the current detected gesture as JSON
            gesture_data = pose_system.get_output()

            # Print the JSON data for demonstration
            if gesture_data["output"]:
                print(f"Gesture data: {gesture_data}")

            # Update game state based on gesture
            if gesture_data["output"] and gesture_data["output"] != last_gesture:
                print(
                    f"Detected: {gesture_data['output']} with confidence {gesture_data['confidence']}"
                )
                score += int(
                    gesture_data["confidence"] * 10
                )  # Score based on confidence
                last_gesture = gesture_data["output"]

            # Exit on keyboard interrupt
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        # Clean up
        pose_system.stop()
        print(f"Final score: {score}")


if __name__ == "__main__":
    try:
        example_usage()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
