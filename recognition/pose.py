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
        """Detect a punch gesture - either jab (left arm extended) or cross (right arm extended)"""
        if not pose_landmarks:
            return False, 0.0, None

        PUNCH_ARM_ANGLE = 130

        # Get relevant landmarks for both arms
        right_shoulder = pose_landmarks[11]  # Left shoulder
        right_elbow = pose_landmarks[13]  # Left elbow
        right_wrist = pose_landmarks[15]  # Left wrist

        left_shoulder = pose_landmarks[12]  # Right shoulder
        left_elbow = pose_landmarks[14]  # Right elbow
        left_wrist = pose_landmarks[16]  # Right wrist

        # Check if either arm is extended
        left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self._calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )

        # Check if arms are forward
        left_arm_forward = left_wrist.z < left_elbow.z
        right_arm_forward = right_wrist.z < right_elbow.z

        # Check if other arm is in blocking position (bent elbow, ~90 degrees)
        left_blocking = 70 < left_elbow_angle < 120
        right_blocking = 70 < right_elbow_angle < 120

        # Calculate confidence for each punch type
        jab_confidence = 0.0
        cross_confidence = 0.0

        # JAB - left arm extended, right arm blocking
        if left_elbow_angle > PUNCH_ARM_ANGLE and left_arm_forward and right_blocking:
            angle_confidence = min(1.0, max(0.0, (left_elbow_angle - 130) / 50))
            z_diff = left_elbow.z - left_wrist.z
            forward_confidence = min(1.0, max(0.0, z_diff * 10))
            block_confidence = min(
                1.0, max(0.0, (1.0 - abs(right_elbow_angle - 90) / 30))
            )
            jab_confidence = (
                angle_confidence * 0.5
                + forward_confidence * 0.3
                + block_confidence * 0.2
            )

        # CROSS - right arm extended, left arm blocking
        if right_elbow_angle > PUNCH_ARM_ANGLE and right_arm_forward and left_blocking:
            angle_confidence = min(1.0, max(0.0, (right_elbow_angle - 130) / 50))
            z_diff = right_elbow.z - right_wrist.z
            forward_confidence = min(1.0, max(0.0, z_diff * 10))
            block_confidence = min(
                1.0, max(0.0, (1.0 - abs(left_elbow_angle - 90) / 30))
            )
            cross_confidence = (
                angle_confidence * 0.5
                + forward_confidence * 0.3
                + block_confidence * 0.2
            )

        # Determine which type of punch and confidence
        if jab_confidence > cross_confidence:
            if jab_confidence > 0.5:  # Threshold
                return True, jab_confidence, "punch"
        else:
            if cross_confidence > 0.5:  # Threshold
                return True, cross_confidence, "punch"

        return False, 0.0, None

    def detect_kick(self, pose_landmarks):
        """Detect a kick gesture - either leg extended forward"""
        if not pose_landmarks:
            return False, 0.0, None

        # Get relevant landmarks for right leg
        left_hip = pose_landmarks[24]  # Right hip
        left_knee = pose_landmarks[26]  # Right knee
        left_ankle = pose_landmarks[28]  # Right ankle

        # Get relevant landmarks for left leg
        right_hip = pose_landmarks[23]  # Left hip
        right_knee = pose_landmarks[25]  # Left knee
        right_ankle = pose_landmarks[27]  # Left ankle

        # Calculate angles for both legs
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)

        # Check if either leg is raised (ankle is higher than knee)
        right_leg_raised = right_ankle.y < right_knee.y
        left_leg_raised = left_ankle.y < left_knee.y

        # Calculate height difference for confidence calculation
        right_height_diff = right_knee.y - right_ankle.y if right_leg_raised else 0
        left_height_diff = left_knee.y - left_ankle.y if left_leg_raised else 0

        # Calculate confidence for right kick
        right_angle_confidence = min(1.0, max(0.0, (right_knee_angle - 130) / 50))
        right_height_confidence = min(1.0, max(0.0, right_height_diff * 10))
        right_confidence = right_angle_confidence * 0.6 + right_height_confidence * 0.4

        # Calculate confidence for left kick
        left_angle_confidence = min(1.0, max(0.0, (left_knee_angle - 130) / 50))
        left_height_confidence = min(1.0, max(0.0, left_height_diff * 10))
        left_confidence = left_angle_confidence * 0.6 + left_height_confidence * 0.4

        # Determine which leg is kicking (if any) based on confidence
        is_right_kick = right_knee_angle > 150 and right_leg_raised
        is_left_kick = left_knee_angle > 150 and left_leg_raised

        # Return the kick with higher confidence
        if is_right_kick and is_left_kick:
            if right_confidence > left_confidence:
                return True, right_confidence, "RIGHT_KICK"
            else:
                return True, left_confidence, "LEFT_KICK"
        elif is_right_kick:
            return True, right_confidence, "RIGHT_KICK"
        elif is_left_kick:
            return True, left_confidence, "LEFT_KICK"

        return False, 0.0, None

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

        # Clean up old timestamps (older than 10 seconds)
        expired_gestures = []
        for g, ts in self.gesture_timestamps.items():
            if current_time - ts > 10.0:  # 10 seconds cleanup
                expired_gestures.append(g)

        # Remove expired gestures
        for g in expired_gestures:
            del self.gesture_timestamps[g]

        # Check for gestures with confidence
        is_punch, punch_confidence, _ = self.detect_punch(pose_landmarks)
        is_kick, kick_confidence, _ = self.detect_kick(pose_landmarks)
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
        display_camera=False,
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
        """Main processing loop with optimized landmark processing"""
        last_processed_time = 0
        processing_interval = 1 / 30  # Process at most 30 frames per second
        skip_frames_count = 0

        # Track system load
        system_load = 0

        # Create a simplified connection set for faster drawing
        # Only connections relevant for gesture detection
        simplified_connections = [
            (11, 13),
            (13, 15),  # Left arm
            (12, 14),
            (14, 16),  # Right arm
            (23, 25),
            (25, 27),  # Left leg
            (24, 26),
            (26, 28),  # Right leg
        ]

        while self.is_running and self.cap.isOpened():
            current_time = time.time()

            # Adaptive frame skipping based on system load
            should_process = current_time - last_processed_time >= processing_interval

            success, image = self.cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            # Store original image flipped horizontally
            image = cv2.flip(image, 1)
            self.current_frame = image.copy()

            # Process frame only when needed
            if should_process:
                skip_frames_count = 0
                last_processed_time = current_time

                process_start_time = time.time()

                # Convert the image to RGB and process it
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                # Process the frame asynchronously
                frame_timestamp_ms = int(current_time * 1000)
                self.landmarker.detect_async(mp_image, frame_timestamp_ms)

                # Create a copy of the image for drawing only if needed
                if self.display_camera:
                    display_img = image.copy()

                    # Draw landmarks if we have a result and display is enabled
                    if (
                        self.pose_callback.result
                        and self.pose_callback.result.pose_landmarks
                        and self.display_landmarks
                    ):

                        for landmarks in self.pose_callback.result.pose_landmarks:
                            # Draw only important landmarks manually for better performance
                            # Key points for gesture recognition
                            key_points = [
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                23,
                                24,
                                25,
                                26,
                                27,
                                28,
                            ]

                            # Draw key landmarks (larger and more visible)
                            for idx in key_points:
                                if idx < len(landmarks):  # Check if the index is valid
                                    lm = landmarks[idx]
                                    h, w, c = display_img.shape
                                    cx, cy = int(lm.x * w), int(lm.y * h)
                                    # Draw important landmarks bigger
                                    cv2.circle(
                                        display_img, (cx, cy), 5, (0, 255, 0), -1
                                    )

                            # Draw simplified connections manually
                            for connection in simplified_connections:
                                start_idx, end_idx = connection
                                if start_idx < len(landmarks) and end_idx < len(
                                    landmarks
                                ):
                                    start_point = landmarks[start_idx]
                                    end_point = landmarks[end_idx]

                                    h, w, c = display_img.shape
                                    start_x, start_y = int(start_point.x * w), int(
                                        start_point.y * h
                                    )
                                    end_x, end_y = int(end_point.x * w), int(
                                        end_point.y * h
                                    )

                                    cv2.line(
                                        display_img,
                                        (start_x, start_y),
                                        (end_x, end_y),
                                        (0, 255, 0),
                                        2,
                                    )

                    # Calculate and display FPS
                    self.frame_count += 1
                    elapsed_time = current_time - self.start_time
                    if elapsed_time >= 1.0:
                        self.fps = self.frame_count / elapsed_time
                        self.frame_count = 0
                        self.start_time = current_time

                    # Display FPS and gesture info
                    cv2.putText(
                        display_img,
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
                            display_img,
                            gesture_text,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )

                    # Display skipped frames
                    if skip_frames_count > 0:
                        cv2.putText(
                            display_img,
                            f"Skipped: {skip_frames_count}",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 165, 0),
                            2,
                        )

                    # Display instructions
                    cv2.putText(
                        display_img,
                        "Gestures: Punch, Kick, Block",
                        (10, display_img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    # Show the image
                    cv2.imshow("Pose Recognition", display_img)

                    # Store the processed frame
                    self.processed_frame = display_img

                # Calculate processing time for this frame
                process_end_time = time.time()
                frame_process_time = process_end_time - process_start_time

                # Update system load average (EMA - exponential moving average)
                system_load = 0.7 * system_load + 0.3 * frame_process_time

                # Adjust the processing interval based on system load
                if system_load > 0.03:  # If processing takes > 30ms
                    processing_interval = min(
                        0.1, system_load * 1.5
                    )  # Cap at 10 FPS min
                else:
                    processing_interval = 1 / 30  # Try to maintain 30 FPS
            else:
                # Count skipped frames for monitoring
                skip_frames_count += 1

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.is_running = False
                break

            # Small sleep to prevent CPU overuse when not processing
            if not should_process:
                time.sleep(0.001)

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
        model_path="./models/pose/pose_landmarker_full.task", display_camera=False
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
            if gesture_data.output:
                print(f"Gesture data: {gesture_data}")

            # Update game state based on gesture
            if gesture_data.output and gesture_data.output != last_gesture:
                print(
                    f"Detected: {gesture_data.output} with confidence {gesture_data.confidence}"
                )
                score += int(gesture_data.confidence * 10)  # Score based on confidence
                last_gesture = gesture_data.output

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
