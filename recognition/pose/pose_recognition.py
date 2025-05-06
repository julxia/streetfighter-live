import cv2
import mediapipe as mp
import numpy as np
import os
import time
from threading import Thread


from .pose_callback import PoseResultCallback
from ..model_types import ModelOutput


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
