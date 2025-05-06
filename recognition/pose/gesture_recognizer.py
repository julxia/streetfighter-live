import numpy as np
import time


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

        # Key tuning parameters - adjust these to control sensitivity
        PUNCH_ARM_ANGLE = 120  # Min angle for extended arm (lower = more sensitive)
        MIN_VISIBILITY = (
            0.65  # Landmark visibility requirement (lower = more sensitive)
        )
        HIP_VISIBILITY = 0.3  # Hip visibility requirement (lower = more sensitive)
        Z_THRESHOLD = 0.01  # How far forward wrist must be vs elbow (higher = stricter)
        BLOCKING_MIN = 45  # Min angle for blocking arm (lower = more sensitive)
        BLOCKING_MAX = 120  # Max angle for blocking arm (higher = more sensitive)
        CONFIDENCE_THRESHOLD = (
            0.6  # Final confidence threshold (lower = more sensitive)
        )

        # Confidence calculation weights - must sum to 1.0
        ANGLE_WEIGHT = 0.5
        FORWARD_WEIGHT = 0.3
        BLOCK_WEIGHT = 0.2

        # Get relevant landmarks for both arms
        right_shoulder = pose_landmarks[11]  # Left shoulder
        right_elbow = pose_landmarks[13]  # Left elbow
        right_wrist = pose_landmarks[15]  # Left wrist

        left_shoulder = pose_landmarks[12]  # Right shoulder
        left_elbow = pose_landmarks[14]  # Right elbow
        left_wrist = pose_landmarks[16]  # Right wrist

        # Check visibility of key points
        key_landmarks = [
            right_shoulder,
            right_elbow,
            right_wrist,
            left_shoulder,
            left_elbow,
            left_wrist,
        ]
        if any(lm.visibility < MIN_VISIBILITY for lm in key_landmarks):
            return False, 0.0, None

        # Check if body is sufficiently visible
        left_hip = pose_landmarks[23]
        right_hip = pose_landmarks[24]
        if (
            left_hip.visibility < HIP_VISIBILITY
            and right_hip.visibility < HIP_VISIBILITY
        ):
            return False, 0.0, None

        # Calculate arm angles
        left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self._calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )

        # Check forward arm extension with threshold
        left_arm_forward = left_wrist.z < (left_elbow.z - Z_THRESHOLD)
        right_arm_forward = right_wrist.z < (right_elbow.z - Z_THRESHOLD)

        # Check blocking position
        left_blocking = BLOCKING_MIN < left_elbow_angle < BLOCKING_MAX
        right_blocking = BLOCKING_MIN < right_elbow_angle < BLOCKING_MAX

        # Calculate punch confidences
        jab_confidence = 0.0
        cross_confidence = 0.0

        # JAB calculation (left arm punch)
        if left_elbow_angle > PUNCH_ARM_ANGLE and left_arm_forward and right_blocking:
            angle_confidence = min(
                1.0, max(0.0, (left_elbow_angle - PUNCH_ARM_ANGLE) / 50)
            )
            z_diff = left_elbow.z - left_wrist.z
            forward_confidence = min(1.0, max(0.0, z_diff * 10))
            block_confidence = min(
                1.0, max(0.0, (1.0 - abs(right_elbow_angle - 90) / 30))
            )

            jab_confidence = (
                angle_confidence * ANGLE_WEIGHT
                + forward_confidence * FORWARD_WEIGHT
                + block_confidence * BLOCK_WEIGHT
            )

        # CROSS calculation (right arm punch)
        if right_elbow_angle > PUNCH_ARM_ANGLE and right_arm_forward and left_blocking:
            angle_confidence = min(
                1.0, max(0.0, (right_elbow_angle - PUNCH_ARM_ANGLE) / 50)
            )
            z_diff = right_elbow.z - right_wrist.z
            forward_confidence = min(1.0, max(0.0, z_diff * 10))
            block_confidence = min(
                1.0, max(0.0, (1.0 - abs(left_elbow_angle - 90) / 30))
            )

            cross_confidence = (
                angle_confidence * ANGLE_WEIGHT
                + forward_confidence * FORWARD_WEIGHT
                + block_confidence * BLOCK_WEIGHT
            )

        # Determine best punch and check threshold
        if jab_confidence > cross_confidence:
            if jab_confidence > CONFIDENCE_THRESHOLD:
                return True, jab_confidence, "JAB"
        else:
            if cross_confidence > CONFIDENCE_THRESHOLD:
                return True, cross_confidence, "CROSS"

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
