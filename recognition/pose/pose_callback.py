from recognition.pose.gesture_recognizer import GestureRecognizer
from recognition.model_types import ModelOutput


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
