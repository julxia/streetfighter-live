from typing import Literal

from recognition.model_types import ModelOutput
from recognition.pose.pose_recognition import PoseRecognition
from recognition.voice import VoiceRecognition, speech_callback, street_fighter_commands


class RecognitionModels:
    def __init__(self):
        self.models = {
            "pose": PoseRecognition(
                model_path="./models/pose/pose_landmarker_full.task",
                display_camera=True,
            ),
            "voice": VoiceRecognition(
                callback_function=speech_callback,
                phrases_to_detect=street_fighter_commands,
                confidence_threshold=0.5,
                listening_timeout=3,
                phrase_time_limit=2,
                pause_after_no_speech=1.0,
                energy_threshold=3500,
                dynamic_energy_adjustment=True,
            ),
        }

    def start(self, model: Literal["pose", "voice"]) -> None:

        self.models[model].start()

    def stop(self, model: Literal["pose", "voice"]) -> None:
        self.models[model].stop()

    def get_output(self, model: Literal["pose", "voice"]) -> ModelOutput:
        return self.models[model].get_output()

    def is_initialized(self, model: Literal["pose", "voice"]) -> bool:
        return self.models[model].is_initialized()

    def get_frame(self):
        return self.models["pose"].get_current_frame()
