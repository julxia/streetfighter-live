import speech_recognition as sr
import threading
import queue
import time

from recognition.model_types import ModelOutput


class VoiceRecognition:
    def __init__(
        self,
        callback_function=None,
        phrases_to_detect=None,
        confidence_threshold=0.5,
        listening_timeout=5,
        phrase_time_limit=3,
        pause_after_no_speech=1.5,  # Pause duration after failed recognition
        energy_threshold=4000,  # Microphone sensitivity
        dynamic_energy_adjustment=True,
    ):
        """
        Initialize the background speech recognizer

        Args:
            callback_function: Function to call when speech is recognized
            phrases_to_detect: List of phrases to specifically look for
            confidence_threshold: Minimum confidence level to accept recognition (0.0 to 1.0)
            listening_timeout: How long to listen before timing out
            phrase_time_limit: Maximum length of a phrase to detect
            pause_after_no_speech: Time to pause after no speech is detected
            energy_threshold: Microphone sensitivity (higher = less sensitive)
            dynamic_energy_adjustment: Whether to adjust energy threshold automatically
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback_function
        self.phrases = phrases_to_detect
        self.confidence_threshold = confidence_threshold
        self.listening_timeout = listening_timeout
        self.phrase_time_limit = phrase_time_limit
        self.pause_after_no_speech = pause_after_no_speech

        # Set initial energy threshold
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_adjustment

        # Queue for communication between threads
        self.speech_queue = queue.Queue()

        # Flag to control the background thread
        self.is_running = False
        self.thread = None

        # Track initialization status
        self._initialized = False

        # For performance monitoring
        self.recognition_attempts = 0
        self.successful_recognitions = 0

        # Adjust for ambient noise
        self._calibrate_for_ambient_noise()

    def _calibrate_for_ambient_noise(self):
        """Calibrate the recognizer for ambient noise"""
        try:
            with self.microphone as source:
                print("Adjusting for ambient noise... Please be quiet.")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print(
                    f"Ambient noise adjustment complete. Energy threshold: {self.recognizer.energy_threshold}"
                )
                self._initialized = True
        except Exception as e:
            print(f"Error calibrating for ambient noise: {e}")
            self._initialized = False

    def start(self):
        """Start the background recognition thread"""
        if self.thread is not None and self.thread.is_alive():
            print("Recognition already running!")
            return

        # Check if initialization was successful
        if not self._initialized:
            print(
                "Warning: System not properly initialized. Attempting to initialize again."
            )
            self._calibrate_for_ambient_noise()
            if not self._initialized:
                print("Error: Failed to initialize voice recognition system.")
                return False

        self.is_running = True
        self.thread = threading.Thread(target=self._recognize_in_background)
        self.thread.daemon = True  # Thread will close when main program exits
        self.thread.start()
        print("Background speech recognition started")
        return True

    def stop(self):
        """Stop the background recognition thread"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
            print("Background speech recognition stopped")

            # Print performance stats
            if self.recognition_attempts > 0:
                success_rate = (
                    self.successful_recognitions / self.recognition_attempts
                ) * 100
                print(
                    f"Recognition stats: {self.successful_recognitions}/{self.recognition_attempts} successful ({success_rate:.1f}%)"
                )

    def _recognize_in_background(self):
        """Background thread function that continuously listens for speech"""
        while self.is_running:
            try:
                with self.microphone as source:
                    print("Listening...")
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=self.listening_timeout,
                            phrase_time_limit=self.phrase_time_limit,
                        )
                    except sr.WaitTimeoutError:
                        print("No speech detected within timeout period")
                        time.sleep(0.5)  # Small pause to prevent CPU overuse
                        continue

                self.recognition_attempts += 1

                try:
                    # Always get the full results with confidence scores
                    result = self.recognizer.recognize_google(
                        audio, language="en-US", show_all=True
                    )

                    # Check if we got any results
                    if not result or "alternative" not in result:
                        print("No recognition results")
                        time.sleep(
                            self.pause_after_no_speech
                        )  # Wait before listening again
                        continue

                    # Get the best result and its confidence
                    alternatives = result["alternative"]
                    best_result = alternatives[0]
                    transcript = best_result["transcript"]

                    # Google doesn't always provide confidence for all results
                    confidence = best_result.get("confidence", 0.0)

                    # If no confidence from Google, estimate from alternative count
                    if confidence == 0.0 and len(alternatives) > 1:
                        # Simple estimation based on position in alternatives list
                        confidence = 1.0 / len(alternatives)

                    print(
                        f"Recognized: '{transcript}' with confidence {confidence:.2f}"
                    )

                    # Check against confidence threshold
                    if confidence < self.confidence_threshold:
                        print(f"Confidence too low ({confidence:.2f}), ignoring")
                        time.sleep(
                            self.pause_after_no_speech
                        )  # Wait before listening again
                        continue

                    # If we have specific phrases to detect, check against them
                    if self.phrases:
                        matched_phrase = None
                        highest_similarity = 0

                        # Check for exact matches and close matches
                        for phrase in self.phrases:
                            similarity = self._calculate_text_similarity(
                                phrase.lower(), transcript.lower()
                            )
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                matched_phrase = phrase

                        # Require higher threshold for phrase matching
                        phrase_threshold = 0.7  # Adjust as needed

                        if highest_similarity >= phrase_threshold:
                            self.successful_recognitions += 1
                            timestamp = time.time()
                            match_confidence = highest_similarity * confidence
                            result_tuple = (timestamp, matched_phrase, match_confidence)

                            print(
                                f"Detected command: {matched_phrase} with confidence {match_confidence:.2f}"
                            )

                            # Add to queue and callback if provided
                            self.speech_queue.put(result_tuple)
                            if self.callback:
                                self.callback(result_tuple)
                        else:
                            print(
                                f"No close match found, best was {matched_phrase} with similarity {highest_similarity:.2f}"
                            )
                            time.sleep(
                                self.pause_after_no_speech
                            )  # Wait before listening again
                    else:
                        # General speech recognition
                        self.successful_recognitions += 1
                        timestamp = time.time()
                        result_tuple = (timestamp, transcript, confidence)

                        # Add to queue and callback if provided
                        self.speech_queue.put(result_tuple)
                        if self.callback:
                            self.callback(result_tuple)

                except sr.UnknownValueError:
                    print("Speech not understood")
                    time.sleep(
                        self.pause_after_no_speech
                    )  # Wait before listening again
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    time.sleep(2)  # Longer wait for API errors
                except KeyError:
                    print("Unexpected response format from Google API")
                    time.sleep(1)

            except Exception as e:
                print(f"Error in speech recognition: {e}")
                time.sleep(1)  # Prevent tight loop if continuous errors

    def get_output(self) -> ModelOutput:
        """Non-blocking method to get the latest recognized speech"""
        if not self.speech_queue.empty():
            _, text, confidence = self.speech_queue.get()
            return ModelOutput(output=text, confidence=confidence)
        return ModelOutput(output=None, confidence=1.0)

    def recalibrate(self):
        """Recalibrate the recognizer during runtime if needed"""
        self._calibrate_for_ambient_noise()

    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings
        Simple implementation using character-level comparison
        Returns value between 0 and 1, where 1 is exact match
        """
        # For better accuracy, consider using more sophisticated algorithms
        # like Levenshtein distance or word-level comparison

        # Simple word presence check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def is_initialized(self):
        """
        Check if the voice recognition system is properly initialized

        Returns:
            bool: True if the system is initialized and ready to use
        """
        return self._initialized and self.thread is not None and self.thread.is_alive()


# Example callback with different confidence handling
def speech_callback(result):
    timestamp, text, confidence = result

    print(f"Game received: '{text}' (confidence: {confidence:.2f})")

    # Example of handling different confidence levels
    if confidence > 0.8:
        print(f"HIGH confidence command: {text}")
        # Implement high-confidence game actions
    elif confidence > 0.6:
        print(f"MEDIUM confidence command: {text}")
        # Implement medium-confidence game actions
    else:
        print(f"LOW confidence command: {text}")
        # Implement low-confidence game actions or ask for confirmation


# Commands for Street Fighter game
street_fighter_commands = [
    "Single Player",
    "Multiplayer",
    "Hadouken",
    "Lightning",
    "Ice",
    "Pause",
    "Resume",
    "Exit",
    "Fire",
    "Lightning",
    "Ice",
    "Quit",
]


# Create and start the recognizer
def main():
    # Create speech recognizer with optimized parameters
    speech_recognizer = VoiceRecognition(
        callback_function=speech_callback,
        phrases_to_detect=street_fighter_commands,
        confidence_threshold=0.5,
        listening_timeout=3,
        phrase_time_limit=2,
        pause_after_no_speech=1.0,
        energy_threshold=3500,
        dynamic_energy_adjustment=True,
    )

    print("Speech Recognition initialized?", speech_recognizer.is_initialized())

    speech_recognizer.start()

    print("Speech Recognition initialized?", speech_recognizer.is_initialized())

    # Your game loop would go here
    try:
        while True:
            # Simulate game loop
            time.sleep(0.1)

            # Example of checking for commands in main loop
            output = speech_recognizer.get_output()
            if output:
                text, confidence = output["output"], output["confidence"]

                # Game logic based on command and confidence
                if "hadouken" in text.lower():
                    power = int(confidence * 100)  # Scale confidence to damage
                    print(f"Hadouken attack! Power: {power}")
                elif "ice" in text.lower():
                    duration = confidence * 5  # Scale confidence to effect duration
                    print(f"Ice attack! Freeze duration: {duration:.1f}s")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        speech_recognizer.stop()


if __name__ == "__main__":
    main()
