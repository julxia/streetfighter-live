# import cv2
# import time

# from pose_recognition import PoseRecognition


# # Example usage
# def example_usage():
#     """Example of how to use the PoseRecognition API"""
#     # Initialize pose recognition
#     pose_system = PoseRecognition(
#         model_path="./models/pose/pose_landmarker_full.task", display_camera=False
#     )

#     try:
#         print("Starting pose recognition...")
#         print("Pose System initialized?", pose_system.is_initialized())

#         # Start pose recognition in a separate thread
#         pose_system.start(threaded=True)

#         print("Pose System initialized?", pose_system.is_initialized())

#         # Game loop
#         score = 0
#         last_gesture = None

#         while True:
#             # Get the current detected gesture as JSON
#             gesture_data = pose_system.get_output()

#             # Print the JSON data for demonstration
#             if gesture_data.output:
#                 print(f"Gesture data: {gesture_data}")

#             # Update game state based on gesture
#             if gesture_data.output and gesture_data.output != last_gesture:
#                 print(
#                     f"Detected: {gesture_data.output} with confidence {gesture_data.confidence}"
#                 )
#                 score += int(gesture_data.confidence * 10)  # Score based on confidence
#                 last_gesture = gesture_data.output

#             # Exit on keyboard interrupt
#             if cv2.waitKey(5) & 0xFF == ord("q"):
#                 break

#             time.sleep(0.01)  # Small sleep to prevent CPU overuse

#     except KeyboardInterrupt:
#         print("Stopped by user")
#     finally:
#         # Clean up
#         pose_system.stop()
#         print(f"Final score: {score}")


# if __name__ == "__main__":
#     try:
#         example_usage()
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         import traceback

#         traceback.print_exc()

# __all__ = ["PoseRecognition"]
