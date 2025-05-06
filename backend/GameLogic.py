# import Player
from recognition.models import RecognitionModels

# Multiplayer Changes
import time


class GameLogic:
    def __init__(self, multiplayer_client=None):
        # This gets the speech up and running. As well as initalizes the other variables
        # to access when the read() method is called.
        self.gameState = "start"
        self.gameMode = "menu"
        self.models = RecognitionModels()
        self.isSinglePlayer = None

        # Multiplayer Changes
        self.multiplayer_client = multiplayer_client  # Store the network client

        # Get speech recognition model started and running
        self.models.start("voice")
        # Ensures program does not proceed until speech recognition model is initialized
        while not (self.models.is_initialized("voice")):
            continue

        # Other vars we may need for V1
        # self.player1 = None
        # self.player2 = None
        # self.poseReady = False
        # self.speechReady = False

        # Multiplayer Changes
        self.player_health = 100
        self.opponent_health = 100
        self.last_attack_time = 0
        self.attack_cooldown = 0.5  # Seconds between attacks
        self.is_winner = False

    def start(self):
        self.gameState = "running"
        # Start gesture recognition and do not continue until the model is setup
        print("<--- GameLogic Game Starting... --->")
        self.models.start("pose")
        while not (self.models.is_initialized("pose")):
            continue

        # Multiplayer Changes
        # Reset health for new game
        self.player_health = 100
        self.opponent_health = 100

    # def initialize(self):
    #     pass

    # def run(self):
    #     pass

    def get_frame(self):
        return self.models.get_frame()

    # Multiplayer Changes
    def get_opponent_frame(self):
        if self.multiplayer_client and not self.isSinglePlayer:
            return self.multiplayer_client.get_opponent_frame()

        return None

    def read(self):
        voice_output = self.models.get_output("voice")
        pose_output = self.models.get_output("pose")
        # Multiplayer Changes
        current_time = time.time()

        voice_command = voice_output.output if voice_output.output else None

        if voice_command:
            print(">> Voice output: ", voice_output)

        # 1. For starting the game in single or multi player game
        if voice_command and self.gameState == "start":
            if voice_command == "Single Player":
                self.isSinglePlayer = True
                # Multiplayer Changes
                return True  # Single player selected
            elif voice_command == "Multiplayer":
                # Multiplayer Changes
                if self.multiplayer_client:
                    if self.multiplayer_client.connect():
                        self.isSinglePlayer = False
                        return False  # Multiplayer selected and connected
                    else:
                        print(f"Failed to connect to multiplayer server")
                        return None
                else:
                    print(f"Multiplayer client not initialized")
                    return None
            else:
                return None

        # 2. For getting inputs while the game is running in single player
        # Multiplayer Changes
        if self.gameState == "running":
            # Handle voice commands for exiting the game
            if voice_command and voice_command.lower() in {"exit", "quit"}:
                return {"state": "terminate"}

            if voice_command and voice_command.lower() in {"fire", "lightning", "ice"}:
                special_move = voice_command.lower()

                attack_result = {
                    "state": self.gameState,
                    "AttackType": "punch",
                }

                # In multiplayer mode, send current action to server
                if not self.isSinglePlayer and self.multiplayer_client:
                    if current_time - self.last_attack_time > self.attack_cooldown:
                        self.last_attack_time = current_time

                        # Send current frame and action to server
                        self.multiplayer_client.send_data(
                            attack_result, self.get_frame()
                        )

                        # Update health values from server
                        self.player_health, self.opponent_health = (
                            self.multiplayer_client.get_health()
                        )

                        # Add health info to result
                        attack_result["player_health"] = self.player_health
                        attack_result["opponent_health"] = self.opponent_health

                        # End game if either player reaches 0 HP.
                        if self.player_health <= 0 or self.opponent_health <= 0:
                            if self.opponent_health <= 0:
                                self.is_winner = True
                            return {"state": "terminate", "winner": self.is_winner}

                return attack_result

            # Handle pose outputs
            if pose_output:
                attack_result = {
                    "state": self.gameState,
                    "AttackType": pose_output.output,
                }

                # In multiplayer mode, send current action to server
                if not self.isSinglePlayer and self.multiplayer_client:
                    if (
                        pose_output.output
                        and current_time - self.last_attack_time > self.attack_cooldown
                    ):
                        self.last_attack_time = current_time

                        # Send current frame and action to server
                        self.multiplayer_client.send_data(
                            attack_result, self.get_frame()
                        )

                        # Update health values from server
                        self.player_health, self.opponent_health = (
                            self.multiplayer_client.get_health()
                        )

                        # Add health info to result
                        attack_result["player_health"] = self.player_health
                        attack_result["opponent_health"] = self.opponent_health

                        # End game if either player reaches 0 HP.
                        if self.player_health <= 0 or self.opponent_health <= 0:
                            if self.opponent_health <= 0:
                                self.is_winner = True
                            return {"state": "terminate", "winner": self.is_winner}
                    else:
                        # Send just the frame for synchronization when not attacking
                        self.multiplayer_client.send_data(None, self.get_frame())

                    # Check opponent's action
                    opponent_action = self.multiplayer_client.get_opponent_action()
                    if opponent_action:
                        attack_result["opponent_action"] = opponent_action

                return attack_result

    def terminate(self):
        # Shut down Hao's pose model and update gameState to start
        self.models.stop("pose")
        self.gameState = "start"

        # Multiplayer Changes
        if self.multiplayer_client and not self.isSinglePlayer:
            self.multiplayer_client.disconnect()

    # def initiateSinglePlayer(self):
    #     self.gameMode = "single"
    #     self.gameState = "ongoing"
    #     self.player1 = Player()

    # def initiateMultiPlayer(self):
    #     self.gameMode = "multi"
    #     self.gameState = "ongoing"
    #     self.player1 = Player()
    #     self.player2 = Player()

    # def updateGameState(self, newGameState):
    #     self.gameState = newGameState

    # def terminateGame(self):
    #     self.gameState = "terminated"

    # def getGameState(self):
    #     return self.gameState

    # def changeGameState(self):
    #     # Call this when either the game is finished (V1: One player's health reaches 0) or when the player
    #     # chooses to quit.
    #     inputCommand = self.speech.get_latest_speech["text"] # The text
    #     if self.gameMode == "menu":
    #         if inputCommand == "single player":
    #             self.initiateSinglePlayer()
    #         else:
    #             print(f"Only single player is supported at the moment...")

    #     if self.gameMode != "single":
    #         print("Only Single Player mode is supported at the moment.")
    #         return

    #     if self.gameState == "ongoing":
    #         if inputCommand == "continue":
    #             print("Game continues...")
    #         elif inputCommand == "terminate":
    #             print("Terminating game...")
    #             self.terminateGame()
    #         else:
    #             print(f"Unknown command: {inputCommand}")

    #     elif self.gameState == "terminated":
    #         print("Game has already ended. Returning to main menu.")
    #         self.gameMode = "menu"
    #         self.player1 = None
    #         self.player2 = None

    # def performAction(self):
    #     # Right now only in single player so only concerned w player 1
    #     # Call this method when the player is performing a fighting move.
    #     command = self.pose.get_gesture["pose"]
    #     return (self.player1, command, self.player1.moveset[command])
