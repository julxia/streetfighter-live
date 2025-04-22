# import Player
from recognition.models import RecognitionModels


class GameLogic:
    def __init__(self):
        # This gets the speech up and running. As well as initalizes the other variables
        # to access when the read() method is called.
        self.gameState = "start"
        self.gameMode = "menu"
        self.models = RecognitionModels()
        self.isSinglePlayer = None

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

    def start(self):
        self.gameState = "running"
        # Start gesture recognition and do not continue until the model is setup
        self.models.start("pose")
        while not (self.models.is_initialized("pose")):
            continue

    # def initialize(self):
    #     pass

    # def run(self):
    #     pass

    def read(self):
        voice_output = self.models.get_output("voice")
        pose_output = self.models.get_output("pose")

        if voice_output.output:
            print(">> Voice output: ", voice_output)

        if voice_output.output:
            # 1. For starting the game in single or multi player game
            if self.gameState == "start":
                if voice_output.output == "single player":
                    self.isSinglePlayer = True
                elif voice_output.output == "multiplayer":
                    self.isSinglePlayer = False
                else:
                    self.isSinglePlayer = None
                return self.isSinglePlayer

            # 2. For getting inputs while the game is running in single player
            elif self.gameState == "running":
                if voice_output.output == "exit":
                    return {"state": "terminate"}
                return {
                    "state": self.gameState,
                    "AttackType": pose_output.output,
                }

    def terminate(self):
        # Shut down Hao's pose model and update gameState to start
        self.models.stop("pose")
        self.gameState = "start"

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
