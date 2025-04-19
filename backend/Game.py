import Player

class Game():
    def __init__(self):
        self.gameState = "ongoing"
        self.gameMode = "menu"
        self.player1 = None
        self.player2 = None
    
    def initiateSinglePlayer(self):
        self.gameMode = "Single"
        self.player1 = Player()

    def initiateMultiPlayer(self):
        self.gameMode = "Multi"
        self.player1 = Player()
        self.player2 = Player()

    def updateGameState(self, newGameState):
        self.gameState = newGameState
    
    def terminateGame(self):
        self.gameState = "Terminated"

    def runGame(self):
        # TODO: Figure out how to transfer into various different gamestates/update gamestates.
        pass

        
