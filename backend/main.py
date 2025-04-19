import Game

def main():
    print(f"Starting game...")
    game = Game()
    while game.gameState != "Terminated":
        game.runGame()