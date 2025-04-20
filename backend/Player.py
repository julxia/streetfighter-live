import Block
import Kick
import Punch

# global dict of movesets
MOVESETDICT = {
    "punch": Punch(),
    "kick": Kick(),
    "block": Block()
}

class Player():
    def __init__(self):
        self.health = 100
        self.moves = MOVESETDICT

    def action(self, input):
        #V0: Single player. Don't worry about blocking behavior right now.
        return (input, self.moves[input].damage)
    
    def getHealth(self):
        return self.health
    
    def takeDamage(self, damageRecieved):
        self.health -= damageRecieved