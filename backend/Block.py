class Block():
    def __init__(self):
        self.damage = 0
        self.is_low = False
        self.is_high = False

class LowBlock(Block):
    def __init__(self):
        super().__init__()
        self.is_low = True

class HighBlock():
    def __init__(self):
        super().__init__()
        self.is_high = True