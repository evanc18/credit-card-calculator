
class Card:
    def __init__(self, selection):
        self.selection = selection
        self.cost = 0
        self.points = 0
        self.cash = 0
        self.mutate()