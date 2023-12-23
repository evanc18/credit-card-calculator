import numpy as np

class Card:
    def __init__(self, index):
        self.index = select_card(index)
        self.name = ''
        self.mul_points = np.zeros((9))
        self.mul_cashback = np.zeros((9))


    def select_card(self, index):
        # select a card from the list of cards
        if index == -1:
            # randomly select a card
            return np.random.randint(0, 3)
        
