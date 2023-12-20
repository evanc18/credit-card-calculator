import numpy as np

class Consoomer:
    def __init__(self, name):
        self.name = name
        self.population = None
        self.cards = []
        self.purchases = np.zeros()
        self.cash = 0
        self.points = []

    def init_cards(self):
        num_cards = np.random.randint(1, 5)
        for i in range(num_cards):
            card = Card(selection=-1)
            self.cards.append(card)
    
    def init_purchases(self):
        num_purchases = np.random.randint(1, 5)
        for i in range(num_purchases):
            purchase = Purchase()
            self.purchases.append(purchase)
    
    def mutate_cards(self):
    
    def mutate_