import cupy as cp
import numpy as np
import Card

class Consoomer:
    def __init__(self, index):
        self.index = index
        self.population = None
        self.cards = []
        self.purchases = cp.zeros((10), dtype=int)
        self.fees = 0
        self.points = cp.zeros((10), dtype=int)

    def init_cards(self, cards):
        num_cards = cp.random.randint(1, 5).item()
        for i in range(num_cards):
            self.cards.append(cp.random.randint(0, cp.shape(cards)[0]))
        self.cards = cp.asarray(self.cards)
    
    def designate_cards(self):
        # randomly assign cards from self.cards into self.purchases repeats allowed
        self.purchases = cp.random.choice(self.cards, len(self.purchases)).astype(int)

    def mutate_cards(self, retention=0.80, shuffle=0.75):
        """
        retention: probability of keeping all cards from previous generation
        shuffle: probability of using same card for a given category from previous generation
        """
        num_cards = len(self.cards)
        num_purchases = len(self.purchases)
        retention_rand = cp.random.rand(num_cards)
        shuffle_rand = cp.random.rand(num_purchases)
        for i in range(num_cards):
            cur_card = self.cards[i]
            if cp.random.random() >= retention_rand[i]:
                new_card = cp.random.randint(0, num_cards)
                self.cards[i] = int(new_card)
                mask = self.purchases == cur_card
                self.purchases[mask] = new_card
        # shuffle purchases
        for i in range(num_purchases):
            if cp.random.random() >= shuffle_rand[i]:
                self.purchases[i] = cp.random.choice(self.cards, 1)

    def breed(self, mate):
        new_cons = Consoomer(-1)
        #pick length of cards as random choice between the parents length
        new_cons.cards = cp.zeros((np.random.choice([len(self.cards), len(mate.cards)])))
        for i in range(len(new_cons.cards)):
            if i < len(mate.cards) and i < len(self.cards):
                if cp.random.random() >= 0.5:
                    new_cons.cards[i] = mate.cards[i]
                else:
                    new_cons.cards[i] = self.cards[i]
            elif i >= len(mate.cards) and i < len(self.cards):
                new_cons.cards[i] = self.cards[i]
            elif i < len(mate.cards) and i >= len(self.cards):
                new_cons.cards[i] = mate.cards[i]
        new_cons.designate_cards()
        return new_cons
    
    def calc_annual_fee(self, annual_fee_mat, first_year=True):
        self.fees = 0
        for i in range(len(self.cards)):
            self.fees -= annual_fee_mat[int(self.cards[i])]
        return self.fees
    

            