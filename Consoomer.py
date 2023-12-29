import numpy as np
from multiprocessing import Pool
from copy import deepcopy

class Consoomer:
    def __init__(self, index, purchases=np.zeros((10), dtype=int)):
        self.index = index
        self.population = None
        self.cards = []
        self.purchases = purchases
        self.fees = 0
        self.points = np.zeros((10), dtype=int)

    def init_cards(self, cards_ref):
        num_cards = np.random.randint(3, 6)
        for i in range(num_cards):
            if i == 0:
                self.cards.append(23) #amex gold i already have this one
            else:
                card = np.random.randint(0, np.shape(cards_ref)[0])
                while card in self.cards:
                    card = np.random.randint(0, np.shape(cards_ref)[0])
                self.cards.append(card)
        self.cards = np.asarray(self.cards)

    def designate_cards(self):
        # randomly assign cards from self.cards into self.purchases repeats allowed
        self.purchases = np.random.choice(self.cards, len(self.purchases)).astype(int)
    
    def shuffle_cards(self, shuffle=0.75):
        # randomly assign cards from self.cards into self.purchases repeats allowed4
        #print(f"Shuffling cards: {self.cards} with distribution {self.purchases}")
        for i in range(len(self.purchases)):
            if np.random.random() >= shuffle:
                self.purchases[i] = np.random.choice(self.cards)
       #
        #print(f"Shuffled cards: {self.cards} with distribution {self.purchases}")
        return self

    def mutate_cards(self, retention=0.70):
        """
        retention: probability of keeping all cards from previous generation
        shuffle: probability of using same card for a given category from previous generation
        """
        num_cards = len(self.cards)
        num_purchases = len(self.purchases)
        for i in range(num_cards):
            cur_card = self.cards[i]
            if np.random.random() >= retention:
                new_card = np.random.randint(0, self.card_mat_len)
                while new_card in self.cards:
                    #print(f"New card {new_card} already in cards {self.cards}")
                    new_card = np.random.randint(0, self.card_mat_len)
                self.cards[i] = int(new_card)
                mask = self.purchases == cur_card
                self.purchases[mask] = new_card
                #print(f"Mutated card: {cur_card} to {new_card}")
        # shuffle purchases
        self = self.shuffle_cards()
        #print(f"Mutated cards: {self.cards} with distribution {self.purchases}")
        return self
    
    def breed(self, mate, card_mat_len, index=-1):
        new_cons = deepcopy(self)
        new_cons.cards = []
        new_cons.purchases = np.zeros((10), dtype=int)
        new_cons.card_mat_len = card_mat_len
        #pick length of cards as random choice between the parents length
        off = 0
        #print(f"\nStart breed: {self.cards}, {mate.cards}")
        for i in range(len(new_cons.purchases)):
            if len(new_cons.cards) < 6:
                #print(self.purchases[i], mate.purchases[i])
                new_cons.purchases[i] = np.random.choice([self.purchases[i], mate.purchases[i]])
                #print(f"New purchase: {new_cons.purchases[i]}")
                if not new_cons.purchases[i] in new_cons.cards:
                     new_cons.cards = np.append(new_cons.cards, new_cons.purchases[i])
            else:
                #start copying from the front of the new_cons.purchases array
                #for example 
                #[1,2,3,4,5,x,x,x]
                #if we hit 5 card limit at 5, fill in the x's with 1,2,3
                new_cons.purchases[i] = new_cons.purchases[0+off]
                off += 1
        if len(new_cons.cards) == 1:
            new_cons.cards = np.append(new_cons.cards, np.random.randint(0, card_mat_len-1))
        #print(f"End breed: {new_cons.cards}, {new_cons.purchases}")
        #print("Start mutate")
        new_cons = new_cons.mutate_cards()
        return new_cons
    
    def calc_annual_fee(self, annual_fee_mat, first_year=True):
        self.fees = 0
        for i in range(len(self.cards)):
            self.fees -= annual_fee_mat[int(self.cards[i])]
        return self.fees
    
    def calc_bonus_offer(self, offer_points_mat):
        self.points = 0
        for i in range(len(self.cards)):
            self.points += offer_points_mat[int(self.cards[i])]
        return self.points
    

            