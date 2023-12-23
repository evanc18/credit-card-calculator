import numpy as np
import Card

class Consoomer:
    def __init__(self, index):
        self.index = index
        self.population = None
        self.cards = []
        self.purchases = np.zeros((10), dtype=int)
        self.fees = 0
        self.points = np.zeros((10), dtype=int)

    def init_cards(self, cards):
        num_cards = np.random.randint(2, 5)
        for i in range(num_cards):
            self.cards.append(np.random.randint(0, np.shape(cards)[0]))
        self.cards = np.asarray(self.cards)

    def designate_cards(self):
        # randomly assign cards from self.cards into self.purchases repeats allowed
        self.purchases = np.random.choice(self.cards, len(self.purchases)).astype(int)

    def mutate_cards(self, retention=0.90, shuffle=0.80):
        """
        retention: probability of keeping all cards from previous generation
        shuffle: probability of using same card for a given category from previous generation
        """
        num_cards = len(self.cards)
        num_purchases = len(self.purchases)
        retention_rand = np.random.rand(num_cards)
        shuffle_rand = np.random.rand(num_purchases)
        for i in range(num_cards):
            cur_card = self.cards[i]
            if np.random.random() >= retention_rand[i]:
                new_card = np.random.randint(0, self.card_mat_len-1)
                while new_card in self.cards:
                    new_card = np.random.randint(0, self.card_mat_len-1)
                self.cards[i] = int(new_card)
                mask = self.purchases == cur_card
                self.purchases[mask] = new_card
        # shuffle purchases
        for i in range(num_purchases):
            if np.random.random() >= shuffle_rand[i]:
                self.purchases[i] = np.random.choice(self.cards)

    def breed(self, mate, card_mat_len):
        new_cons = Consoomer(-1)
        new_cons.card_mat_len = card_mat_len
        #pick length of cards as random choice between the parents length
        off = 0
        for i in range(len(new_cons.purchases)):
            if len(new_cons.cards) <= 5:
                new_cons.purchases[i] = np.random.choice([self.purchases[i], mate.purchases[i]])
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

        new_cons.mutate_cards()
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
    

            