
import numpy as np
from Consoomer import *
from Budget import *
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

CARD_MAT_LEN = 0

class Simulation:
    def __init__(self, population_size=100, generations=4000, budgets=120, elitism_rate=0.10):
        self.population_size = population_size
        self.generations = generations
        self.budgets = budgets
        #self.learning_rate = learning_rate
        self.elitism_rate = elitism_rate
        self.consoomers = [Consoomer(i) for i in range(self.population_size)]
        self.cards = None
        self.rewards = np.zeros((self.population_size, self.generations))
    
    def run(self):
        points_mat = self.cards[:, -20:-10].astype(int)
        cashback_mat = self.cards[:, -10:].astype(int) 
        annual_fee_mat = self.cards[:, 1].astype(int)
        anniv_points_mat = self.cards[:, 5].astype(int)
        offer_points_mat = self.cards[:, 6].astype(int)
        print(f'Initializing {len(self.consoomers)} consoomers and cards...')
        for c in self.consoomers:
            c.init_cards(self.cards)
            c.designate_cards()
        
        print(f'Running simulation with {self.generations} generations...')
        card_use_mat = np.zeros((len(self.cards), self.generations), dtype=int)
        for g in range(self.generations):
            #COUNT HOW MANY TIMES EACH CARD IS POSSESSED BY A CONSUMER
            for c in self.consoomers:
                for i in range(len(c.cards)):
                    card_use_mat[int(c.cards[i]), g] += 1


            gen_budgets = Budget(self.budgets).test_budgets
            for i in tqdm(range(0, len(gen_budgets), 12), desc=f'Generation {g+1}/{self.generations}', ncols=100):
                year = np.transpose(gen_budgets[i:i+12])
                #print(year)
                # Create 3D matrix for purchases
                points = np.array([np.diag(points_mat[c.purchases, :]) for c in self.consoomers]).reshape(len(self.consoomers), 10, 1)
                cashback = np.array([np.diag(cashback_mat[c.purchases, :]) for c in self.consoomers]).reshape(len(self.consoomers), 10, 1)

                points = np.sum(points * year, axis=1)
                cashback = np.sum(cashback * year, axis=1)

                bonus_p = np.zeros((len(self.consoomers)))
                if i == 0:
                    #we are adding the bonus offer for the first year
                    bonus_p = np.array([c.calc_bonus_offer(offer_points_mat) for c in self.consoomers])
                else:
                    bonus_p = np.array([c.calc_bonus_offer(anniv_points_mat) for c in self.consoomers])

                # Convert points to cash
                bonus_p = bonus_p / 100
                points = points / 100

                annual_fee = np.array([c.calc_annual_fee(annual_fee_mat) for c in self.consoomers])
                #print(annual_fee)

                # Update rewards
                self.rewards[:, g] = np.sum(points + cashback, axis=1)  + annual_fee + bonus_p

            print('Generation: {}, Mean: {:.2f}, Max: {:.2f}, Min: {:.2f}, Std: {:.2f}'.format(g+1, np.mean(self.rewards[:, g]), np.max(self.rewards[:, g]), np.min(self.rewards[:, g]), np.std(self.rewards[:, g])))
            #print('Breeding and mutating...')

            # Select the population for breeding via rank method
            sorted = np.argsort(self.rewards[:, g])
            elites = math.floor(self.elitism_rate * len(sorted))
            

            # SELECTION VIA RANK
            #tot = sum(sorted)+len(sorted)
            #prob = np.array([(i+1)/tot for i in range(len(sorted))])
            #breeders = np.random.choice(sorted, size=math.floor((len(sorted)-elites)/2), p=prob)
            

            # SELECTION VIA FITNESS ROULETTE
            pos_rwds = self.rewards[:, g] - min(self.rewards[:, g])
            tot = sum(pos_rwds[:])
            prob = np.array(pos_rwds/tot)
            breeders = np.random.choice(sorted, size=math.floor((len(sorted)-elites)/2), p=prob)

            print("Best consoomer: {}".format(self.consoomers[sorted[-1]].cards))
            for j in range(0, len(breeders)-1):
                self.consoomers[elites+j]  = self.consoomers[breeders[j]].breed(self.consoomers[breeders[j+1]], np.shape(self.cards)[0])
            
            self.consoomers[0:elites] = [self.consoomers[i] for i in sorted[-elites:]]

        np.savetxt("card_use_mat.csv", card_use_mat, delimiter=",", fmt='%.0f')
            
            

    def load_cards(self):
        dat = pd.read_csv("C:/Users/EvanChase/Documents/Repos/credit-card-calculator/cards.csv")
        self.cards = np.asarray(dat.iloc[0:30, 2:].to_numpy())
        self.cards = np.nan_to_num(self.cards)
        
    def display_results(self):
        # Create a figure with 2 subplots
        fig, axs = plt.subplots(ncols=3, nrows=2)
        plt.subplots_adjust(hspace=1)

        # Graph 0: Max reward vs Generation
        max_rewards = np.max(self.rewards, axis=0)
        print(max_rewards)
        axs[0, 0].plot(range(self.generations), max_rewards, label='Max Reward')
        axs[0, 0].set_xlabel('Generation')
        axs[0, 0].set_ylabel('Max Reward')
        axs[0, 0].set_title('Max Reward vs Generation')
        axs[0, 0].legend()

        # Graph 1: Avg Reward vs Generation
        avg_rewards = np.mean(self.rewards, axis=0)
        axs[1, 0].plot(range(self.generations), avg_rewards, label='Average Reward')
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Average Reward')
        axs[1, 0].set_title('Average Reward vs Generation')
        axs[1, 0].legend()


        # Graph 2: Generation 1 reward distribution
        axs[0, 1].hist(self.rewards[:,0], bins=max(1000, self.population_size // 1000))
        mean_reward = np.mean(self.rewards[:,0])
        axs[0, 1].axvline(mean_reward, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reward:.2f}')
        axs[0, 1].set_xlabel('Reward')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Reward Distribution for Generation 1')
        axs[0, 1].legend()

        # Graph 3: Last generation reward distribution
        axs[1, 1].hist(self.rewards[:,-1], bins=max(1000, self.population_size // 1000))
        mean_reward = np.mean(self.rewards[:,-1])
        axs[1, 1].axvline(mean_reward, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reward:.2f}')
        axs[1, 1].set_xlabel('Reward')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].set_title(f'Reward Distribution for Generation {self.generations}')
        axs[1, 1].legend()

        # Graph 4: Purchases of top 20 consoomers
        top_5_indices = np.argsort(self.rewards.sum(axis=1))[-20:]
        top_5_indices = top_5_indices.tolist()
        top_5_purchases = [self.consoomers[i].purchases for i in top_5_indices]
        print(top_5_purchases)
        axs[0, 2].set_xlabel('Category')
        axs[0, 2].set_ylabel('Card')
        axs[0, 2].imshow(top_5_purchases, cmap='hot', interpolation='nearest')
        axs[0, 2].set_title('Purchases of Top 5 Consoomers')

        # Display the plots
        plt.show()


if __name__ == '__main__':
    # Run the simulation
    sim = Simulation()
    sim.load_cards()
    sim.run()
    sim.display_results()