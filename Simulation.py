import cupy as cp
import numpy as np
from Consoomer import *
from Budget import *
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class Simulation:
    def __init__(self, population_size=5000, generations=150, budgets=120, learning_rate=0.05):
        self.population_size = population_size
        self.generations = generations
        self.budgets = budgets
        self.learning_rate = learning_rate
        self.consoomers = [Consoomer(i) for i in range(self.population_size)]
        self.cards = None
        self.rewards = cp.zeros((self.population_size, self.generations))
    
    def run(self):
        points_mat = self.cards[:, -20:-10].astype(int)
        cashback_mat = self.cards[:, -10:].astype(int) 
        annual_fee_mat = self.cards[:, 1].astype(int)
        print(f'Initializing {len(self.consoomers)} consoomers and cards...')
        for c in self.consoomers:
            c.init_cards(self.cards)
            c.designate_cards()
        
        print(f'Running simulation with {self.generations} generations...')
        for g in range(self.generations):
            gen_budgets = Budget(self.budgets).test_budgets
            for i in tqdm(range(0, len(gen_budgets), 12), desc=f'Generation {g+1}/{self.generations}', ncols=100):
                year = cp.transpose(gen_budgets[i:i+12])
                #print(year)
                # Create 3D matrix for purchases
                points = cp.array([cp.diag(points_mat[c.purchases, :]) for c in self.consoomers]).reshape(len(self.consoomers), 10, 1)
                cashback = cp.array([cp.diag(cashback_mat[c.purchases, :]) for c in self.consoomers]).reshape(len(self.consoomers), 10, 1)
                points = cp.sum(points * year, axis=1)
                cashback = cp.sum(cashback * year, axis=1)

                # Convert points to cash
                points = points / 100

                annual_fee = cp.array([c.calc_annual_fee(annual_fee_mat) for c in self.consoomers])
                #print(annual_fee)

                # Update rewards
                self.rewards[:, g] = cp.sum(points + cashback, axis=1)  + annual_fee

            # breed the population, select the top x% of the population and breed them with each other
            #print standard deviation of rewards, max, min, mean all on one line
            print('Generation: {}, Mean: {:.2f}, Max: {:.2f}, Min: {:.2f}, Std: {:.2f}'.format(g+1, cp.mean(self.rewards[:, g]).item(), cp.max(self.rewards[:, g]).item(), cp.min(self.rewards[:, g]).item(), cp.std(self.rewards[:, g]).item()))
            print('Breeding and mutating...')
            breed_rate = math.floor(1/self.learning_rate)
            div = int(self.population_size/breed_rate)
            breeders = cp.argsort(self.rewards[:, g])[-div:]
            bottom = cp.argsort(self.rewards[:, g])[:div]
            # the middle % of the population mutate their cards
            mutators = cp.argsort(self.rewards[:, g])[div:-div]
            for j in range(0, len(breeders)-1, 2):
                #bottom 25% of population go broke and are replaced with children of top 25%
                #print(self.consoomers)
                self.consoomers[bottom[j].item()]  = self.consoomers[breeders[j].item()].breed(self.consoomers[breeders[j+1].item()])
            for j in range(0, len(mutators)):
                self.consoomers[mutators[j].item()].mutate_cards()

    def load_cards(self):
        dat = pd.read_csv('cards.csv')
        self.cards = cp.asarray(dat.iloc[0:30, 2:].to_numpy())
        self.cards = cp.nan_to_num(self.cards)
        
        
    def display_results(self):
        # Create a figure with 2 subplots
        fig, axs = plt.subplots(4)
        plt.subplots_adjust(hspace=1)

        # Graph 1: Avg Reward vs Generation
        avg_rewards = cp.mean(self.rewards, axis=0)
        axs[0].plot(range(self.generations), avg_rewards.get(), label='Average Reward')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Average Reward')
        axs[0].set_title('Average Reward vs Generation')
        axs[0].legend()

        # Graph 2: Generation 1 reward distribution
        axs[1].hist(self.rewards[:,0].get(), bins=max(100, self.population_size // 1000))
        mean_reward = cp.mean(self.rewards[:,0])
        axs[1].axvline(mean_reward.get(), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reward.get():.2f}')
        axs[1].set_xlabel('Reward')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Reward Distribution for Generation 1')
        axs[1].legend()

        # Graph 3: Last generation reward distribution
        axs[2].hist(self.rewards[:,-1].get(), bins=max(100, self.population_size // 1000))
        mean_reward = cp.mean(self.rewards[:,-1])
        axs[2].axvline(mean_reward.get(), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reward.get():.2f}')
        axs[2].set_xlabel('Reward')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title(f'Reward Distribution for Generation {self.generations}')
        axs[2].legend()

        # Graph 4: Purchases of top 5 consoomers
        top_5_indices = cp.argsort(self.rewards.sum(axis=1))[-5:]
        top_5_indices = cp.asnumpy(top_5_indices).tolist()
        top_5_purchases = [cp.asnumpy(self.consoomers[i].purchases) for i in top_5_indices]
        print(top_5_purchases)
        axs[3].set_xlabel('Category')
        axs[3].set_ylabel('Card')
        axs[3].imshow(top_5_purchases, cmap='hot', interpolation='nearest')
        axs[3].set_title('Purchases of Top 5 Consoomers')

        # Display the plots
        plt.show()


if __name__ == '__main__':
    # Run the simulation
    print(cp.__version__)
    print(cp.cuda.runtime.runtimeGetVersion())
    try:
        device_id = cp.cuda.get_device_id()
        print(f'CuPy is using GPU device ID: {device_id}')
    except cp.cuda.runtime.CUDARuntimeError as e:
        print('CuPy is not using a GPU:', e)
    sim = Simulation()
    sim.load_cards()
    sim.run()
    sim.display_results()