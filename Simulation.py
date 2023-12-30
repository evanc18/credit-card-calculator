
import numpy as np
from Consoomer import *
from Budget import *
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy import stats


CARD_MAT_LEN = 0

class Simulation:
    
    def __init__(self, population_size=300, generations=5000, budgets=6*12, elitism_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.budgets = budgets
        #self.learning_rate = learning_rate
        self.elitism_rate = elitism_rate
        print("Initializing simulation...")
        self.consoomers = [Consoomer(i) for i in range(self.population_size)]

        self.cards_ref = None
        self.rewards = np.zeros((self.population_size, self.generations))
    
    def run(self):
        self.points_mat = self.cards_ref[:, -22:-11].astype(int)
        print(np.shape(self.points_mat))
        self.cashback_mat = self.cards_ref[:, -11:].astype(float) 
        self.annual_fee_mat = self.cards_ref[:, 3].astype(int)
        self.anniv_points_mat = self.cards_ref[:, 4].astype(int)
        self.offer_points_mat = self.cards_ref[:, 5].astype(int)
        self.offer_cash_mat = self.cards_ref[:, 6].astype(int)
        print(f'Initializing {len(self.consoomers)} consoomers and cards...')
        for c in self.consoomers:
            c.init_cards(self.cards_ref)
            #print(f"Initialized cards: {c.cards}")
            c.designate_cards()
        
        print(f'Running simulation with {self.generations} generations...')
        card_use_mat = np.zeros((len(self.cards_ref), self.generations), dtype=int)
        mode_mat = np.zeros((len(self.cards_ref), 11))
        for g in range(self.generations):
            m = 0
            while m < len(self.consoomers):
                c = self.consoomers[m]
                for i in range(len(c.cards)):
                    card_use_mat[int(c.cards[i]), g] += 1
                for p in range(len(c.purchases)):
                    mode_mat[c.purchases[p], p] += 1
                m += 1
            #print(card_use_mat[:, g])
            mode_cards = [np.argmax(mode_mat[:, i]) for i in range(11)]

            gen_budgets = Budget(self.budgets).test_budgets
            for i in tqdm(range(0, len(gen_budgets), 12), desc=f'Generation {g+1}/{self.generations}', ncols=100):
                self.year = np.transpose(gen_budgets[i:i+12])
                self.i = i
                self.rewards[:, g] = np.array([self.run_sim_for_consoomer(c) for c in self.consoomers]).reshape(len(self.consoomers))
            

            print('Generation: {}, Mean: {:.2f}, Max: {:.2f}, Min: {:.2f}, Std: {:.2f}'.format(g+1, np.mean(self.rewards[:, g]), np.max(self.rewards[:, g]), np.min(self.rewards[:, g]), np.std(self.rewards[:, g])))
            print(f'Mode: {mode_cards}')
            #print('Breeding and mutating...')

            # Select the population for breeding
            sorted_rewards = np.argsort(self.rewards[:, g])
            
            #elites are going to be the top x% of the population but UNIQUE ONLY
            elites = math.floor(self.elitism_rate * len(sorted_rewards))
            unique_sorted = []
            unique_cards = set()

            i = len(sorted_rewards) - 1
            while len(unique_sorted) < elites and i >= 0:
                current_cards = tuple(self.consoomers[sorted_rewards[i]].cards)
                if current_cards not in unique_cards:
                    unique_sorted.append(sorted_rewards[i])
                    unique_cards.add(current_cards)
                i -= 1

            unique_sorted = np.array(unique_sorted)

            #Assign unmodified top unique elites
            next_elites = [deepcopy(self.consoomers[i]) for i in unique_sorted[0:elites]]

            # Bottom elite% of the population are copies of the top elite% of the population but shuffled
            next_elites_shuffled = [deepcopy(self.consoomers[i]).shuffle_cards() for i in unique_sorted[0:elites]]

            # SELECTION VIA FITNESS ROULETTE. TOP REWARDS EARNERS HAVE HIGHER PROBABILITY OF BEING SELECTED
            pos_rwds = self.rewards[:, g] - min(self.rewards[:, g])
            tot = sum(pos_rwds[:])
            prob = np.array(pos_rwds/tot)
            breeders = np.random.choice(sorted_rewards, size=math.floor((len(sorted_rewards)-(2*elites))), p=prob)

            best_consoomers = sorted_rewards[-1:-4:-1]
            for i in best_consoomers:
                print(f"Best {i} consoomer: {self.consoomers[i].cards} with distribution {self.consoomers[i].purchases}")
            print('\n')

            bred = []
            for j in range(len(breeders)):
                if(j == len(breeders)-1):
                    je = self.consoomers[breeders[j]].breed(self.consoomers[breeders[0]], np.shape(self.cards_ref)[0], j)
                else:
                    je = self.consoomers[breeders[j]].breed(self.consoomers[breeders[j+1]], np.shape(self.cards_ref)[0], j)
                bred.append(je)
                #print("Bred and got cards " + str(self.consoomers[elites+j].cards) + " and distribution " + str(self.consoomers[elites+j].purchases))

            self.consoomers = next_elites + next_elites_shuffled + bred
            np.savetxt("card_use_mat.csv", card_use_mat, delimiter=",", fmt='%.0f')
    
    def run_sim_for_consoomer(self, c):
        points = np.array([np.diag(self.points_mat[c.purchases, :])]).reshape(1, 11, 1)
        cashback = np.array([np.diag(self.cashback_mat[c.purchases, :])]).reshape(1, 11, 1)
        cashback, other_cb, points, other_p = self.calc_custom_rewards(cashback, points, c)

        points = np.sum(points * self.year, axis=1)
        cashback = np.sum(cashback * self.year, axis=1)

        bonus_p = np.zeros((len(self.consoomers)))
        if self.i == 0:
            #we are adding the bonus offer for the first year
            bonus_p = np.array([c.calc_bonus_offer(self.offer_points_mat)])
            bonus_cb = np.array([c.calc_bonus_offer(self.offer_cash_mat)])
        else:
            bonus_p = np.array([c.calc_bonus_offer(self.anniv_points_mat)])
            bonus_cb = np.array([0])
    
        # Convert points to cash
        bonus_p = bonus_p / 100
        other_p = other_p / 100
        points = points / 100

        annual_fee = np.array([c.calc_annual_fee(self.annual_fee_mat)])
        #print(f"Annual fee: {annual_fee}, points: {points}, cashback: {cashback}, bonus_p: {bonus_p}, bonus_cb: {bonus_cb}, other_cb: {other_cb}, other_p: {other_p}")
        return np.sum(points + cashback, axis=1) + annual_fee + bonus_p + bonus_cb + other_cb + other_p
    
    def calc_custom_rewards(self, cashback, points, c):
        # bank of america customized cash rewards allows for 3% cashback on a category of your choice from
        # gas, online shopping, dining, travel, drug stores, or home improvement/furnishing
        other_cb = 0
        other_p = 0
        if 30 in c.cards:
            bofa_max_cc_bonus_cb = 2500 * 4 #max 3% is on first 2500 per quarter, assumption you can space out purchases
            bofa_cc_categories = np.where(c.purchases == 30, 1, 0) #change this to the index of the bank of america card
            quarter_sum = np.sum(self.year, axis=1)
            travel_sum = np.sum(quarter_sum * np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])*bofa_cc_categories)
            max_cat_not_trav = np.argmax(quarter_sum*np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]) * bofa_cc_categories)
            if travel_sum < quarter_sum[max_cat_not_trav]:
                if quarter_sum[max_cat_not_trav] < bofa_max_cc_bonus_cb:
                    cashback[0, max_cat_not_trav, 0] = 0.03
                else:
                    other_cb += (bofa_max_cc_bonus_cb * 0.03) + ((quarter_sum[max_cat_not_trav] - bofa_max_cc_bonus_cb) * 0.01)
                    cashback[0, max_cat_not_trav, 0] = 0
            else:
                if travel_sum < bofa_max_cc_bonus_cb:
                    for i in [0, 1, 2, 3]:
                        cashback[0, i, 0] = 0.03
                else:
                    other_cb += (bofa_max_cc_bonus_cb * 0.03) + ((travel_sum - bofa_max_cc_bonus_cb) * 0.01)
                    for i in [0, 1, 2, 3]:
                        cashback[0, i, 0] = 0

        # citi custom cash allows for 5% cashback on a category of your choice from restaurants, gas stations, grocery stores, select travel, select transit, select streaming services, drugstores, home improvement stores, fitness clubs, live entertainment
        if 45 in c.cards:
            ccc_max_cc_bonus_cb = 500 * 12
            ccc_cc_categories = np.where(c.purchases == 45, 1, 0) #change this to the index of the citi card
            year_sum = np.sum(self.year, axis=1)
            travel_sum = np.sum(year_sum * np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])*ccc_cc_categories)
            max_cat_not_trav = np.argmax(year_sum * np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])*ccc_cc_categories)
            if travel_sum < year_sum[max_cat_not_trav]:
                if year_sum[max_cat_not_trav] < ccc_max_cc_bonus_cb:
                    cashback[0, max_cat_not_trav, 0] = 0.05
                else:
                    other_cb += (ccc_max_cc_bonus_cb * 0.05) + ((year_sum[max_cat_not_trav] - ccc_max_cc_bonus_cb) * 0.01)
                    cashback[0, max_cat_not_trav, 0] = 0
            else:
                if travel_sum < ccc_max_cc_bonus_cb:
                    for i in [0, 1]:
                        cashback[0, i, 0] = 0.05
                else:
                    other_cb += (ccc_max_cc_bonus_cb * 0.05) + ((travel_sum - ccc_max_cc_bonus_cb) * 0.01)
                    for i in [0, 1, 2, 3]:
                        cashback[0, i, 0] = 0
        
        # robinhood x1 card, 2x on everything up to 1000
        if 50 in c.cards:
            x1_min_cc_bonus_p = 1000
            x1_cc_categories = np.where(c.purchases == 50, 1, 0) #change this to the index of the robinhood card
            for month in range(len(self.year[1])):
                x1_sum = np.sum(self.year[:, month] * x1_cc_categories)
                if x1_sum < 1000:
                    other_p += 2 * x1_sum
                elif x1_sum > 1000 and x1_sum < 7500:
                    other_p += (2 * 1000) + (3 * (x1_sum - 1000))
                else:
                    other_p += (2 * 1000) + (3 * (x1_sum - 1000)) + (2 * (x1_sum - 7500))
            points[0, x1_cc_categories, 0] = 0
        
        if 999 in c.cards:
            x1_min_cc_bonus_p = 1000
            x1_cc_categories = np.where(c.purchases == 51, 1, 0)
            for quarter in range(4):
                five_per_bonus_cat = np.random.choice([0, 0, 0, 0, 4, 5, 6, 7, 8, 0, 10])
                if x1_cc_categories[five_per_bonus_cat] == 1:
                    if c.purchases[five_per_bonus_cat] <= 1500:
                        other_cb += 0.05 * 1500
                    else:
                        other_cb += 0.05 * 1500 + (0.01 * (c.purchases[five_per_bonus_cat] - 1500))

            

        return cashback, other_cb, points, other_p

    def load_cards(self):
        dat = pd.read_csv("C:/Users/EvanChase/Documents/Repos/credit-card-calculator/cards.csv")
        self.cards_ref = np.asarray(dat.iloc[0:53, 2:].to_numpy())
        print(self.cards_ref)
        self.cards_ref = np.nan_to_num(self.cards_ref)
        
    def display_results(self):
        # Create a figure with 2 subplots
        fig, axs = plt.subplots(ncols=3, nrows=2)
        plt.subplots_adjust(hspace=1)

        # Graph 0: Max reward vs Generation
        max_rewards = np.max(self.rewards, axis=0)
        print(max_rewards)
        #smooth results by averaging every of generations
        max_rewards = np.convolve(max_rewards, np.ones((100,))/100, mode='valid')
        axs[0, 0].plot(range(len(max_rewards)), max_rewards, label='Max Reward')
        axs[0, 0].set_xlabel('Generation')
        axs[0, 0].set_ylabel('Max Reward')
        axs[0, 0].set_title('Max Reward vs Generation')
        axs[0, 0].legend()

        # Graph 1: Avg Reward vs Generation
        avg_rewards = np.mean(self.rewards, axis=0)
        #smooth results by averaging every 10 generations
        avg_rewards = np.convolve(avg_rewards, np.ones((100,))/100, mode='valid')
        axs[1, 0].plot(range(len(avg_rewards)), avg_rewards, label='Average Reward')
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
        top_5_indices = np.argsort(self.rewards.sum(axis=1))[-50:]
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