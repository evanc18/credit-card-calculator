import numpy as np

RANGES = np.array([
    [0, 1500], # flight
    [0, 400],  # hotel
    [0, 2000], # airbnb
    [0, 850],  # rental
    [200, 400], # grocery
    [150, 350], # dining
    [40, 100], # streaming
    [200, 500], # shopping
    [0, 200] # other
])

AVG_FLIGHT = 250 # average flight price
AVG_AIRBNB = 200 # average airbnb price per night
AVG_HOTEL = 200 # average hotel price per night
AVG_CAR = 80 # average car rental price per day

WORK_TRIPS = [
    [0, 3], # number of trips
    [1, 11] # number of days per trip
]
WORK_TRIPS_MAX = 20 # max number of days for work trips

PERSONAL_TRIPS = [
    [0, 2], # number of trips
    [3, 8], # number of days per trip
]

class Budget:
    def __init__(self, num):
        self.num = num
        self.test_budgets = [] # list of spendings

    def populate(self):
        for i in range(self.num):
            # work trips
            w_trips = np.ones((WORK_TRIPS_MAX,1))
            while np.sum(w_trips) >= WORK_TRIPS_MAX:
                w_trips = np.array([np.random.randint(WORK_TRIPS[1][0], WORK_TRIPS[1][1]) for i in range(np.random.randint(WORK_TRIPS[0][0], WORK_TRIPS[0][1]))])

            # personal trips
            p_trips = np.array([np.random.randint(PERSONAL_TRIPS[1][0], PERSONAL_TRIPS[1][1]) for i in range(np.random.randint(PERSONAL_TRIPS[0][0], PERSONAL_TRIPS[0][1]))])

            b = np.zeros((len(RANGES)))
            b[0] = (len(w_trips) + len(p_trips)) * AVG_FLIGHT
            #if a work trip is less than 3 days, stay in a hotel
            b[1] = np.sum(w_trips[w_trips < 3]) * AVG_HOTEL
            #a personal trip has a 25% chance of staying in a hotel
            b[1] += np.sum(p_trips[np.random.choice([0,1], p=[.75, .25], size=len(p_trips)) == 1])  * AVG_HOTEL
            #if a work trip is 3 days or more, stay in an airbnb with a 50% chance or 0
            b[2] = np.sum(w_trips[w_trips >= 3] * AVG_AIRBNB * np.random.randint(0,2,len(w_trips[w_trips >= 3])))
            #a work trip has a 75% chance of renting a car
            b[3] = np.sum(w_trips * np.random.choice([0, 1], p=[0.25, 0.75], size=len(w_trips)) * AVG_CAR)
            #all other categories generated randomly in ranges
            b[4:] = np.array([np.random.randint(RANGES[i][0], RANGES[i][1]) for i in range(4, len(RANGES))])

            self.test_budgets.append(np.transpose(b))
            #print('\nflight: ${:.2f}, hotel: ${:.2f}, airbnb: ${:.2f}, rental: ${:.2f}, grocery: ${:.2f}, dining: ${:.2f}, streaming: ${:.2f}, shopping: ${:.2f}, other: ${:.2f}'.format(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]))
            #print('You have {} work trips with and {} personal trips. Your total budget is ${:.2f}.\n'.format(len(w_trips), len(p_trips), sum(b)))

        #print(np.hstack((self.test_budgets, np.sum(self.test_budgets, axis=1).reshape(-1,1))))
    
if __name__ == "__main__":
    b = Budget(100)
    b.populate()

    