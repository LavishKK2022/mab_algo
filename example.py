from mab_algo import *

########################## Initalise Actions

actions = ["A", "B", "C", "D"]

########################## Initalise Averager

averager = SimpleAverage()

averager = WeightedAverage(step_size = 0.2)

averager = ExponentialMovingAverage(discount = 0.8)

########################## Initialise Algorithm

algo = EpsilonGreedy(
    actions = actions,
    averager = averager, # Can use any averager
    epsilon = 0.5
)

algo = UCB(
    actions = actions,
    averager = averager, # Can use any averager
    exploration = 0.35
)

algo = GradientBandit(
    actions = actions,
    averager = averager, # Can use any averager
    step_size = 0.9
)

########################## Step Through Algorithm
action = algo.step()
reward = 0.5 # for taking above 'action' 
next_action = algo.step(reward)
reward = 5 # for taking above 'next_action'
next_action = algo.step(reward)
print(next_action)
# ... continue stepping through and providing reward values ...
########################## 