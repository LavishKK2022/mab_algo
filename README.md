# Multi-Armed Bandits
Implementation of Multi Armed Bandit algorithms from "Reinforcement Learning - An Introduction" by Richard S. Sutton and Andrew G. Barto

This library suggests the next actions based on the rewards received for each action.


## Algorithms

This library covers the following algorithms:
- Epsilon Greedy
- Upper Confidence Bound
- Gradient Bandit

### Epsilon Greedy

This algorithm selects an action based on an `epsilon` variable.

| Probability | Action | 
| ----------- | -------|
| 1 - epsilon | action with highest estimated reward |
| epsilon | random action|

> *Note* 0 <=`epsilon`<= 1

### Upper Confidence Bound 

This algorithm selects the next action based on the confidence in the estimated values. 

This uses an `exploration` parameter to control the balance between exploration and exploitation.

A larger `exploration` value means the algorithm explores more.

> *Note* `exploration` > 0

### Gradient Bandit Algorithm

This algorithm selects an action based on a preference.

Internally it uses a soft-max distribution to convert preferences to probabilities.

It has a `step_size` parameter, which influences how much preferences are changed at each step.
A larger `step_size` will learn faster from the most recent reward.

> *Note* `step_size` > 0

## Averaging Functions

The algorithms could be used with any averaging function:

- Simple Average (a.k.a running average)
- Weighted Average
- Exponential Moving Average

### Simple Average

A simple average is the same as a running average.
As time steps are advanced, the average is adjusted to account for the reward.

A higher reward, increases the average and a lower reward decreases the average.

### Weighted Average

A weighted average operates much like a running average but it applies a `step_size` parameter.
The `step_size` parameter controls how much weighting is given to new rewards.

A higher step_size gives more weighting to the most recent reward.

> *Note* `step_size` > 0

### Exponential Moving Average

An exponential moving average applies a `discount` multiplier to previous rewards and (1 - `discount`) multiplier to recent rewards.

A higher `discount` means more weight is given to previous average, making recent rewards less influential.

A lower `discount` means more weight is given to the most recent reward, making older rewards less influential.

## Usage

To use this algorithm first import this library:
```python
from mab_algo import *
```

Declare a set of actions:

```python
actions = ["A", "B", "C"]
```

Select any averaging function:

```python
averager = SimpleAverage()
averager = WeightedAverage(step_size = 0.2)
averager = ExponentialMovingAverage(discount = 0.8)
```
> *Note* This uses example `step_size` and `discount` values, please refer to the sections above to learn more about them.

Select any algorithm:

```python
algo = EpsilonGreedy(
    actions = actions,
    averager = averager,
    epsilon = 0.5
)

algo = UCB(
    actions = actions,
    averager = averager,
    exploration = 0.35
)

algo = GradientBandit(
    actions = actions,
    averager = averager, 
    step_size = 0.9
)
```

> *Note* This uses example `epsilon`, `exploration` and `step_size` values, please refer to the sections above to learn more about them.

Step through the algorithms to get the next actions:

```python
action = algo.step() 
action = algo.step(reward = 0.5)
action = algo.step(reward = 5)
# ... continue as needed
```



#### Verbose
```python
action = algo.step() # get initial action
reward = 0.5 # for taking 'action' - reward must be supplied by you 
next_action = algo.step(reward)
reward = 5 # for taking 'next_action' - reward must be supplied by you
next_action = algo.step(reward)
# ... continue stepping through and providing reward values to get new actions...
```

`example.py` file contains an example of the above in action!

## Testing

Unit tests are provided in the 'tests' directory.

Tests can be executed using this command:

```bash
python3 -m unittest discover tests
```

#### Check Out My Other Projects:
[Lavish Kamal Kumar Portfolio Website](https://lavish-kumar.com)