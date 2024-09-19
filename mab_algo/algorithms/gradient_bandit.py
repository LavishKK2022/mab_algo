from .algorithm import Algorithm
from random import random
from math import e, pow
from typing import List, Union

class GradientBandit(Algorithm):
    def __init__(self, actions: List[str], averager, step_size: Union[int, float]) -> None:
        """Initialise actions, averaging policy and epsilon value. 
        A larger step size will learn faster from recent rewards.

        Args:
            actions (List[str]): The possible actions to choose from.
            averager: The averaging scheme to apply when updating the rewards.
            step_size ([int |float]): The rate at which the algorithm learns from recent rewards. step_size > 0.
        """
        self.step_size = step_size
        self.chosen_action = actions[0]
        super().__init__(actions, averager)
        self.probabilities = [(1/len(self.actions))] * len(self.actions)
        self.preferences = [0] * len(self.actions)

    def _input_validation(self):
        """Ensures the step_size > 0"""
        super()._input_validation()
        assert self.step_size > 0, "Step size value must be greater than 0"
    
    def step(self, reward: Union[int, float ] = 0) -> str:
        """Step through gradient bandit algorithm, returning next action to take at each step.

        Args:
            reward (int | float, optional): This is the reward received for taking the previous action. Can be ommitted if on first step - defaults to 0.

        Returns:
            str: The action to take next.
        """
        if not self.initialised:
            self.chosen_action = self._cumulative_distribution(self.actions, self.probabilities)
            self.initialised = True
            return self.chosen_action
        
        # Update the reward
        self.averager.update(self._action_index(self.chosen_action), reward)
        self._update_preferences(reward)
        
        # Apply the softmax distribution converting preferences to probabilities
        self._softmax_distribution()
        self.chosen_action = self._cumulative_distribution(self.actions, self.probabilities)
        return self.chosen_action
        
    def _update_preferences(self, reward: Union[int,float]):
        """Update the preferences based on received rewards.

        Args:
            reward (int | float): The reward used to update the preferences.
        """
        chosen_index = self.actions.index(self.chosen_action)
        for i in range(len(self.preferences)):
            if i == chosen_index:
                self.preferences[i] + (self.step_size * (1 - self.probabilities[i]) * (reward - self.averager.estimated_Qvalues[i]))
            else:
                self.preferences[i] - (self.step_size * self.probabilities[i] * (reward - self.averager.estimated_Qvalues[i]))

    def _softmax_distribution(self):
        """Converts preferences into probabilities"""
        denom = 0
        
        for i in range(len(self.actions)):
            denom += (e ** self.preferences[i])
        
        for i in range(len(self.actions)):
            self.probabilities[i] = (e ** self.preferences[i]) / denom
    
    def _cumulative_distribution(self, actions: List[Union[str, int]], probabilities: List[float]) -> Union[int, str]:
        """Describes the policy to select an action, given a set of probabilities. 

        Args:
            actions (List[str, int]): List of actions to choose from.
            probabilities (List[float]): List of probabilities with the positions corresponding to the position of the actions.

        Returns:
            int|str: Returns the selected index or action
        """   
        rand = random()
        bound = 0
        
        for i in range(len(actions)):
            bound += probabilities[i]
            
            if rand < bound:
                return actions[i]
                 