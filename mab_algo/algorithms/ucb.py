from .algorithm import Algorithm
from math import sqrt, log
from typing import List, Union

class UCB(Algorithm):
    def __init__(self, actions: List[str], averager, exploration: Union[int, float]) -> None:
        """Initialise actions, averaging policy and exploration parameter. 
        A higher exploration parameter, exploits less and explores more.

        Args:
            actions (List[str]): The possible actions to choose from.
            averager: The averaging scheme to apply when updating the rewards.
            exploration (Union[int, float]): Exploration value to describe balance between exploration and exploitation. exploration > 0.
        """
        self.exploration = exploration
        self.time_step = 1
        self.action_counts = {}
        for action in actions:
            self.action_counts[action] = 0
        super().__init__(actions, averager)
        self.averager.estimated_Qvalues = [0] * len(actions)
            
    def _input_validation(self):
        """Ensures exploration parameter > 0"""
        super()._input_validation()
        assert self.exploration > 0, "Exploration value must be greater than 0"
    
    def step(self, reward: Union[int, float ] = 0) -> str:
        """Step through Upper Confidence Bound algorithm, returning next action to take at each step.

        Args:
            reward (int |float, optional): This is the reward received for taking the previous action. Can be ommitted if on first step - defaults to 0.

        Returns:
            str: The action to take next.
        """
        if not self.initialised:
            self.chosen_action = self._cumulative_distribution(self.actions)
            self.action_counts[self.chosen_action] += 1
            self.initialised = True
            return self.chosen_action
        
        # Update the reward
        self.averager.update(self._action_index(self.chosen_action), reward)
        self.time_step += 1
        ucb_values = [0] * len(self.actions)
    
        for i in range(len(self.actions)):
            estimatedQvalue = self.averager.estimated_Qvalues[i]
            if estimatedQvalue == float('inf'):
                # If action not explored yet
                self.chosen_action = self.actions[i]
                self.action_counts[self.chosen_action] += 1
                return self.chosen_action
            
            if self.action_counts[self.actions[i]] > 0:
                # Perform UCB calculation
                ucb_values[i] = estimatedQvalue + (self.exploration * sqrt(log(self.time_step)/self.action_counts[self.actions[i]]))
            else:
                ucb_values[i] = float('inf')
        
        self.chosen_action = self.actions[ucb_values.index(max(ucb_values))]
        self.action_counts[self.chosen_action] += 1
        return self.chosen_action
            
    def _cumulative_distribution(self, actions: List[Union[str, int]]) -> Union[int, str]:
        """Describes the policy to break ties or select the action when algorithm is first stepped through (after initialisation).

        Args:
            actions: The actions or action indexes to select from.

        Returns:
            str|int: Returns the selected index or action
        """
        return actions[0]
    