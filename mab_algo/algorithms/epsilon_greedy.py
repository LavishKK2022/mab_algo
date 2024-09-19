from .algorithm import Algorithm
from random import randrange, random
from typing import List, Union

class EpsilonGreedy(Algorithm):

    def __init__(self, actions: List[str], averager, epsilon: float) -> None:
        """Initialise actions, averaging policy and epsilon value. 
        A larger epsilon value, explores more.
        An epsilon value of 0 is the same as a greedy algorithm.

        Args:
            actions (List[str]): The possible actions to choose from.
            averager: The averaging scheme to apply when updating the rewards.
            epsilon (float): Epsilon value to describe balance between exploration and exploitation. 0 <= epsilon <= 1
        """
        self.epsilon = epsilon
        super().__init__(actions, averager)
        
    def _input_validation(self):
        """Ensures 0 <= epsilon <= 1"""
        super()._input_validation()
        assert self.epsilon >= 0, "Epsilon value must be greater than or equal to 0"
        assert self.epsilon <= 1, "Epsilon value must be less than or equal to 1"
        
    def step(self, reward: Union[int, float ] = 0) -> str:
        """Step through epsilon greedy algorithm, returning next action to take at each step.

        Args:
            reward (int |float, optional): This is the reward received for taking the previous action. Can be ommitted if on first step - defaults to 0.

        Returns:
            str: The action to take next.
        """
        if not self.initialised:
            self.chosen_action = self._cumulative_distribution(self.actions)
            self.initialised = True
            return self.chosen_action
    
        # Update the reward
        self.averager.update(self._action_index(self.chosen_action), reward)
        
        if random() <= self.epsilon:
            # perform exploration
            return self._cumulative_distribution(self.actions)
        else:
            # perform exploitation
            argmax = float('-inf')
            argmax_indexes = []
            
            for i in range(len(self.actions)):
                # select action with largest estimated Q value
                estimated_Qvalue = self.averager.estimated_Qvalues[i]
                if estimated_Qvalue > argmax:
                    argmax = estimated_Qvalue
                    argmax_indexes = [i]
                elif estimated_Qvalue == argmax:
                    argmax_indexes.append(i)
            
            if len(argmax_indexes) > 1:
                # Break ties
                return self.actions[self._cumulative_distribution(argmax_indexes)]
            else:
                return self.actions[argmax_indexes[0]]

    
    def _cumulative_distribution(self, actions: List[Union[str, int]]) -> Union[int, str]:
        """Describes the policy to break ties or select the action when algorithm is first stepped through (after initialisation).

        Args:
            actions(List[str|int]): The actions or action indexes to select from.

        Returns:
            str|int: Returns the selected index or action
        """
        return actions[randrange(len(actions))]