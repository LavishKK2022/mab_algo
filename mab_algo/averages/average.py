from abc import ABC, abstractmethod
from typing import Union

class Average(ABC):
    @abstractmethod
    def update(self, index: int, new_reward: Union[int, float]) -> None:
        """Updates the Q values for the specified action.

        Args:
            index (int): Refers to the position of the action in the initial list.
            new_reward (float | int): The reward for taking the action.
        """
        pass
    
    def _set_q_values_count(self, num_actions: int) -> None:
        """ Set the number of actions.

        Args:
            num_actions (int): The number of actions or "levers".
        """
        self.estimated_Qvalues = [0] * num_actions
    
    def _store(self, index: int, new_average: Union[int,float]) -> None:
        """Store the new_average Q value estimate in the list index position matching the index of the action.

        Args:
            index (int): The index to store the new average. This should match the index of the action in the original list.
            new_average (float | int ): The new calculated average to be stored.
        """
        self.estimated_Qvalues[index] = new_average
        