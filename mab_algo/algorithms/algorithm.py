from abc import ABC, abstractmethod
from typing import List, Union

class Algorithm(ABC):
    def __init__(self, actions: List[str] , averager) -> None:
        """Initialises an algorithm with a list of actions and an averaging scheme.

        Args:
            actions (List[str]): The possible actions to choose from.
            averager: The averaging scheme to apply when updating the rewards.
        """
        self.actions = actions
        self.averager = averager
        self.averager._set_q_values_count(len(actions))
        self._input_validation()
        self.initialised = False
        self.chosen_action = actions[0]
        
    def _input_validation(self):
        """Ensures there is at least one action in the list of actions"""
        assert len(self.actions) > 0, "Undefined actions: need to supply a list of at least 1 action"
        
    def _action_index(self, seek_action: str) -> int:
        """Retrieves the index of an action from the actions list. Throws ValueError if not found.

        Args:
            seek_action (str): The action to be searched.

        Returns:
            int: The index of the specific action.
        """
        return self.actions.index(seek_action)
    
    @abstractmethod
    def step(self, reward) -> str:
        pass
    
    @abstractmethod
    def _cumulative_distribution(self, actions: List[Union[str, int]]) -> Union[int, str]:
        pass
    