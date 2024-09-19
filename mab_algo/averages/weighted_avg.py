from .average import Average
from typing import Union

class WeightedAverage(Average):
    def __init__(self, step_size: Union[int, float]) -> None:
        """Initialises the step_size parameter for a weighted average.

        Args:
            step_size (int | float): The weighting factor (step_size > 0) for new rewards
        """
        self.step_size = step_size
        self._input_validation()
        
    def _input_validation(self):
        """Ensures the step_size is greater than 0"""
        assert self.step_size > 0, "Step size should be greater than 0"
        
    def update(self, index, new_reward) -> None:
        estimated_Qvalue = self.estimated_Qvalues[index]
        new_average = estimated_Qvalue + (self.step_size * (new_reward - estimated_Qvalue))
        super()._store(index, new_average)
    