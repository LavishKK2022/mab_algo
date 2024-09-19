from .average import Average
from typing import Union

class ExponentialMovingAverage(Average):
    def __init__(self, discount: float) -> None:
        """Initialise Exponential Moving Average with discount variable

        Args:
            discount (float): Value 0 <= discount <= 1. A higher discount reduces the weight of future rewards
        """
        self.discount = discount
        self._input_validation()
        
    def _input_validation(self):
        """Ensures 0 <= discount <= 1."""
        assert self.discount <= 1, "Discount factor must be less than or equal to 1"
        assert self.discount >= 0, "Discount factor must be greater than or equal to 0"
        
    def update(self, index: int, new_reward: Union[int, float]) -> None:
        estimated_Qvalue = self.estimated_Qvalues[index]
        new_average = (estimated_Qvalue * self.discount) + ((1 - self.discount) * new_reward)
        super()._store(index, new_average)
    