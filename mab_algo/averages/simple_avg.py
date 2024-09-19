from .average import Average
from typing import Union

class SimpleAverage(Average):
    def __init__(self) -> None:
        self.nth_term = 0
        
    def update(self, index: int, new_reward: Union[float, int]) -> None:
        self.nth_term += 1
        estimated_Qvalue = self.estimated_Qvalues[index]
        new_average = estimated_Qvalue + ((1/self.nth_term) * (new_reward - estimated_Qvalue))
        super()._store(index, new_average)
        