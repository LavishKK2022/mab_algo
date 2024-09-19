import unittest
from mab_algo import *

class TestWeightedAverage(unittest.TestCase):
    
    def test_step_size_greater_than_zero(self):
        step_size = 0.5
        self.assertGreater(step_size, 0)
        
        try:
            WeightedAverage(step_size)
        except AssertionError as AE:
            self.fail("Assertion Error raised for step_size greater than 0")
        
    
    def test_step_size_less_than_zero(self):
        step_size = -0.1
        self.assertLess(step_size, 0)
        
        with self.assertRaises(AssertionError):
            WeightedAverage(step_size)
    
    
    def test_calculated_average(self):
        index = 0
        actions = 10
        reward_first = 100
        reward_second = 200
        step_size = 0.5
        
        self.assertGreater(step_size, 0)
        self.assertLess(index, actions)
        
    
        average = WeightedAverage(step_size)
        average._set_q_values_count(actions)
        
        average.update(0,reward_first)
        Q_value = average.estimated_Qvalues[index]
        self.assertEqual(Q_value, 50)
        
        average.update(0,reward_second)
        Q_value = average.estimated_Qvalues[index]
        self.assertEqual(Q_value, 125)
        

if __name__ == '__main__':
    unittest.main()