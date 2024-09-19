import unittest
from mab_algo import *

class TestSimpleAverage(unittest.TestCase):
    
    def test_calculated_average(self):
        index = 0
        actions = 10
        reward_first = 100
        reward_second = 200
        
        self.assertLess(index, actions)
    
        average = SimpleAverage()
        average._set_q_values_count(actions)
        
        average.update(0,reward_first)
        Q_value = average.estimated_Qvalues[index]
        self.assertEqual(Q_value, 100)
        
        average.update(0,reward_second)
        Q_value = average.estimated_Qvalues[index]
        self.assertEqual(Q_value, 150)
        

if __name__ == '__main__':
    unittest.main()