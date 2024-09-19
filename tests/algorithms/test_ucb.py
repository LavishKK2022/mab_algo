import unittest
from mab_algo import *

class TestUCB(unittest.TestCase):
    
    ACTIONS = ["A", "B", "C"]
    AVERAGE = SimpleAverage()
    
    def test_exploration_less_than_zero(self):
        exploration = -1.5
        self.assertLess(exploration, 0)
        
        with self.assertRaises(AssertionError):
            GradientBandit(TestUCB.ACTIONS, TestUCB.AVERAGE, exploration)
    
        
    def test_exploration_equal_to_zero(self):
        exploration = 0
        self.assertEqual(exploration, 0)
        
        with self.assertRaises(AssertionError):
            GradientBandit(TestUCB.ACTIONS, TestUCB.AVERAGE, exploration)
            
    def test_exploration_greater_than_zero(self):
        exploration = 0.5
        self.assertGreater(exploration, 0)
        
        try:
            GradientBandit(TestUCB.ACTIONS, TestUCB.AVERAGE, exploration)
        except AssertionError as AE:
            self.fail("Assertion Error raised for exploration greater than zero")  
            
    def test_step(self):
        exploration = 2
        self.assertGreater(exploration, 0)
        
        algo = GradientBandit(TestUCB.ACTIONS, TestUCB.AVERAGE, exploration)
        action = algo.step()
        self.assertIn(action, TestUCB.ACTIONS)
        action = algo.step(reward = 10)
        self.assertIn(action, TestUCB.ACTIONS)
        

if __name__ == '__main__':
    unittest.main()