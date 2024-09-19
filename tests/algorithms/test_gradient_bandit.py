import unittest
from mab_algo import *

class TestGradientBandit(unittest.TestCase):
    
    ACTIONS = ["A", "B", "C"]
    AVERAGE = SimpleAverage()
    
    def test_step_size_less_than_zero(self):
        step_size = -1.5
        self.assertLess(step_size, 0)
        
        with self.assertRaises(AssertionError):
            GradientBandit(TestGradientBandit.ACTIONS, TestGradientBandit.AVERAGE, step_size)
    
        
    def test_step_size_equal_to_zero(self):
        step_size = 0
        self.assertEqual(step_size, 0)
        
        with self.assertRaises(AssertionError):
            GradientBandit(TestGradientBandit.ACTIONS, TestGradientBandit.AVERAGE, step_size)
            
    def test_step_size_greater_than_zero(self):
        step_size = 0.5
        self.assertGreater(step_size, 0)
        
        try:
            GradientBandit(TestGradientBandit.ACTIONS, TestGradientBandit.AVERAGE, step_size)
        except AssertionError as AE:
            self.fail("Assertion Error raised for step_size greater than zero")  
            
    def test_step(self):
        step_size = 1
        self.assertGreater(step_size, 0)
        
        algo = GradientBandit(TestGradientBandit.ACTIONS, TestGradientBandit.AVERAGE, step_size)
        action = algo.step()
        self.assertIn(action, TestGradientBandit.ACTIONS)
        action = algo.step(reward = 10)
        self.assertIn(action, TestGradientBandit.ACTIONS)
        

if __name__ == '__main__':
    unittest.main()