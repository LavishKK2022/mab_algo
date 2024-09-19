import unittest
from mab_algo import *

class TestAlgorithm(unittest.TestCase):
    
    def test_actions_given_less_than_zero(self):
        actions = []
        
        with self.assertRaises(AssertionError):
            EpsilonGreedy(actions, SimpleAverage(), 0.5)
        
    
    def test_actions_given_greater_than_zero(self):
        actions = ["Test"] * 10
        
        try:
            EpsilonGreedy(actions, SimpleAverage(), 0.5)
        except AssertionError as AE:
            self.fail("Assertion Error raised for action count greater than 0")

if __name__ == '__main__':
    unittest.main()