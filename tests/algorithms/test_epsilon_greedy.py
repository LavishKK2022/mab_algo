import unittest
from mab_algo import *

class TestEpsilonGreedy(unittest.TestCase):
    
    ACTIONS = ["A", "B", "C"]
    AVERAGE = SimpleAverage()
    
    def test_epsilon_greater_than_one(self):
        epsilon = 1.5
        self.assertGreater(epsilon, 1)
        
        with self.assertRaises(AssertionError):
            EpsilonGreedy(TestEpsilonGreedy.ACTIONS, TestEpsilonGreedy.AVERAGE, epsilon)
    
    def test_epsilon_less_than_zero(self):
        epsilon = -1.5
        self.assertLess(epsilon, 0)
        
        with self.assertRaises(AssertionError):
            EpsilonGreedy(TestEpsilonGreedy.ACTIONS, TestEpsilonGreedy.AVERAGE, epsilon)
    
    def test_epsilon_equal_to_one(self):
        epsilon = 1
        self.assertEqual(epsilon, 1)
        
        try:
            EpsilonGreedy(TestEpsilonGreedy.ACTIONS, TestEpsilonGreedy.AVERAGE, epsilon)
        except AssertionError as AE:
            self.fail("Assertion Error raised for epsilon equal to 1")
        
    def test_epsilon_equal_to_zero(self):
        epsilon = 0
        self.assertEqual(epsilon, 0)
        
        try:
            EpsilonGreedy(TestEpsilonGreedy.ACTIONS, TestEpsilonGreedy.AVERAGE, epsilon)
        except AssertionError as AE:
            self.fail("Assertion Error raised for epsilon equal to 0")   
            
    def test_epsilon_within_bounds(self):
        epsilon = 0.5
        self.assertGreaterEqual(epsilon, 0)
        self.assertLessEqual(epsilon, 1)
        
        try:
            EpsilonGreedy(TestEpsilonGreedy.ACTIONS, TestEpsilonGreedy.AVERAGE, epsilon)
        except AssertionError as AE:
            self.fail("Assertion Error raised for epsilon within bounds")  
            
    def test_exploration(self):
        epsilon = 1
        self.assertGreaterEqual(epsilon, 0)
        self.assertLessEqual(epsilon, 1)
        
        algo = EpsilonGreedy(TestEpsilonGreedy.ACTIONS, TestEpsilonGreedy.AVERAGE, epsilon)
        action = algo.step()
        self.assertIn(action, TestEpsilonGreedy.ACTIONS)
        action = algo.step(reward = 10)
        self.assertIn(action, TestEpsilonGreedy.ACTIONS)
        
        
    def test_exploitation(self):
        epsilon = 0
        self.assertGreaterEqual(epsilon, 0)
        self.assertLessEqual(epsilon, 1)
        
        algo = EpsilonGreedy(TestEpsilonGreedy.ACTIONS, TestEpsilonGreedy.AVERAGE, epsilon)
        action = algo.step()
        self.assertIn(action, TestEpsilonGreedy.ACTIONS)
        action = algo.step(reward = 10)
        self.assertIn(action, TestEpsilonGreedy.ACTIONS)

if __name__ == '__main__':
    unittest.main()