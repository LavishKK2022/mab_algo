import unittest
from mab_algo import *

class TestExponentialMovingAverage(unittest.TestCase):
    
    def test_discount_greater_than_one(self):
        discount = 1.1
        self.assertGreater(discount, 1)
        
        with self.assertRaises(AssertionError):
            ExponentialMovingAverage(discount)
    
    def test_discount_less_than_zero(self):
        discount = -0.1
        self.assertLess(discount, 0)
        
        with self.assertRaises(AssertionError):
            ExponentialMovingAverage(discount)
    
    def test_discount_equal_to_one(self):
        discount = 1
        self.assertEqual(discount, 1)
        
        try:
            ExponentialMovingAverage(discount)
        except AssertionError as AE:
            self.fail("Assertion Error raised for discount value of 1.")
    
    def test_discount_equal_to_zero(self):
        discount = 0
        self.assertEqual(discount, 0)
  
        try:
            ExponentialMovingAverage(discount)
        except AssertionError as AE:
            self.fail("Assertion Error raised for discount value of 0.")
    
    def test_discount_within_range(self):
        discount = 0.5
        self.assertGreaterEqual(discount, 0)
        self.assertLessEqual(discount, 1)
        
        try:
            ExponentialMovingAverage(discount)
        except AssertionError as AE:
            self.fail("Assertion Error raised for discount value within range.")
    
    def test_calculated_average(self):
        discount = 0.8
        index = 0
        actions = 10
        reward = 100
        
        self.assertGreaterEqual(discount, 0)
        self.assertLessEqual(discount, 1)
        self.assertLess(index, actions)
    
        average = ExponentialMovingAverage(discount)
        average._set_q_values_count(actions)
        
        average.update(0,reward)
        Q_value = average.estimated_Qvalues[index]
        self.assertAlmostEqual(Q_value, 20)
        
        average.update(0,reward)
        Q_value = average.estimated_Qvalues[index]
        self.assertAlmostEqual(Q_value, 36)
        

if __name__ == '__main__':
    unittest.main()