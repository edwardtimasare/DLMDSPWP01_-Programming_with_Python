import unittest
import numpy as np
import pandas as pd


from assignment import (least_squares, find_best_functions)

class TestLeastSquares(unittest.TestCase):
    
    def test_least_squares(self):
        self.assertAlmostEqual(least_squares(np.array([1,2,3]), np.array([1,2,3])), 0.0)
        self.assertAlmostEqual(least_squares(np.array([1,2,3]), np.array([0,0,0])), 14.0)

class TestFindBestFunctions(unittest.TestCase):

    def test_find_best_functions_y1(self):
        training_data = pd.DataFrame({'y1': [0, 1]})
        ideal_functions = pd.DataFrame({'f1': [0, 1], 'f2': [1, 0], 'f3': [0.5, 0.5]})
        self.assertEqual(find_best_functions(training_data['y1'], ideal_functions), ['f1', 'f3', 'f2'])
        
    def test_find_best_functions_y2(self):
        training_data = pd.DataFrame({'y2': [1, 0]})
        ideal_functions = pd.DataFrame({'f1': [0, 1], 'f2': [1, 0], 'f3': [0.5, 0.5]})
        self.assertEqual(find_best_functions(training_data['y2'], ideal_functions), ['f2', 'f3', 'f1'])
    
    def test_find_best_functions_y3(self):
        training_data = pd.DataFrame({'y3': [1, 0]})
        ideal_functions = pd.DataFrame({'f1': [0, 1], 'f2': [1, 0], 'f3': [0.5, 0.5]})
        self.assertEqual(find_best_functions(training_data['y3'], ideal_functions), ['f2', 'f3', 'f1'])

    def test_find_best_functions_y4(self):
        training_data = pd.DataFrame({'y4': [1, 0]})
        ideal_functions = pd.DataFrame({'f1': [0, 1], 'f2': [1, 0], 'f3': [0.5, 0.5]})
        self.assertEqual(find_best_functions(training_data['y4'], ideal_functions), ['f2', 'f3', 'f1'])

if __name__ == '__main__':
    unittest.main()
