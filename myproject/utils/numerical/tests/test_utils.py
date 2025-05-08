import unittest
import numpy as np
from myproject.utils.numerical.utils.index_utils_num import IndexUtilsNum
from myproject.utils.numerical.utils.derivative_utils_num import DerivativeUtilsNum

class TestIndexUtilsNum(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.n = 4  # 4-dimensional space
        self.index_utils = IndexUtilsNum(self.n)

    def test_initialization(self):
        """Test if the IndexUtilsNum class initializes correctly."""
        self.assertEqual(self.index_utils.n, self.n)
        self.assertIsInstance(self.index_utils, IndexUtilsNum)

    def test_christoffel_indices(self):
        """Test Christoffel symbol index generation."""
        indices = self.index_utils.generate_index_christoffel()
        
        # Check if indices are generated
        self.assertIsNotNone(indices)
        self.assertTrue(len(indices) > 0)
        
        # Check format of indices
        for idx in indices:
            self.assertEqual(len(idx), 3)  # Christoffel symbols have 3 indices
            self.assertTrue(all(0 <= i < self.n for i in idx))  # Indices within range

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with n=1
        index_utils_1d = IndexUtilsNum(1)
        indices = index_utils_1d.generate_index_christoffel()
        self.assertEqual(len(indices), 1)
        
        # Test with n=2
        index_utils_2d = IndexUtilsNum(2)
        indices = index_utils_2d.generate_index_christoffel()
        self.assertTrue(len(indices) > 0)

class TestDerivativeUtilsNum(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Test function: f(x) = x^2
        self.f = lambda x: x[0]**2 + x[1]**2
        
        # Test metric function: g_ij = δ_ij
        self.g_func = lambda x: np.eye(len(x))
        
        # Test point
        self.x = np.array([1.0, 2.0])
        
        # Initialize derivative utils
        self.du_scalar = DerivativeUtilsNum(
            f=self.f,
            x=self.x,
            mu=0,  # derivative with respect to first coordinate
            h=1e-6
        )
        
        self.du_metric = DerivativeUtilsNum(
            g_func=self.g_func,
            x=self.x,
            mu=0,
            i=0,
            j=0,
            h=1e-6
        )

    def test_initialization(self):
        """Test if the DerivativeUtilsNum class initializes correctly."""
        self.assertIsNotNone(self.du_scalar)
        self.assertIsNotNone(self.du_metric)
        self.assertEqual(self.du_scalar.h, 1e-6)
        self.assertEqual(self.du_metric.h, 1e-6)

    def test_numerical_partial_scalar(self):
        """Test numerical partial derivative of scalar function."""
        # For f(x,y) = x^2 + y^2, ∂f/∂x = 2x
        derivative = self.du_scalar.numerical_partial_scalar()
        expected = 2.0  # 2 * x[0] = 2 * 1.0 = 2.0
        self.assertAlmostEqual(derivative, expected, places=5)

    def test_numerical_partial_g(self):
        """Test numerical partial derivative of metric tensor."""
        # For g_ij = δ_ij, all partial derivatives should be zero
        derivative = self.du_metric.numerical_partial_g()
        self.assertAlmostEqual(derivative, 0.0, places=5)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with zero point
        du_zero = DerivativeUtilsNum(
            f=self.f,
            x=np.zeros(2),
            mu=0,
            h=1e-6
        )
        derivative = du_zero.numerical_partial_scalar()
        self.assertAlmostEqual(derivative, 0.0, places=5)
        
        # Test with different step size
        du_h = DerivativeUtilsNum(
            f=self.f,
            x=self.x,
            mu=0,
            h=1e-4
        )
        derivative = du_h.numerical_partial_scalar()
        self.assertAlmostEqual(derivative, 2.0, places=3)  # Less precise with larger h

if __name__ == '__main__':
    unittest.main() 