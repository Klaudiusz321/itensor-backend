import unittest
import numpy as np
from myproject.utils.numerical.core import NumericTensorCalculator

class TestNumericTensorCalculator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Test metric function for flat space (identity matrix)
        self.flat_metric = lambda x: np.eye(len(x))
        
        # Test metric function for 2D polar coordinates
        def polar_metric(x):
            r, theta = x
            g = np.zeros((2, 2))
            g[0, 0] = 1.0
            g[1, 1] = r**2
            return g
        self.polar_metric = polar_metric
        
        # Initialize calculator with different metrics
        self.flat_calculator = NumericTensorCalculator(self.flat_metric, h=1e-8)  # Smaller h for better precision
        self.polar_calculator = NumericTensorCalculator(self.polar_metric, h=1e-8)
        
        # Test points
        self.flat_point = np.array([0.0, 0.0])
        self.polar_point = np.array([1.0, 0.0])  # r=1, theta=0

    def test_initialization(self):
        """Test if the calculator initializes correctly."""
        self.assertIsNotNone(self.flat_calculator)
        self.assertIsNotNone(self.polar_calculator)
        self.assertEqual(self.flat_calculator.h, 1e-8)
        self.assertEqual(self.polar_calculator.h, 1e-8)

    def test_compute_christoffel_flat_space(self):
        """Test Christoffel symbols computation in flat space."""
        Gamma = self.flat_calculator.compute_christoffel(self.flat_point)
        
        # In flat space, all Christoffel symbols should be zero
        self.assertTrue(np.allclose(Gamma, np.zeros_like(Gamma), atol=1e-10))
        
        # Check shape
        self.assertEqual(Gamma.shape, (2, 2, 2))

    def test_compute_christoffel_polar(self):
        """Test Christoffel symbols computation in polar coordinates."""
        Gamma = self.polar_calculator.compute_christoffel(self.polar_point)
        
        # Check shape
        self.assertEqual(Gamma.shape, (2, 2, 2))
        
        # In polar coordinates, some Christoffel symbols should be non-zero
        self.assertFalse(np.allclose(Gamma, np.zeros_like(Gamma), atol=1e-10))
        
        # Check specific values (r=1, theta=0)
        expected_Gamma = np.zeros((2, 2, 2))
        expected_Gamma[1, 1, 0] = 1.0  # Γ^θ_θr = 1/r
        expected_Gamma[1, 0, 1] = 1.0  # Γ^θ_rθ = 1/r
        expected_Gamma[0, 1, 1] = -1.0  # Γ^r_θθ = -r
        
        # Print detailed debugging information
        print("\nDetailed Christoffel symbol comparison:")
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    print(f"Γ^{a}_{b}{c}:")
                    print(f"  Expected: {expected_Gamma[a,b,c]:.6f}")
                    print(f"  Actual:   {Gamma[a,b,c]:.6f}")
                    print(f"  Diff:     {abs(Gamma[a,b,c] - expected_Gamma[a,b,c]):.6f}")
        
        # Use a more lenient tolerance for numerical computations
        self.assertTrue(np.allclose(Gamma, expected_Gamma, atol=1e-3))

    def test_compute_riemann_flat_space(self):
        """Test Riemann tensor computation in flat space."""
        Gamma = self.flat_calculator.compute_christoffel(self.flat_point)
        R = self.flat_calculator.compute_riemann(self.flat_point, Gamma)
        
        # In flat space, all Riemann tensor components should be zero
        self.assertTrue(np.allclose(R, np.zeros_like(R), atol=1e-10))
        
        # Check shape
        self.assertEqual(R.shape, (2, 2, 2, 2))

    def test_compute_riemann_polar(self):
        """Test Riemann tensor computation in polar coordinates."""
        Gamma = self.polar_calculator.compute_christoffel(self.polar_point)
        R = self.polar_calculator.compute_riemann(self.polar_point, Gamma)
        
        # Check shape
        self.assertEqual(R.shape, (2, 2, 2, 2))
        
        # Print actual values for debugging
        print("\nActual Riemann tensor:")
        print(R)
        
        # In polar coordinates, Riemann tensor should be zero (2D space is always flat)
        # Use a more lenient tolerance for numerical computations
        self.assertTrue(np.allclose(R, np.zeros_like(R), atol=1e-3))

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with zero point
        zero_point = np.zeros(2)
        Gamma = self.flat_calculator.compute_christoffel(zero_point)
        self.assertEqual(Gamma.shape, (2, 2, 2))
        
        # Test with different step size
        calculator = NumericTensorCalculator(self.flat_metric, h=1e-4)
        Gamma = calculator.compute_christoffel(self.flat_point)
        self.assertEqual(Gamma.shape, (2, 2, 2))

if __name__ == '__main__':
    unittest.main() 