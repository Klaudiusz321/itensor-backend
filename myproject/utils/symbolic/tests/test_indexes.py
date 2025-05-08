import unittest
import sympy as sp
import numpy as np
from myproject.utils.symbolic.utilis.indexes import Indexes

class TestIndexes(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.n = 4  # 4-dimensional space
        self.indexes = Indexes(self.n)
        
        # Create a simple metric tensor for testing
        self.g = sp.eye(self.n)  # Identity matrix as metric
        self.g_inv = sp.eye(self.n)  # Inverse metric (same as metric for identity)
        
        # Create a test tensor
        self.test_tensor = [[[[1 for _ in range(self.n)] for _ in range(self.n)] 
                            for _ in range(self.n)] for _ in range(self.n)]

    def test_initialization(self):
        """Test if the Indexes class initializes correctly."""
        self.assertEqual(self.indexes.n, self.n)
        self.assertIsInstance(self.indexes, Indexes)

    def test_generate_index_christoffel(self):
        """Test Christoffel symbol index generation."""
        indices = self.indexes.generate_index_christoffel()
        
        # Check if indices are generated
        self.assertIsNotNone(indices)
        self.assertTrue(len(indices) > 0)
        
        # Check format of indices
        for idx in indices:
            self.assertEqual(len(idx), 3)  # Christoffel symbols have 3 indices
            self.assertTrue(all(0 <= i < self.n for i in idx))  # Indices within range

    def test_generate_index_riemann(self):
        """Test Riemann tensor index generation."""
        indices = self.indexes.generate_index_riemann()
        
        # Check if indices are generated
        self.assertIsNotNone(indices)
        self.assertTrue(len(indices) > 0)
        
        # Check format of indices
        for idx in indices:
            self.assertEqual(len(idx), 4)  # Riemann tensor has 4 indices
            self.assertTrue(all(0 <= i < self.n for i in idx))  # Indices within range
            
        # Check symmetry properties
        for a, b, c, d in indices:
            self.assertTrue(a <= b)  # First pair of indices is ordered
            self.assertTrue(c <= d)  # Second pair of indices is ordered
            self.assertTrue((a * self.n + b) <= (c * self.n + d))  # Pairs are ordered

    def test_generate_index_ricci(self):
        """Test Ricci tensor index generation."""
        indices = self.indexes.generate_index_ricci()
        
        # Check if indices are generated
        self.assertIsNotNone(indices)
        self.assertTrue(len(indices) > 0)
        
        # Check format of indices
        for idx in indices:
            self.assertEqual(len(idx), 2)  # Ricci tensor has 2 indices
            self.assertTrue(all(0 <= i < self.n for i in idx))  # Indices within range

    def test_lower_indices(self):
        """Test lowering tensor indices."""
        lowered = self.indexes.lower_indices(self.test_tensor, self.g)
        
        # Check if tensor is lowered
        self.assertIsNotNone(lowered)
        self.assertEqual(len(lowered), self.n)
        
        # Check structure of lowered tensor
        for i in range(self.n):
            self.assertEqual(len(lowered[i]), self.n)
            for j in range(self.n):
                self.assertEqual(len(lowered[i][j]), self.n)
                for k in range(self.n):
                    self.assertEqual(len(lowered[i][j][k]), self.n)

    def test_raise_indices(self):
        """Test raising tensor indices."""
        raised = self.indexes.raise_indices(self.test_tensor, self.g_inv)
        
        # Check if tensor is raised
        self.assertIsNotNone(raised)
        self.assertEqual(len(raised), self.n)
        
        # Check structure of raised tensor
        for i in range(self.n):
            self.assertEqual(len(raised[i]), self.n)
            for j in range(self.n):
                self.assertEqual(len(raised[i][j]), self.n)
                for k in range(self.n):
                    self.assertEqual(len(raised[i][j][k]), self.n)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with n=1
        indexes_1d = Indexes(1)
        self.assertEqual(len(indexes_1d.generate_index_ricci()), 1)
        
        # Test with invalid metric
        with self.assertRaises(Exception):
            self.indexes.lower_indices(self.test_tensor, None)
        
        # Test with invalid tensor
        with self.assertRaises(Exception):
            self.indexes.lower_indices(None, self.g)

if __name__ == '__main__':
    unittest.main() 