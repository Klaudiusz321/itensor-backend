import unittest
import sympy as sp
from myproject.utils.symbolic.compute_tensor import ComputeTensor

class TestComputeTensor(unittest.TestCase):
    
    def setUp(self):
        self.n = 4
        self.g = sp.eye(self.n)
        self.g_inv = sp.eye(self.n)
        self.R_abcd = [[[[1 for _ in range(self.n)] for _ in range(self.n)] 
                        for _ in range(self.n)] for _ in range(self.n)]
        self.Ricci = {}
        self.Scalar_Curvature = 0
        
        self.tensor = ComputeTensor(
            R_abcd=self.R_abcd,
            Ricci=self.Ricci,
            Scalar_Curvature=self.Scalar_Curvature,
            g=self.g,
            g_inv=self.g_inv,
            n=self.n
        )

    def test_generate_indexes(self):
        # Nie wywołujemy Indexes ręcznie, korzystamy z pól ComputeTensor
        self.assertTrue(len(self.tensor.christoffel_indexes) > 0)
        self.assertTrue(len(self.tensor.riemann_indexes) > 0)
        self.assertTrue(len(self.tensor.ricci_indexes) > 0)

    def test_lower_riemann_indices(self):
        lowered = self.tensor.lower_riemann_indices()
        self.assertIsNotNone(lowered)
        self.assertEqual(len(lowered), self.n)

    def test_raise_riemann_indices(self):
        raised = self.tensor.raise_riemann_indices()
        self.assertIsNotNone(raised)
        self.assertEqual(len(raised), self.n)

if __name__ == "__main__":
    unittest.main()
