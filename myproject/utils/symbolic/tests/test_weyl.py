import unittest
import sympy as sp
from myproject.utils.symbolic.compute_tensor import ComputeTensor
from myproject.utils.symbolic.weyl import ComputeWeylTensor

class TestComputeWeylTensor(unittest.TestCase):
    
    def setUp(self):
        self.n = 4
        self.g = sp.eye(self.n)
        self.g_inv = sp.eye(self.n)
        self.R_abcd = {
            (0, 1, 0, 1): 1,
            (1, 0, 1, 0): -1
        }
        self.Ricci = {(0, 0): 1, (1, 1): 1}
        self.Scalar_Curvature = 2
        
        self.tensor = ComputeTensor(
            R_abcd=self.R_abcd,
            Ricci=self.Ricci,
            Scalar_Curvature=self.Scalar_Curvature,
            g=self.g,
            g_inv=self.g_inv,
            n=self.n
        )

    def test_compute_weyl_tensor_flat(self):
        # Ustawienie przestrzeni płaskiej
        self.tensor.R_abcd = {}
        self.tensor.Ricci = {}
        self.tensor.Scalar_Curvature = 0
        
        weyl_calculator = ComputeWeylTensor(self.tensor)
        result = weyl_calculator.compute_weyl_tensor()
        
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        self.assertEqual(result[i][j][k][l], 0)

    def test_compute_weyl_tensor_non_flat(self):
        # Test dla przestrzeni niepłaskiej
        weyl_calculator = ComputeWeylTensor(self.tensor)
        result = weyl_calculator.compute_weyl_tensor()
        
        # Sprawdzenie, czy tensor Weyla jest niezerowy
        non_zero = any(
            result[i][j][k][l] != 0 
            for i in range(self.n) 
            for j in range(self.n)
            for k in range(self.n) 
            for l in range(self.n)
        )
        self.assertTrue(non_zero)

if __name__ == "__main__":
    unittest.main()
