import logging
import sympy as sp
from utils.symbolic.simplification.custom_simplify import weyl_simplify
from .compute_tensor import ComputeTensor

logger = logging.getLogger(__name__)

class ComputeWeylTensor:
    """Klasa do obliczania tensora Weyla na podstawie obiektu ComputeTensor."""
    
    def __init__(self, compute_tensor: ComputeTensor):
        self.R_abcd = compute_tensor.R_abcd
        self.Ricci = compute_tensor.Ricci
        self.Scalar_Curvature = compute_tensor.Scalar_Curvature
        self.g = compute_tensor.g
        self.g_inv = compute_tensor.g_inv
        self.n = compute_tensor.n

    def compute_weyl_tensor(self):
        logger.info("Obliczam tensor Weyla")
        
        Weyl = [[[[0 for _ in range(self.n)] for _ in range(self.n)] 
                 for _ in range(self.n)] for _ in range(self.n)]
        
        if self.n < 4:
            logger.warning(f"Tensor Weyla jest zawsze zero dla n={self.n} < 4")
            return Weyl
        
        is_flat = all(value == 0 for value in self.R_abcd.values()) if isinstance(self.R_abcd, dict) else False
        
        if is_flat:
            logger.info("Flat space detected - Weyl tensor is zero")
            return Weyl
        
        factor_1 = 2 / (self.n - 2)
        factor_2 = 2 / ((self.n - 1) * (self.n - 2))
        
        if isinstance(self.Ricci, dict):
            Ricci_matrix = sp.zeros(self.n, self.n)
            for (i, j), value in self.Ricci.items():
                Ricci_matrix[i, j] = value
            self.Ricci = Ricci_matrix
        
        if isinstance(self.R_abcd, dict):
            R_matrix = [[[[0 for _ in range(self.n)] for _ in range(self.n)] 
                         for _ in range(self.n)] for _ in range(self.n)]
            for (a, b, c, d), value in self.R_abcd.items():
                R_matrix[a][b][c][d] = value
            self.R_abcd = R_matrix
        
        for rho in range(self.n):
            for sigma in range(self.n):
                for mu in range(self.n):
                    for nu in range(self.n):
                        riemann_term = self.R_abcd[rho][sigma][mu][nu]
                        
                        ricci_combined = factor_1 * (
                            self.g[rho, mu] * self.Ricci[sigma, nu] -
                            self.g[rho, nu] * self.Ricci[sigma, mu] -
                            self.g[sigma, nu] * self.Ricci[rho, mu] +
                            self.g[sigma, mu] * self.Ricci[rho, nu]
                        )
                        
                        scalar_combined = factor_2 * self.Scalar_Curvature * (
                            self.g[rho, mu] * self.g[sigma, nu] -
                            self.g[rho, nu] * self.g[sigma, mu]
                        )
                        
                        Weyl[rho][sigma][mu][nu] = weyl_simplify(
                            riemann_term - ricci_combined + scalar_combined
                        )
        
        return Weyl
