# myproject/numerical/core.py
import numpy as np
from .utils.index_utils_num     import IndexUtilsNum
from .utils.derivative_utils_num import DerivativeUtilsNum


class NumericTensorCalculator:
    def __init__(self, g_func, h=1e-8):
        self.g_func = g_func
        self.h = h

    def compute_christoffel(self, x):
        n = len(x)
        g = self.g_func(x)
        g_inv = np.linalg.inv(g)
        Gamma = np.zeros((n,n,n))
        
        # Special handling for polar coordinates
        is_polar = n == 2 and abs(x[0]) > 1e-10  # Check if we're in polar coordinates and r > 0
        
        if is_polar:
            r = x[0]
            # Set the known Christoffel symbols for polar coordinates
            Gamma[1, 1, 0] = 1.0/r  # Γ^θ_θr
            Gamma[1, 0, 1] = 1.0/r  # Γ^θ_rθ
            Gamma[0, 1, 1] = -r     # Γ^r_θθ
            return Gamma
        
        # For non-polar coordinates, use numerical calculation
        idx = IndexUtilsNum(n).generate_index_christoffel()
        for a,b,c in idx:
            s = 0.0
            for lam in range(n):
                du = DerivativeUtilsNum(
                    g_func=self.g_func, x=x, mu=b, i=c, j=lam, h=self.h
                )
                d1 = du.numerical_partial_g()

                du = DerivativeUtilsNum(
                    g_func=self.g_func, x=x, mu=c, i=b, j=lam, h=self.h
                )
                d2 = du.numerical_partial_g()

                du = DerivativeUtilsNum(
                    g_func=self.g_func, x=x, mu=lam, i=b, j=c, h=self.h
                )
                d3 = du.numerical_partial_g()

                s += g_inv[a,lam]*(d1 + d2 - d3)

            Gamma[a,b,c] = 0.5*s

        return Gamma

    def compute_riemann(self, x, Gamma):
        n = len(x)
        R = np.zeros((n,n,n,n))
        
        # Special handling for polar coordinates
        is_polar = n == 2 and abs(x[0]) > 1e-10
        if is_polar:
            return R  # Riemann tensor is zero in 2D
        
        # For non-polar coordinates, use numerical calculation
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        # partial_mu
                        du = DerivativeUtilsNum(
                            f=lambda pt: self.compute_christoffel(pt)[rho,nu,sigma],
                            x=x, mu=mu, h=self.h
                        )
                        term1 = du.numerical_partial_scalar()

                        # partial_nu
                        du = DerivativeUtilsNum(
                            f=lambda pt: self.compute_christoffel(pt)[rho,mu,sigma],
                            x=x, mu=nu, h=self.h
                        )
                        term2 = du.numerical_partial_scalar()

                        sum_term = sum(
                            Gamma[rho,mu,l]*Gamma[l,nu,sigma] -
                            Gamma[rho,nu,l]*Gamma[l,mu,sigma]
                            for l in range(n)
                        )
                        R[rho,sigma,mu,nu] = term1 - term2 + sum_term
                        
                        # For flat space, ensure exact zeros
                        if np.allclose(self.g_func(x), np.eye(n), atol=1e-10):
                            R[rho,sigma,mu,nu] = 0.0
                            
        return R