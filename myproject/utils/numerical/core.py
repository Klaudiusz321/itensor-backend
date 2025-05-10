# myproject/utils/numerical/core.py

import numpy as np
from .utils.index_utils_num import IndexUtilsNum
from .utils.derivative_utils_num import DerivativeUtilsNum

class NumericTensorCalculator:
    def __init__(self, g_func, h=1e-4):
        """
        g_func: funkcja przyjmująca wektor x (1D numpy array)
                i zwracająca macierz metryki g(x) (nxn numpy array)
        h: krok dla różnic skończonych
        """
        self.g_func = g_func
        self.h = h

    def compute_christoffel(self, x):
        n = len(x)
        g = self.g_func(x)

        # odwracanie macierzy metryki, z fallbackem na pseudoinwersję
        try:
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            g_inv = np.linalg.pinv(g)

        Gamma = np.zeros((n, n, n))

        # uniwersalne numeryczne wyliczenie symboli Christoffela:
        idx = IndexUtilsNum(n).generate_index_christoffel()
        for a, b, c in idx:
            s = 0.0
            for lam in range(n):
                # ∂_b g_{c,lam}
                du1 = DerivativeUtilsNum(self.g_func, x, mu=b, i=c, j=lam, h=self.h)
                d1 = du1.numerical_partial_g()
                # ∂_c g_{b,lam}
                du2 = DerivativeUtilsNum(self.g_func, x, mu=c, i=b, j=lam, h=self.h)
                d2 = du2.numerical_partial_g()
                # ∂_lam g_{b,c}
                du3 = DerivativeUtilsNum(self.g_func, x, mu=lam, i=b, j=c, h=self.h)
                d3 = du3.numerical_partial_g()

                s += g_inv[a, lam] * (d1 + d2 - d3)

            Gamma[a, b, c] = 0.5 * s
            
            # Add symmetric component if b != c
            if b != c:
                Gamma[a, c, b] = Gamma[a, b, c]  # Symmetry in lower indices

        return Gamma

    def compute_riemann(self, x, Gamma):
        n = len(x)
        R = np.zeros((n, n, n, n))

        # uniwersalne numeryczne wyliczenie tensora Riemanna:
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        # ∂_mu Γ^rho_{sigma,nu}
                        du1 = DerivativeUtilsNum(
                            f=lambda pt: self.compute_christoffel(pt)[rho, sigma, nu],
                            x=x, mu=mu, h=self.h
                        )
                        term1 = du1.numerical_partial_scalar()

                        # ∂_nu Γ^rho_{sigma,mu}
                        du2 = DerivativeUtilsNum(
                            f=lambda pt: self.compute_christoffel(pt)[rho, sigma, mu],
                            x=x, mu=nu, h=self.h
                        )
                        term2 = du2.numerical_partial_scalar()

                        # suma Γ^rho_{mu,lam} Γ^lam_{sigma,nu} − Γ^rho_{nu,lam} Γ^lam_{sigma,mu}
                        sum_term = 0.0
                        for lam in range(n):
                            sum_term += (
                                Gamma[rho, mu, lam] * Gamma[lam, sigma, nu]
                                - Gamma[rho, nu, lam] * Gamma[lam, sigma, mu]
                            )

                        R[rho, sigma, mu, nu] = term1 - term2 + sum_term

        return R

    def compute_all(self, x):
        """
        Zwraca słownik ze wszystkimi wynikami:
          - metric:        g_{ij}
          - metric_inv:    g^{ij}
          - christoffel:   Γ^i_{jk}
          - riemann_lower: R_{ijkl}
          - ricci:         R_{ij}
          - scalar:        R
          - einstein_lower:G_{ij}
          - einstein_upper:G^i{}_j
        """
        # 1) metryka
        g = self.g_func(x)
        try:
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            g_inv = np.linalg.pinv(g)

        # 2) symbole Christoffela
        Gamma = self.compute_christoffel(x)
        
        # Print non-zero Christoffel symbols for debugging
        print("Computed Christoffel symbols:")
        n = len(x)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if abs(Gamma[i,j,k]) > 1e-6:
                        print(f"Γ^{i}_{{{j}{k}}} = {Gamma[i,j,k]}")

        # 3) pełny tensor Riemanna
        R = self.compute_riemann(x, Gamma)
        
        # Print non-zero Riemann tensor components for debugging
        print("Computed Riemann tensor components:")
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        if abs(R[i,j,k,l]) > 1e-6:
                            print(f"R_{{{i}{j}{k}{l}}} = {R[i,j,k,l]}")

        # 4) tensor Ricciego: Ric_{μν} = R^ρ_{ μ ρ ν}
        n = len(x)
        Ric = np.zeros((n, n))
        for mu in range(n):
            for nu in range(n):
                Ric[mu, nu] = sum(R[rho, mu, rho, nu] for rho in range(n))
                
        # Print non-zero Ricci tensor components for debugging
        print("Computed Ricci tensor components:")
        for i in range(n):
            for j in range(n):
                if abs(Ric[i,j]) > 1e-6:
                    print(f"Ric_{{{i}{j}}} = {Ric[i,j]}")

        # 5) krzywizna skalarna: R = g^{μν} Ric_{μν}
        # używamy tensordot, aby poprawnie zsumować po obu indeksach
        R_scalar = float(np.tensordot(g_inv, Ric, axes=([0, 1], [0, 1])))
        print(f"Scalar curvature: {R_scalar}")

        # 6) tensor Einsteina (niższe indeksy)
        G_lower = Ric - 0.5 * R_scalar * g

        # 7) tensor Einsteina (podniesiony indeks)
        G_upper = g_inv @ G_lower

        return {
            "metric":         g,
            "metric_inv":     g_inv,
            "christoffel":    Gamma,
            "riemann_lower":  R,
            "ricci":          Ric,
            "scalar":         R_scalar,
            "einstein_lower": G_lower,
            "einstein_upper": G_upper,
        }
