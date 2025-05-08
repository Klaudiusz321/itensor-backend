import sympy as sp
from .simplification.custom_simplify import custom_simplify
from .utilis.indexes import Indexes
import logging

logger = logging.getLogger(__name__)

class ComputeTensor:
    def __init__(self, R_abcd, Ricci, Scalar_Curvature, g, g_inv, n):
        self.R_abcd = R_abcd
        self.Ricci = Ricci
        self.Scalar_Curvature = Scalar_Curvature
        self.g = g
        self.g_inv = g_inv
        self.n = n

        # Inicjalizacja menedżera indeksów
        self.indexes_manager = Indexes(n)
        self.christoffel_indexes = self.indexes_manager.generate_index_christoffel()
        self.riemann_indexes      = self.indexes_manager.generate_index_riemann()
        self.ricci_indexes        = self.indexes_manager.generate_index_ricci()

    def lower_riemann_indices(self):
        """Obniża indeksy tensora Riemanna za pomocą metryki."""
        self.R_abcd = self.indexes_manager.lower_indices(self.R_abcd, self.g)
        return self.R_abcd
    
    def raise_riemann_indices(self):
        """Podnosi indeksy tensora Riemanna za pomocą odwrotnej metryki."""
        self.R_abcd = self.indexes_manager.raise_indices(self.R_abcd, self.g_inv)
        return self.R_abcd

    def show_indexes(self):
        """Wyświetla automatycznie wygenerowane indeksy."""
        print("Indeksy Riemanna:", self.index_riemann)
        print("Indeksy Ricciego:", self.index_ricci)
        print("Indeksy Christoffela:", self.index_christoffel)

    def oblicz_tensory(self):
        """
        Oblicza podstawowe tensory geometryczne dla podanej metryki.
        """
        n = self.n
        logger.info(f"Rozpoczynam obliczenia tensorów dla {n} wymiarów")

        # Obliczanie tensora metrycznego
        g_inv = self.g.inv() if isinstance(self.g, sp.Matrix) else self.g_inv
        
        # Obliczanie symboli Christoffela
        Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    Gamma_sum = 0
                    for lam in range(n):
                        partial_mu = sp.diff(self.g[nu, lam], self.g[mu])
                        partial_nu = sp.diff(self.g[mu, lam], self.g[nu])
                        partial_lam = sp.diff(self.g[mu, nu], self.g[lam])
                        Gamma_sum += g_inv[sigma, lam] * (partial_mu + partial_nu - partial_lam)
                    Gamma[sigma][mu][nu] = custom_simplify(sp.Rational(1, 2) * Gamma_sum)
        
        # Obliczanie tensora Riemanna
        R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        term1 = sp.diff(Gamma[rho][nu][sigma], self.g[mu])
                        term2 = sp.diff(Gamma[rho][mu][sigma], self.g[nu])
                        sum_term = sum(
                            Gamma[rho][mu][lam] * Gamma[lam][nu][sigma] 
                            - Gamma[rho][nu][lam] * Gamma[lam][mu][sigma]
                            for lam in range(n)
                        )
                        R_abcd[rho][sigma][mu][nu] = custom_simplify(term1 - term2 + sum_term)

        # Obniżanie indeksów tensora Riemanna
        R_abcd = self.indexes_manager.lower_indices(R_abcd, self.g)

        # Obliczanie tensora Ricciego
        Ricci = sp.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                Ricci[mu, nu] = sum(R_abcd[rho][mu][rho][nu] for rho in range(n))
                Ricci[mu, nu] = custom_simplify(Ricci[mu, nu])

        # Skalar krzywizny (Ricci scalar)
        Scalar_Curvature = sum(
            g_inv[mu, nu] * Ricci[mu, nu] for mu in range(n) for nu in range(n)
        )
        Scalar_Curvature = custom_simplify(Scalar_Curvature)

        return self.g, Gamma, R_abcd, Ricci, Scalar_Curvature
