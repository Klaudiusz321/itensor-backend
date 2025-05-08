import sympy as sp
import numpy as np
from ..simplification.custom_simplify import custom_simplify
import logging


class Output:
    def __init__(self, g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n, Weyl=None):
        self.g = g
        self.Gamma = Gamma
        self.R_abcd = R_abcd
        self.Ricci = Ricci
        self.Scalar_Curvature = Scalar_Curvature
        self.n = n
        self.Weyl = Weyl
        self.G_upper = G_upper
        self.G_lower = G_lower
    def generate_output(self):
        
        logger = logging.getLogger(__name__)
        logger.info(f"Generowanie wyników dla przestrzeni {self.n}-wymiarowej")
        
        lines = []

        # METRYKA
        lines.append("Metric tensor components (textual format and LaTeX):")
        for i in range(self.n):
            for j in range(i, self.n):
                val = custom_simplify(self.g[i, j])
                if val != 0:
                    lines.append(f"g_({i}{j}) = {val}")
                    lines.append(f"g_{{{i}{j}}} = \\({sp.latex(val)}\\)")

        # SYMBOL CHRISTOFFELA
        lines.append("Non-zero Christoffel symbols (textual format and LaTeX):")
        for a in range(self.n):
            for b in range(self.n):
                for c in range(self.n):
                    val = custom_simplify(self.Gamma[a][b][c])
                    if val != 0:
                        lines.append(f"Γ^({a})_({b}{c}) = {val}")
                        lines.append(f"\\Gamma^{{{a}}}_{{{b}{c}}} = \\({sp.latex(val)}\\)")

        # TENSOR RIEMANNA
        lines.append("Non-zero components of the Riemann tensor:")
        for a in range(self.n):
            for b in range(self.n):
                for c in range(self.n):
                    for d in range(self.n):
                        val = custom_simplify(self.R_abcd[a][b][c][d])
                        if val != 0:
                            lines.append(f"R_({a}{b}{c}{d}) = {val}")
                            lines.append(f"R_{{{a}{b}{c}{d}}} = \\({sp.latex(val)}\\)")

        # TENSOR RICCIEGO
        lines.append("Non-zero components of the Ricci tensor:")
        for i in range(self.n):
            for j in range(self.n):
                val = custom_simplify(self.Ricci[i, j])
                if val != 0:
                    lines.append(f"R_({i}{j}) = {val}")
                    lines.append(f"R_{{{i}{j}}} = \\({sp.latex(val)}\\)")

        # TENSOR EINSTEINA
        lines.append("Non-zero Einstein tensor components:")
        for i in range(self.n):
            for j in range(self.n):
                val = custom_simplify(self.G_lower[i, j])
                if val != 0:
                    lines.append(f"G_({i}{j}) = {val}")
                    lines.append(f"G_{{{i}{j}}} = \\({sp.latex(val)}\\)")

        # KRZYWIZNA SKALARNA
        if self.Scalar_Curvature != 0:
            lines.append("Scalar curvature R:")
            lines.append(f"R = {self.Scalar_Curvature}")
            lines.append(f"R = \\({sp.latex(self.Scalar_Curvature)}\\)")

        # TENSOR WEYLA
        if self.Weyl is not None:
            lines.append("Non-zero Weyl tensor components:")
            has_nonzero_weyl = False
            
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        for l in range(self.n):
                            val = custom_simplify(self.Weyl[i][j][k][l])
                            if val != 0:
                                has_nonzero_weyl = True
                                lines.append(f"C_{i}{j}{k}{l} = {val}")
                                lines.append(f"C_{{{i}{j}{k}{l}}} = \\({sp.latex(val)}\\)")
            
            if not has_nonzero_weyl:
                # Jeśli wszystkie komponenty są zerowe, dodaj specjalną notatkę
                lines.append("Wszystkie komponenty tensora Weyla są zerowe.")
                lines.append("C_{ijkl} = 0 dla wszystkich indeksów.")
                
                # Sprawdź czy to może być metryka FLRW
                if self.n == 4:
                    # Sprawdź charakterystyczne cechy FLRW
                    if abs(self.g[0,0] + 1) < 1e-10:  # g_00 = -1
                        diag_only = True
                        for i in range(self.n):
                            for j in range(self.n):
                                if i != j and abs(self.g[i,j]) > 1e-10:
                                    diag_only = False
                                    break
                        if diag_only:
                            lines.append("Zerowy tensor Weyla sugeruje, że to jest metryka FLRW (jednorodna i izotropowa przestrzeń).")
                
                # Ogólna informacja o interpretacji
                lines.append("Zerowy tensor Weyla oznacza, że przestrzeń jest lokalnie konforemnie płaska.")
        else:
            lines.append("Tensor Weyla nie został obliczony.")

        return "\n".join(lines)



