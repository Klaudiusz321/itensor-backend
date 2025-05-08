import sympy as sp
import numpy as np
from ..simplification.custom_simplify import custom_simplify
import logging

def generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n, Weyl=None):
    """
    Generuje wynikowy tekst na podstawie obliczonych tensorów.
    
    Parametry:
    g - tensor metryczny
    Gamma - symbole Christoffela
    R_abcd - tensor Riemanna
    Ricci - tensor Ricciego
    Scalar_Curvature - krzywizna skalarna
    G_upper - tensor Einsteina z podniesionymi indeksami
    G_lower - tensor Einsteina z opuszczonymi indeksami
    n - wymiar przestrzeni
    Weyl - tensor Weyla (opcjonalny)
    
    Zwraca:
    Sformatowany tekst z wynikami obliczeń.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generowanie wyników dla przestrzeni {n}-wymiarowej")
    
    lines = []

    # METRYKA
    lines.append("Metric tensor components (textual format and LaTeX):")
    for i in range(n):
        for j in range(i, n):
            val = custom_simplify(g[i, j])
            if val != 0:
                lines.append(f"g_({i}{j}) = {val}")
                lines.append(f"g_{{{i}{j}}} = \\({sp.latex(val)}\\)")

    # SYMBOL CHRISTOFFELA
    lines.append("Non-zero Christoffel symbols (textual format and LaTeX):")
    for a in range(n):
        for b in range(n):
            for c in range(n):
                val = custom_simplify(Gamma[a][b][c])
                if val != 0:
                    lines.append(f"Γ^({a})_({b}{c}) = {val}")
                    lines.append(f"\\Gamma^{{{a}}}_{{{b}{c}}} = \\({sp.latex(val)}\\)")

    # TENSOR RIEMANNA
    lines.append("Non-zero components of the Riemann tensor:")
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    val = custom_simplify(R_abcd[a][b][c][d])
                    if val != 0:
                        lines.append(f"R_({a}{b}{c}{d}) = {val}")
                        lines.append(f"R_{{{a}{b}{c}{d}}} = \\({sp.latex(val)}\\)")

    # TENSOR RICCIEGO
    lines.append("Non-zero components of the Ricci tensor:")
    for i in range(n):
        for j in range(n):
            val = custom_simplify(Ricci[i, j])
            if val != 0:
                lines.append(f"R_({i}{j}) = {val}")
                lines.append(f"R_{{{i}{j}}} = \\({sp.latex(val)}\\)")

    # TENSOR EINSTEINA
    lines.append("Non-zero Einstein tensor components:")
    for i in range(n):
        for j in range(n):
            val = custom_simplify(G_lower[i, j])
            if val != 0:
                lines.append(f"G_({i}{j}) = {val}")
                lines.append(f"G_{{{i}{j}}} = \\({sp.latex(val)}\\)")

    # KRZYWIZNA SKALARNA
    if Scalar_Curvature != 0:
        lines.append("Scalar curvature R:")
        lines.append(f"R = {Scalar_Curvature}")
        lines.append(f"R = \\({sp.latex(Scalar_Curvature)}\\)")

    # TENSOR WEYLA
    if Weyl is not None:
        lines.append("Non-zero Weyl tensor components:")
        has_nonzero_weyl = False
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        val = custom_simplify(Weyl[i][j][k][l])
                        if val != 0:
                            has_nonzero_weyl = True
                            lines.append(f"C_{i}{j}{k}{l} = {val}")
                            lines.append(f"C_{{{i}{j}{k}{l}}} = \\({sp.latex(val)}\\)")
        
        if not has_nonzero_weyl:
            # Jeśli wszystkie komponenty są zerowe, dodaj specjalną notatkę
            lines.append("Wszystkie komponenty tensora Weyla są zerowe.")
            lines.append("C_{ijkl} = 0 dla wszystkich indeksów.")
            
            # Sprawdź czy to może być metryka FLRW
            if n == 4:
                # Sprawdź charakterystyczne cechy FLRW
                if abs(g[0,0] + 1) < 1e-10:  # g_00 = -1
                    diag_only = True
                    for i in range(n):
                        for j in range(n):
                            if i != j and abs(g[i,j]) > 1e-10:
                                diag_only = False
                                break
                    if diag_only:
                        lines.append("Zerowy tensor Weyla sugeruje, że to jest metryka FLRW (jednorodna i izotropowa przestrzeń).")
            
            # Ogólna informacja o interpretacji
            lines.append("Zerowy tensor Weyla oznacza, że przestrzeń jest lokalnie konforemnie płaska.")
    else:
        lines.append("Tensor Weyla nie został obliczony.")

    return "\n".join(lines)



