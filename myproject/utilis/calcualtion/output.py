import sympy as sp
import numpy as np
from ..simplification import custom_simplify
def generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n):
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
        

    return "\n".join(lines)



def generate_numerical_curvature(Scalar_Curvature, wspolrzedne, parametry, ranges, points_per_dim=50):
    """
    Generuje numeryczne wartości krzywizny dla wizualizacji
    """
    try:
        # Konwertujemy wyrażenie symboliczne na funkcję numeryczną
        scalar_curvature_fn = sp.lambdify(list(wspolrzedne) + list(parametry), Scalar_Curvature, modules=['numpy'])
        
        # Tworzymy siatki punktów dla każdej współrzędnej
        coordinate_grids = []
        for coord, (min_val, max_val) in zip(wspolrzedne, ranges):
            grid = np.linspace(min_val, max_val, points_per_dim)
            coordinate_grids.append(grid)
        
        # Tworzymy siatkę punktów
        mesh_grids = np.meshgrid(*coordinate_grids)
        points = np.vstack([grid.flatten() for grid in mesh_grids]).T
        
        # Obliczamy wartości krzywizny dla każdego punktu
        curvature_values = []
        for point in points:
            try:
                # Dodajemy wartości parametrów (np. masa M dla metryki Schwarzschilda)
                full_point = list(point) + [1.0]  # 1.0 to domyślna wartość parametru (np. M)
                value = float(scalar_curvature_fn(*full_point))
                if np.isfinite(value):
                    curvature_values.append(value)
                else:
                    curvature_values.append(0)
            except Exception as e:
                print(f"Error calculating point {point}: {e}")
                curvature_values.append(0)
        
        curvature_values = np.array(curvature_values)
        
        # Normalizacja wartości
        valid_values = curvature_values[np.isfinite(curvature_values)]
        if len(valid_values) > 0:
            vmin, vmax = np.percentile(valid_values, [5, 95])
            curvature_values = np.clip(curvature_values, vmin, vmax)
        
        return {
            'points': points.tolist(),
            'values': curvature_values.tolist(),
            'ranges': ranges,
            'coordinates': [str(coord) for coord in wspolrzedne],
            'parameters': [str(param) for param in parametry]
        }
    except Exception as e:
        print(f"Error in generate_numerical_curvature: {e}")
        return None



