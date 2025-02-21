import numpy as np
import sympy as sp

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