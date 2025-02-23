import sympy as sp
import numpy as np
from scipy.interpolate import griddata

def generate_numerical_curvature(Scalar_Curvature, wspolrzedne, parametry, ranges, points_per_dim=50):
    """
    Generuje numeryczne wartości krzywizny dla wizualizacji.
    
    Parametry:
      - Scalar_Curvature (sp.Expr): Wyrażenie symboliczne opisujące krzywiznę skalarną.
      - wspolrzedne (list): Lista symboli współrzędnych (sp.Symbol).
      - parametry (list): Lista dodatkowych parametrów (sp.Symbol), np. masa M dla metryki Schwarzschilda.
      - ranges (list): Lista krotek (min, max) dla każdej współrzędnej.
      - points_per_dim (int): Liczba punktów na wymiar (domyślnie 50).
    
    Zwraca:
      Słownik z:
        - 'points': Listę punktów (jako listy wartości) utworzonych na siatce.
        - 'values': Listę odpowiadających wartości krzywizny.
        - 'ranges': Podane zakresy.
        - 'coordinates': Lista nazw współrzędnych (jako stringi).
        - 'parameters': Lista nazw parametrów (jako stringi).
    
    W przypadku błędu wypisuje komunikat i zwraca None.
    """
    try:
        # Konwertujemy wyrażenie symboliczne na funkcję numeryczną (moduły: numpy)
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
                # Dodajemy wartości parametrów (np. masa M), tutaj domyślnie ustawiamy 1.0
                full_point = list(point) + [1.0]
                value = float(scalar_curvature_fn(*full_point))
                if np.isfinite(value):
                    curvature_values.append(value)
                else:
                    curvature_values.append(0)
            except Exception as e:
                print(f"Error calculating point {point}: {e}")
                curvature_values.append(0)
        
        curvature_values = np.array(curvature_values)
        
        # Normalizacja wartości: przycinamy do percentyla 5 i 95
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
