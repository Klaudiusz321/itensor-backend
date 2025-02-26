import numpy as np
import sympy as sp
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

def auto_substitute_defaults(expr, default=1.0, user_defaults=None):
   
    if user_defaults is None:
        user_defaults = {}
    
    subs = {}
    # expr.free_symbols zwraca zbiór wszystkich wolnych symboli w wyrażeniu
    for s in expr.free_symbols:
        # Jeśli użytkownik podał wartość dla danego symbolu (na podstawie nazwy),
        # używamy jej, w przeciwnym razie podstawiamy domyślną wartość.
        subs[s] = user_defaults.get(s.name, default)
    
    # Podstawienie wartości do wyrażenia
    new_expr = expr.subs(subs)
    return new_expr

def generate_numerical_curvature(Scalar_Curvature, wspolrzedne, parametry, ranges, points_per_dim=50):
    """
    Generuje numeryczne wartości krzywizny dla wizualizacji
    """
    try:
        print("\n=== Debug generate_numerical_curvature ===")
        print(f"Oryginalne wyrażenie: {Scalar_Curvature}")

        # 1. Podstawiamy wartości za pochodne czasowe i funkcje czasu
        substitutions = {
            sp.Symbol('t'): 1.0,  # Ustalony czas
            sp.Function('a')(sp.Symbol('t')): 1.0,  # a(t) = 1
            sp.Derivative(sp.Function('a')(sp.Symbol('t')), sp.Symbol('t')): 0.1,  # a'(t) = 0.1
            sp.Derivative(sp.Function('a')(sp.Symbol('t')), (sp.Symbol('t'), 2)): 0.01  # a''(t) = 0.01
        }
        
        simplified_expr = Scalar_Curvature.subs(substitutions)
        print(f"Po podstawieniu wartości czasowych: {simplified_expr}")

        # 2. Podstawiamy wartości za pozostałe parametry
        all_symbols = simplified_expr.free_symbols
        print(f"Pozostałe symbole: {all_symbols}")
        
        spatial_coords = [coord for coord in wspolrzedne if str(coord) != 't']
        undefined_symbols = [sym for sym in all_symbols 
                           if sym not in spatial_coords and sym not in parametry]
        
        if undefined_symbols:
            print(f"Podstawiam wartości za symbole: {undefined_symbols}")
            for sym in undefined_symbols:
                simplified_expr = simplified_expr.subs(sym, 1.0)
        
        print(f"Wyrażenie końcowe: {simplified_expr}")

        # 3. Tworzymy funkcję numeryczną
        scalar_curvature_fn = sp.lambdify(spatial_coords, simplified_expr, modules=['numpy'])
        
        # 4. Generujemy siatkę punktów
        coordinate_grids = []
        for coord, (min_val, max_val) in zip(spatial_coords, ranges):
            grid = np.linspace(min_val, max_val, points_per_dim)
            coordinate_grids.append(grid)
        
        mesh_grids = np.meshgrid(*coordinate_grids)
        points = np.vstack([grid.flatten() for grid in mesh_grids]).T
        
        # 5. Obliczamy wartości
        curvature_values = []
        for point in points:
            try:
                value = float(scalar_curvature_fn(*point))
                curvature_values.append(value if np.isfinite(value) else 0)
            except Exception as e:
                print(f"Błąd dla punktu {point}: {e}")
                curvature_values.append(0)
        
        curvature_values = np.array(curvature_values)
        
        # 6. Normalizacja
        valid_values = curvature_values[np.isfinite(curvature_values)]
        if len(valid_values) > 0:
            vmin, vmax = np.percentile(valid_values, [5, 95])
            curvature_values = np.clip(curvature_values, vmin, vmax)
        
        return {
            'points': points.tolist(),
            'values': curvature_values.tolist(),
            'ranges': ranges
        }
        
    except Exception as e:
        print(f"\nBŁĄD w generate_numerical_curvature: {e}")
        import traceback
        traceback.print_exc()
        return None 