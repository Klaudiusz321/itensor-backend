import numpy as np
import sympy as sp
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Ustaw backend niewymagający GUI
import io
import base64

def generate_numerical_curvature(Scalar_Curvature, wspolrzedne, parametry, ranges, points_per_dim=15):
    try:
        print("\nAnaliza wejścia:")
        print("Scalar_Curvature:", Scalar_Curvature)
        print("Współrzędne:", [str(w) for w in wspolrzedne])
        print("Parametry:", [str(p) for p in parametry])

        def calculate_curvature_value(*coords):
            """Oblicza wartość krzywizny dla danych współrzędnych"""
            try:
                subs_dict = {str(w): c for w, c in zip(wspolrzedne, coords)}
                return float(Scalar_Curvature.subs(subs_dict))
            except Exception as e:
                print(f"Error in calculate_curvature_value: {e}")
                return np.nan

        def get_coordinate_ranges():
            """Określa zakresy dla różnych typów współrzędnych"""
            ranges = []
            for coord in wspolrzedne:
                name = str(coord)
                if name == 't':
                    ranges.append([0, 2.0])
                elif name == 'chi':
                    ranges.append([-0.99, 0.99])  # unikamy k*chi^2 = 1
                elif name == 'theta':
                    ranges.append([1e-3, np.pi - 1e-3])
                elif name == 'phi':
                    ranges.append([0, 2*np.pi - 1e-3])
                else:
                    ranges.append([-1.0, 1.0])
            return ranges

        # Generujemy punkty
        coord_ranges = get_coordinate_ranges()
        grid_points = []
        for min_val, max_val in coord_ranges:
            grid_points.append(np.linspace(min_val, max_val, points_per_dim))
        
        # Tworzymy siatkę punktów
        mesh_grids = np.meshgrid(*grid_points)
        points = np.vstack([grid.flatten() for grid in mesh_grids]).T
        
        # Obliczamy wartości krzywizny
        curvature_values = []
        for point in points:
            value = calculate_curvature_value(*point)
            curvature_values.append(value)

        curvature_values = np.array(curvature_values)
        
        # Usuwamy wartości ekstremalne
        nonzero_values = curvature_values[np.abs(curvature_values) > 1e-10]
        if len(nonzero_values) > 0:
            percentile_5 = np.percentile(nonzero_values, 5)
            percentile_95 = np.percentile(nonzero_values, 95)
            curvature_values = np.clip(curvature_values, percentile_5, percentile_95)
            
            print(f"\nStatystyki krzywizny:")
            print(f"Min: {np.min(nonzero_values)}")
            print(f"Max: {np.max(nonzero_values)}")
            print(f"Średnia: {np.mean(nonzero_values)}")
            print(f"Mediana: {np.median(nonzero_values)}")

        # Optymalizacja pamięci
        plt.clf()  # Wyczyść poprzednie wykresy
        plt.close('all')  # Zamknij wszystkie figury
        
        # Mniejszy wykres
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        
        # Ograniczamy liczbę punktów
        max_points = 500  # jeszcze mniej punktów
        if len(points) > max_points:
            step = len(points) // max_points
            plot_points = points[::step]
            plot_values = curvature_values[::step]
        else:
            plot_points = points
            plot_values = curvature_values

        scatter = ax.scatter(plot_points[:, 0], 
                           plot_points[:, 1], 
                           plot_values,
                           c=plot_values,
                           cmap='viridis',
                           s=20)  # jeszcze mniejsze punkty
        
        # Optymalizacja zapisu
        buf = io.BytesIO()
        plt.savefig(buf, format='png', 
                   bbox_inches='tight', 
                   dpi=80,
                   optimize=True, 
                   quality=70)  # większa kompresja
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Czyszczenie pamięci
        plt.close(fig)
        buf.close()

        return {
            'plot': plot_data,
            'coordinates': [str(coord) for coord in wspolrzedne]
        }

    except Exception as e:
        print(f"Error in generate_numerical_curvature: {e}")
        import traceback
        traceback.print_exc()
        return None
