import numpy as np
import sympy as sp
from myproject.utilis.calcualtion.derivative import numeric_derivative, total_derivative

def generate_numerical_curvature(Scalar_Curvature, wspolrzedne, parametry, ranges, points_per_dim=50):
    
    try:
        print("\nAnaliza wejścia:")
        print("Scalar_Curvature:", Scalar_Curvature)
        print("Współrzędne:", [str(w) for w in wspolrzedne])
        print("Parametry:", [str(p) for p in parametry])

        # 1. Znajdujemy symbole używane w wyrażeniu
        t = [s for s in wspolrzedne if str(s) == 't'][0] if any(str(s) == 't' for s in wspolrzedne) else sp.Symbol('t')
        a = sp.Function('a')(t)  # Używamy tego samego symbolu co w Scalar_Curvature
        
        # Znajdujemy pochodne w wyrażeniu
        derivatives = Scalar_Curvature.atoms(sp.Derivative)
        print("Znalezione pochodne:", derivatives)

        def safe_scalar_curvature(chi_val, theta_val, phi_val):
            try:
                # Sprawdzamy warunki osobliwości z mniejszą czułością
                if abs(np.sin(theta_val)) < 1e-5 and abs(theta_val) < 1e-5:
                    # Jesteśmy blisko θ = 0
                    theta_val = 1e-5
                elif abs(np.sin(theta_val)) < 1e-5 and abs(theta_val - np.pi) < 1e-5:
                    # Jesteśmy blisko θ = π
                    theta_val = np.pi - 1e-5

                # Obliczamy wartości funkcji skali i jej pochodnych
                t_val = 2.0
                
                # Funkcja skali z zależnością od współrzędnych
                current_a = np.cosh(t_val) * (1 + 0.1 * chi_val**2) * np.sin(theta_val)
                
                # Pierwsza pochodna
                current_a_prime = (np.sinh(t_val) * (1 + 0.1 * chi_val**2) * np.sin(theta_val) + 
                                 0.2 * chi_val * np.cosh(t_val) * np.sin(theta_val))
                
                # Druga pochodna
                current_a_double_prime = (np.cosh(t_val) * (1 + 0.1 * chi_val**2) * np.sin(theta_val) +
                                        0.4 * chi_val * np.sinh(t_val) * np.sin(theta_val) +
                                        0.2 * np.cosh(t_val) * np.sin(theta_val))

                # Parametr k z zależnością od współrzędnych
                k_value = 1.0 + 0.1 * np.sin(theta_val) * np.cos(phi_val)

                print(f"\nWartości w punkcie [chi={chi_val}, theta={theta_val}, phi={phi_val}]:")
                print(f"a(t) = {current_a}")
                print(f"a'(t) = {current_a_prime}")
                print(f"a''(t) = {current_a_double_prime}")
                print(f"k = {k_value}")

                # Tworzymy podstawienia używając oryginalnych symboli
                substitutions = {
                    t: t_val,
                    a: current_a,
                    sp.Derivative(a, t): current_a_prime,
                    sp.Derivative(a, (t, 2)): current_a_double_prime,
                    sp.Symbol('k'): k_value,
                    sp.Symbol('chi'): chi_val,
                    sp.Symbol('theta'): theta_val,
                    sp.Symbol('phi'): phi_val
                }

                # Wykonujemy podstawienia i upraszczamy
                expr = Scalar_Curvature.subs(substitutions)
                expr = sp.simplify(expr)
                
                try:
                    result = float(expr.evalf())
                    if not np.isfinite(result) or abs(result) > 1e10:
                        print(f"Nieskończona lub zbyt duża wartość: {result}")
                        return 0.0
                    return result
                except Exception as e:
                    print(f"Błąd konwersji: {e}")
                    print("Wyrażenie:", expr)
                    return 0.0

            except Exception as e:
                print(f"Błąd w obliczeniach: {e}")
                return 0.0

        # 3. Generujemy punkty (z unikaniem osobliwości)
        modified_ranges = []
        eps = 1e-3
        for coord in wspolrzedne:
            name = str(coord)
            if name == 'theta':
                modified_ranges.append([eps, np.pi - eps])
            elif name == 'phi':
                modified_ranges.append([0, 2*np.pi - eps])
            else:
                modified_ranges.append([-2.0, 2.0])

        coordinate_grids = []
        for (min_val, max_val) in modified_ranges:
            grid = np.linspace(min_val, max_val, points_per_dim)
            coordinate_grids.append(grid)

        mesh_grids = np.meshgrid(*coordinate_grids)
        points = np.vstack([grid.flatten() for grid in mesh_grids]).T

        # 4. Obliczamy wartości krzywizny
        curvature_values = []
        for i, point in enumerate(points):
            if i % 100 == 0:  # Status co 100 punktów
                print(f"Przetwarzanie punktu {i}/{len(points)}")
            value = safe_scalar_curvature(*point)
            curvature_values.append(value)

        curvature_values = np.array(curvature_values)
        
        # Sprawdzamy czy mamy niezerowe wartości
        nonzero_values = curvature_values[curvature_values != 0]
        if len(nonzero_values) > 0:
            # 5. Usuwamy ekstremalne wartości tylko jeśli mamy niezerowe wartości
            percentile_5 = np.percentile(nonzero_values, 5)
            percentile_95 = np.percentile(nonzero_values, 95)
            curvature_values = np.clip(curvature_values, percentile_5, percentile_95)
        else:
            print("UWAGA: Wszystkie wartości krzywizny są zerowe!")

        result = {
            'points': points.tolist(),
            'values': curvature_values.tolist(),
            'ranges': modified_ranges
        }

        print("\nWygenerowano wynik pomyślnie!")
        print(f"Liczba punktów: {len(points)}")
        print(f"Liczba niezerowych wartości: {len(nonzero_values)}")
        if len(nonzero_values) > 0:
            print(f"Zakres wartości: [{np.min(nonzero_values)}, {np.max(nonzero_values)}]")
        
        return result

    except Exception as e:
        print(f"\nBŁĄD w generate_numerical_curvature: {e}")
        import traceback
        traceback.print_exc()
        return None
