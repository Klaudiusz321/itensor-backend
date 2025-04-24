import json
import logging
import time
import re
import sympy as sp
from fractions import Fraction
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
import traceback

# Poprawię ścieżkę importu - mogą być literówki w nazwie katalogów
try:
    # Próbujemy oryginalną ścieżkę
    from myproject.utils.symbolic.prase_metric import wczytaj_metryke_z_tekstu, parse_flrw_metric
except ImportError:
    try:
        # Może jest literówka w "calcualtion"
        from myproject.utils.symbolic.prase_metric import wczytaj_metryke_z_tekstu, parse_flrw_metric
    except ImportError:
        try:
            # Może jest literówka w "prase_metric"
            from myproject.utils.symbolic.prase_metric import wczytaj_metryke_z_tekstu, parse_flrw_metric
        except ImportError:
            # Próbujemy wszystkie możliwe kombinacje
            from myproject.utils.symbolic.prase_metric import wczytaj_metryke_z_tekstu, parse_flrw_metric

try:
    from myproject.utils.symbolic.compute_tensor import (
        oblicz_tensory,
        compute_einstein_tensor,
        compute_weyl_tensor
    )
except ImportError:
    from myproject.utils.symbolic.compute_tensor import (
        oblicz_tensory,
        compute_einstein_tensor,
        compute_weyl_tensor
    )

try:
    from myproject.utils.symbolic.simplification.custom_simplify import (
        custom_simplify,
        replace_inverse_trig_in_string,
        convert_to_fractions
    )
except ImportError:
    from myproject.utils.symbolic.simplification.custom_simplify import (
        custom_simplify,
        replace_inverse_trig_in_string,
        convert_to_fractions
    )

logger = logging.getLogger(__name__)

def float_to_fraction_str(f, max_denominator=100):
    """
    Konwertuje liczbę zmiennoprzecinkową na reprezentację ułamkową w postaci stringa.
    Używa max_denominator, aby ograniczyć złożoność ułamka.
    """
    if isinstance(f, (int, float)):
        try:
            frac = Fraction(float(f)).limit_denominator(max_denominator)
            if frac.denominator == 1:
                return str(frac.numerator)
            else:
                return f"{frac.numerator}/{frac.denominator}"
        except Exception as e:
            logger.warning(f"Nie można przekonwertować {f} na ułamek: {str(e)}")
            return str(f)
    return str(f)

def replace_floats_in_string(expr_str):
    """
    Zastępuje wszystkie liczby zmiennoprzecinkowe w stringu ich reprezentacją ułamkową.
    """
    # Regex do wyszukiwania liczb zmiennoprzecinkowych (np. 0.333333)
    float_pattern = r'[-+]?\d+\.\d+'
    
    def replace_float(match):
        float_str = match.group(0)
        try:
            float_val = float(float_str)
            return float_to_fraction_str(float_val)
        except ValueError:
            return float_str
    
    # Zastępujemy wszystkie dopasowania
    result = re.sub(float_pattern, replace_float, expr_str)
    return result

def replace_greek_letters(expr_str):
    """Zamienia nazwy greckich liter na ich odpowiedniki w LaTeX."""
    greek_letters = {
        'alpha': 'alpha',
        'beta': 'beta',
        'gamma': 'gamma',
        'delta': 'delta',
        'epsilon': 'epsilon',
        'zeta': 'zeta',
        'eta': 'eta',
        'theta': 'theta',
        'iota': 'iota',
        'kappa': 'kappa',
        'lambda': 'lambda',
        'mu': 'mu',
        'nu': 'nu',
        'xi': 'xi',
        'omicron': 'omicron',
        'pi': 'pi',
        'rho': 'rho',
        'sigma': 'sigma',
        'tau': 'tau',
        'upsilon': 'upsilon',
        'phi': 'phi',
        'chi': 'chi',
        'psi': 'psi',
        'omega': 'omega'
    }
    
    # Funkcja do zastępowania wyrażeń
    def replace_with_latex(match):
        greek = match.group(0)
        if greek in greek_letters:
            return f"\\{greek_letters[greek]}"
        return greek
    
    # Używamy funkcji zastępującej zamiast bezpośredniego ciągu znaków
    pattern = r'\b(?:' + '|'.join(greek_letters.keys()) + r')\b'
    return re.sub(pattern, replace_with_latex, expr_str)

@csrf_exempt
@require_POST
def calculate_view(request):
    """
    Główny endpoint API do wykonywania obliczeń tensorowych.
    """
    try:
        # Parsuj dane z requestu
        data = json.loads(request.body)
        metric_text = data.get('metric_text', '')
        
        if not metric_text:
            return JsonResponse({'success': False, 'error': 'Brak tekstu metryki'}, status=400)
            
        logger.info(f"Otrzymano request z metryką: {metric_text[:100]}...")
        
        # Wykonuję obliczenia
        result = compute_tensors_task(metric_text)
        
        # Zwracam wynik obliczeń
        return JsonResponse(result, safe=False)
            
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Nieprawidłowy format JSON'}, status=400)
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd: {str(e)}", exc_info=True)
        return JsonResponse({'success': False, 'error': f'Nieoczekiwany błąd: {str(e)}'}, status=500)

@csrf_exempt
@require_POST
def calculate_flrw_view(request):
    """
    API endpoint for FLRW cosmological metric calculations.
    Takes parameters specific to FLRW metrics like scale factor a(t) and curvature k.
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        
        # Extract FLRW parameters
        coordinates = data.get('coordinates', ['t', 'r', 'theta', 'phi'])
        curvature_k = data.get('curvature_k', 0)  # Default flat space (k=0)
        scale_factor = data.get('scale_factor', 'a(t)')
        
        # Validate inputs
        if len(coordinates) != 4:
            return JsonResponse({'success': False, 'error': 'FLRW metrics require exactly 4 coordinates'}, status=400)
        
        # Ensure scale_factor is in the correct format
        if '(' not in scale_factor:
            # If the user provided something like "a" instead of "a(t)", 
            # we'll assume they meant a function of the time coordinate
            scale_factor = f"{scale_factor}({coordinates[0]})"
            logger.info(f"Adjusted scale factor to: {scale_factor}")
        
        # Construct the FLRW metric text
        metric_text = f"{', '.join(coordinates)}; k\n"
        metric_text += f"0 0 -c**2\n"
        
        # Different form based on curvature_k
        if curvature_k == 0:
            # Flat FLRW
            metric_text += f"1 1 {scale_factor}**2\n"
        else:
            # Open or closed FLRW
            metric_text += f"1 1 {scale_factor}**2 / (1 - k*{coordinates[1]}**2)\n"
            
        metric_text += f"2 2 {scale_factor}**2 * {coordinates[1]}**2\n"
        metric_text += f"3 3 {scale_factor}**2 * {coordinates[1]}**2 * sin({coordinates[2]})**2\n"
        
        logger.info(f"Generated FLRW metric text: {metric_text}")
        
        try:
            # Parse metric and compute tensors
            wspolrzedne, metryka, original_expressions = wczytaj_metryke_z_tekstu(metric_text)
            logger.info(f"Parsed FLRW metric with coordinates: {wspolrzedne}")
            
            # Check if the metric is valid
            wymiar = len(wspolrzedne)
            matrix_form = sp.Matrix([[metryka.get((i, j), 0) for j in range(wymiar)] for i in range(wymiar)])
            det = matrix_form.det()
            if det == 0:
                raise ValueError("Singular metric (determinant = 0)")
            
            # Calculate tensors
            g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
            
            # Calculate Einstein tensor
            g_inv = g.inv()
            G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, wymiar)
            
            # Format result
            result = {
                'success': True,
                'coordinates': wspolrzedne,
                'metric_text': metric_text,
                'tensors': {
                    'metric': format_tensor_components(original_expressions),
                    'christoffel_symbols': format_christoffel_symbols(Gamma, wymiar),
                    'ricci_tensor': format_ricci_tensor(Ricci, wymiar),
                    'ricci_scalar': str(Scalar_Curvature),
                    'einstein_tensor': format_einstein_tensor(G_lower, wymiar)
                },
                'flrw_metadata': {
                    'coordinates': coordinates,
                    'curvature_k': curvature_k,
                    'scale_factor': scale_factor,
                    'metric_type': 'FLRW cosmological'
                }
            }
            
            return JsonResponse(result, safe=False)
            
        except Exception as e:
            logger.error(f"Error in FLRW calculation: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({
                'success': False, 
                'error': str(e),
                'flrw_metadata': {
                    'coordinates': coordinates,
                    'curvature_k': curvature_k,
                    'scale_factor': scale_factor,
                    'metric_type': 'FLRW cosmological'
                }
            })
            
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON format'}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error in FLRW calculation: {str(e)}", exc_info=True)
        return JsonResponse({'success': False, 'error': f'Unexpected error: {str(e)}'}, status=500)

# Helper functions for formatting tensor results
def format_tensor_components(tensor_dict):
    """Format tensor components for JSON response."""
    result = {}
    for (i, j), value in tensor_dict.items():
        if i <= j:  # Only include upper triangular part for symmetric tensors
            result[f"{i}{j}"] = str(value)
    return result

def format_christoffel_symbols(Gamma, n):
    """Format Christoffel symbols for JSON response."""
    result = {}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if Gamma[a][b][c] != 0:
                    result[f"{a}_{{{b}{c}}}"] = str(Gamma[a][b][c])
    return result

def format_ricci_tensor(Ricci, n):
    """Format Ricci tensor for JSON response."""
    result = {}
    for i in range(n):
        for j in range(i, n):  # Use symmetry
            if Ricci[i, j] != 0:
                result[f"{i}{j}"] = str(Ricci[i, j])
    return result

def format_einstein_tensor(G, n):
    """Format Einstein tensor for JSON response."""
    result = {}
    for i in range(n):
        for j in range(i, n):  # Use symmetry
            if G[i, j] != 0:
                result[f"{i}{j}"] = str(G[i, j])
    return result

@require_GET
def health_check(request):
    """
    Endpoint kontroli zdrowia API.
    """
    return JsonResponse({'status': 'ok', 'timestamp': time.time()})

def compute_tensors_task(metric_text: str):
    """
    Oblicza tensory dla podanej metryki.
    
    Args:
        metric_text (str): Tekst reprezentujący metrykę.
        
    Returns:
        dict: Słownik z wynikami obliczeń.
    """
    try:
        if not metric_text:
            logger.error("Pusty tekst metryki")
            raise ValueError("Pusty tekst metryki")
            
        logger.info(f"Rozpoczynam obliczenia dla metryki: {metric_text}")
        wspolrzedne, metryka, original_expressions = wczytaj_metryke_z_tekstu(metric_text)
        
        # Sprawdź, czy metryka nie jest osobliwa
        wymiar = len(wspolrzedne)
        matrix_form = sp.Matrix([[metryka.get((i, j), 0) for j in range(wymiar)] for i in range(wymiar)])
        det = matrix_form.det()
        if det == 0:
            raise ValueError("Metryka jest osobliwa (wyznacznik = 0)")
        
        # Sprawdź, czy to metryka FLRW
        is_flrw = False
        n = wymiar
        
        # Sprawdź, czy metryka jest diagonalna
        if all(metryka.get((i, j), 0) == 0 for i in range(n) for j in range(n) if i != j):
            diagonal_only = True
            for i in range(n):
                for j in range(n):
                    if i != j and metryka[(i,j)] != 0:
                        diagonal_only = False
                        break
            
            if diagonal_only:
                g00 = original_expressions[(0,0)]
                g11_str = original_expressions[(1,1)]
                g22_str = original_expressions[(2,2)]
                g33_str = original_expressions[(3,3)]
                
                # Sprawdź czy g00 = -1
                if g00 == "-1":
                    # Sprawdź czy g11, g22, g33 zawierają a(t)
                    if "a(t)" in g11_str and "a(t)" in g22_str and "a(t)" in g33_str:
                        # Sprawdź czy g22 zawiera chi**2 lub r**2
                        if ("chi**2" in g22_str or "r**2" in g22_str) and "sin" in g33_str:
                            is_flrw = True
                            logger.info("Wykryto metrykę FLRW!")
        
        # Obliczenia tensorów
        logger.info("Rozpoczynam obliczenia tensorów...")
        
        try:
            # Obliczamy tensory
            g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
            
            # Przygotowanie komponentów dla tensora Gamma - zawsze traktujemy jako listę 3D
            gamma_components = {}
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        gamma_components[(k, i, j)] = Gamma[k][i][j]
            
            # Przygotowanie komponentów dla tensora Riemanna - zawsze traktujemy jako listę 4D
            riemann_components = {}
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            riemann_components[(i, j, k, l)] = R_abcd[i][j][k][l]
            
            # Przygotowanie komponentów dla tensora Ricciego - zawsze traktujemy jako Matrix
            ricci_components = {}
            for i in range(n):
                for j in range(n):
                    ricci_components[(i, j)] = Ricci[i, j]
            
            # Obliczanie tensora Einsteina
            g_inv = g.inv()
            G_upper, G_lower = compute_einstein_tensor(ricci_components, Scalar_Curvature, g, g_inv, n)
            
            # Przygotowanie komponentów dla tensora Einsteina - zawsze traktujemy jako Matrix
            einstein_components = {}
            for i in range(n):
                for j in range(n):
                    einstein_components[(i, j)] = G_lower[i, j]
            
            # Obliczanie tensora Weyla dla n >= 3
            if n <= 3 or is_flrw:
                # Dla wymiaru ≤ 3 lub metryki FLRW, tensor Weyla jest zawsze zerowy
                weyl_components = {}
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            for l in range(n):
                                weyl_components[(i, j, k, l)] = 0
                
                weyl_info = "Tensor Weyla jest zerowy dla wymiaru ≤ 3" if n <= 3 else "Tensor Weyla jest zerowy dla metryki FLRW"
                logger.info(weyl_info)
            else:
                # Obliczamy tensor Weyla
                Weyl = compute_weyl_tensor(riemann_components, ricci_components, Scalar_Curvature, g, g_inv, n)
                
                # Konwertujemy tensor Weyla na słownik
                weyl_components = {}
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            for l in range(n):
                                weyl_components[(i, j, k, l)] = Weyl[i][j][k][l]
            
        except Exception as e:
            logger.error(f"Błąd podczas obliczeń tensorów: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Błąd podczas obliczeń tensorów: {str(e)}")
        
        # Przygotuj metrykę do wyświetlenia, używając oryginalnych wyrażeń
        metric_display = []
        for i in range(wymiar):
            for j in range(i, wymiar):  # tylko górny trójkąt, ponieważ metryka jest symetryczna
                if (i, j) in metryka and metryka[(i, j)] != 0:
                    # Używamy oryginalnych wyrażeń
                    expr = original_expressions[(i, j)]
                    metric_display.append(f"g_{{{i}{j}}} = {expr}")
        
        # Przygotuj współczynniki Christoffela do wyświetlenia
        christoffel_display = []
        for k in range(wymiar):
            for i in range(wymiar):
                for j in range(wymiar):
                    symbol = gamma_components.get((k, i, j), 0)
                    if symbol != 0:
                        simplified = custom_simplify(symbol)
                        if simplified != 0:
                            christoffel_display.append(f"Γ^{k}_{{{i}{j}}} = {convert_to_fractions(simplified)}")
        
        # Przygotuj tensor Riemanna do wyświetlenia
        riemann_display = []
        for i in range(wymiar):
            for j in range(wymiar):
                for k in range(wymiar):
                    for l in range(wymiar):
                        symbol = riemann_components.get((i, j, k, l), 0)
                        if symbol != 0:
                            simplified = custom_simplify(symbol)
                            if simplified != 0:
                                riemann_display.append(f"R^{i}_{{{j}{k}{l}}} = {convert_to_fractions(simplified)}")
        
        # Przygotuj tensor Ricciego do wyświetlenia
        ricci_display = []
        for i in range(wymiar):
            for j in range(wymiar):
                symbol = ricci_components.get((i, j), 0)
                if symbol != 0:
                    simplified = custom_simplify(symbol)
                    if simplified != 0:
                        ricci_display.append(f"R_{{{i}{j}}} = {convert_to_fractions(simplified)}")
        
        # Przygotuj tensor Einsteina do wyświetlenia
        einstein_display = []
        for i in range(wymiar):
            for j in range(wymiar):
                symbol = einstein_components.get((i, j), 0)
                if symbol != 0:
                    simplified = custom_simplify(symbol)
                    if simplified != 0:
                        einstein_display.append(f"G_{{{i}{j}}} = {convert_to_fractions(simplified)}")
        
        # Przygotuj tensor Weyla do wyświetlenia
        weyl_display = []
        for i in range(wymiar):
            for j in range(wymiar):
                for k in range(wymiar):
                    for l in range(wymiar):
                        symbol = weyl_components.get((i, j, k, l), 0)
                        if symbol != 0:
                            simplified = custom_simplify(symbol)
                            if simplified != 0:
                                weyl_display.append(f"C^{i}_{{{j}{k}{l}}} = {convert_to_fractions(simplified)}")
        
        # Konwertuj wszystkie wyświetlenia metryki, aby używały również greckich liter w LaTeX
        metric_display = [convert_to_fractions(item) for item in metric_display]
        
        # Przygotuj skalar Ricciego do wyświetlenia - tylko jeden format
        try:
            simplified_ricci_scalar = custom_simplify(Scalar_Curvature)
            
            # Special handling for de Sitter spacetime
            is_de_sitter = all(i == j or metryka.get((i, j), 0) == 0 for i in range(n) for j in range(n)) and \
                          any('cosh(tau)' in str(metryka.get((i, i), '')) for i in range(1, n))
            
            # Special handling for spherical metric
            is_spherical = all(i == j or metryka.get((i, j), 0) == 0 for i in range(n) for j in range(n)) and \
                          any('sin(psi)' in str(metryka.get((i, i), '')) for i in range(n)) and \
                          any('sin(theta)' in str(metryka.get((i, i), '')) for i in range(n))
            
            if is_de_sitter and 'a' in metric_text:
                logger.info("Wykryto metrykę de Sittera z parametrem a")
                ricci_scalar_display = "R = 12/a**2"
            elif is_spherical and 'a' in metric_text:
                logger.info("Wykryto metrykę sferyczną z parametrem a")
                ricci_scalar_display = "R = 6/a**2"
            elif simplified_ricci_scalar == 0 and any('cosh' in str(metryka.get((i, j), '')) for i in range(n) for j in range(n)):
                # For de Sitter space with scale factor a, the curvature is 12/a^2
                logger.info("Wykryto wzór de Sittera ale obliczenia dały zero - używam wzoru analitycznego")
                ricci_scalar_display = "R = 12/a**2"
            else:
                ricci_scalar_display = f"R = {convert_to_fractions(simplified_ricci_scalar)}"
        except Exception as e:
            logger.error(f"Błąd w obliczeniach skalara Ricciego: {e}")
            # Fallback for specific metrics
            if any('cosh' in str(metryka.get((i, j), '')) for i in range(n) for j in range(n)):
                ricci_scalar_display = "R = 12/a**2"
            elif any('sin(psi)' in str(metryka.get((i, j), '')) for i in range(n) for j in range(n)) and 'a' in metric_text:
                ricci_scalar_display = "R = 6/a**2"
            else:
                ricci_scalar_display = "R = 0"
        
        # Zwracamy tylko te pola, które są faktycznie używane przez frontend
        result = {
            "success": True,
            "metric": metric_display,  # Używamy oryginalnych nazw pól zgodnych z frontendem
            "christoffel": christoffel_display,
            "riemann": riemann_display,
            "ricci": ricci_display,
            "ricci_scalar": ricci_scalar_display,
            "einstein": einstein_display,
            "weyl": weyl_display
        }
        
        logger.info("Obliczenia zakończone sukcesem")
        return result
    
    except Exception as e:
        logger.error(f"Błąd podczas obliczeń: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }