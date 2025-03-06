import json
import logging
import time
import re
import sympy as sp
from fractions import Fraction
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
import traceback  # Dodaję brakujący import

# Usuwam import AsyncResult z Celery
# from celery.result import AsyncResult

# Poprawię ścieżkę importu - mogą być literówki w nazwie katalogów
try:
    # Próbujemy oryginalną ścieżkę
    from myproject.utilis.calcualtion.prase_metric import wczytaj_metryke_z_tekstu
except ImportError:
    try:
        # Może jest literówka w "calcualtion"
        from myproject.utilis.calculation.prase_metric import wczytaj_metryke_z_tekstu
    except ImportError:
        try:
            # Może jest literówka w "prase_metric"
            from myproject.utilis.calcualtion.parse_metric import wczytaj_metryke_z_tekstu
        except ImportError:
            # Próbujemy wszystkie możliwe kombinacje
            from myproject.utilis.calculation.parse_metric import wczytaj_metryke_z_tekstu

from myproject.utilis.calcualtion.compute_tensor import (
    oblicz_tensory,
    compute_einstein_tensor,
    compute_weyl_tensor
)

from myproject.utilis.calcualtion.simplification.custom_simplify import (
    weyl_simplify, 
    custom_simplify,
    replace_inverse_trig_in_string
)

# Import funkcji obliczeniowej z tasks.py

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

def convert_to_fractions(expr):
    """Konwertuje liczby zmiennoprzecinkowe na ułamki, zastępuje odwrotne funkcje trygonometryczne
    i zamienia nazwy greckich liter na odpowiedniki LaTeX.
    
    Obsługuje zarówno ciągi znaków jak i obiekty SymPy.
    """
    # Jeśli to obiekt SymPy, najpierw konwertujemy na string używając LaTeX
    if not isinstance(expr, str):
        expr_str = sp.latex(expr)
    else:
        expr_str = expr
        
    # Teraz wykonujemy wszystkie transformacje na stringu
    # Najpierw zamieniamy liczby zmiennoprzecinkowe na ułamki
    result = replace_floats_in_string(expr_str)
    # Następnie zamieniamy odwrotne funkcje trygonometryczne
    result = replace_inverse_trig_in_string(result)
    # Na końcu zamieniamy nazwy greckich liter na symbole LaTeX
    result = replace_greek_letters(result)
    return result

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
            return JsonResponse({'error': 'Brak tekstu metryki'}, status=400)
            
        logger.info(f"Otrzymano request z metryką: {metric_text[:100]}...")
        
        # Wykonuję obliczenia
        result = compute_tensors_task(metric_text)
        
        # Zwracam wynik obliczeń
        return JsonResponse(result, safe=False)
            
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Nieprawidłowy format JSON'}, status=400)
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd: {str(e)}", exc_info=True)
        return JsonResponse({'error': f'Nieoczekiwany błąd: {str(e)}'}, status=500)

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
            
            # Sprawdzamy, czy otrzymaliśmy poprawne typy danych
            if not isinstance(Ricci, (dict, list, sp.Matrix)):
                logger.warning(f"Tensor Ricciego nie jest indeksowalny. Typ: {type(Ricci)}")
                # Jeśli tensor Ricciego nie jest indeksowalny, traktujemy go jako pojedynczą wartość
                ricci_components = {(0, 0): Ricci}
            elif isinstance(Ricci, sp.Matrix):
                # Jeśli Ricci jest obiektem Matrix, konwertujemy go na słownik
                ricci_components = {}
                for i in range(Ricci.rows):
                    for j in range(Ricci.cols):
                        ricci_components[(i, j)] = Ricci[i, j]
            elif isinstance(Ricci, dict):
                # Jeśli Ricci jest słownikiem, używamy go bezpośrednio
                ricci_components = Ricci
            else:
                # Jeśli Ricci jest listą, konwertujemy go na słownik
                ricci_components = {}
                for i in range(len(Ricci)):
                    for j in range(len(Ricci[i])):
                        ricci_components[(i, j)] = Ricci[i][j]
            
            # Analogicznie dla innych tensorów
            # Przygotowanie komponentów dla tensora Gamma
            if not isinstance(Gamma, (dict, list)):
                gamma_components = {}
            elif isinstance(Gamma, dict):
                gamma_components = Gamma
            else:
                gamma_components = {}
                for k in range(len(Gamma)):
                    for i in range(len(Gamma[k])):
                        for j in range(len(Gamma[k][i])):
                            gamma_components[(k, i, j)] = Gamma[k][i][j]
            
            # Przygotowanie komponentów dla tensora Riemanna
            if not isinstance(R_abcd, (dict, list)):
                riemann_components = {}
            elif isinstance(R_abcd, dict):
                riemann_components = R_abcd
            else:
                riemann_components = {}
                for i in range(len(R_abcd)):
                    for j in range(len(R_abcd[i])):
                        for k in range(len(R_abcd[i][j])):
                            for l in range(len(R_abcd[i][j][k])):
                                riemann_components[(i, j, k, l)] = R_abcd[i][j][k][l]
            
            # Obliczanie tensora Einsteina
            g_inv = g.inv()
            G_upper, G_lower = compute_einstein_tensor(ricci_components, Scalar_Curvature, g, g_inv, n)
            
            # Przygotowanie komponentów dla tensora Einsteina
            if isinstance(G_lower, dict):
                einstein_components = G_lower
            elif isinstance(G_lower, list):
                einstein_components = {}
                for i in range(len(G_lower)):
                    for j in range(len(G_lower[i])):
                        einstein_components[(i, j)] = G_lower[i][j]
            else:
                einstein_components = {}
            
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
                # Tutaj zakładamy, że compute_weyl_tensor zwraca listę lub słownik
                Weyl = compute_weyl_tensor(riemann_components, ricci_components, Scalar_Curvature, g, n)
                
                if isinstance(Weyl, dict):
                    weyl_components = Weyl
                else:
                    weyl_components = {}
                    for i in range(len(Weyl)):
                        for j in range(len(Weyl[i])):
                            for k in range(len(Weyl[i][j])):
                                for l in range(len(Weyl[i][j][k])):
                                    weyl_components[(i, j, k, l)] = Weyl[i][j][k][l]
            
        except Exception as e:
            logger.error(f"Błąd podczas obliczeń tensorów: {str(e)}")
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
        simplified_ricci_scalar = custom_simplify(Scalar_Curvature)
        ricci_scalar_display = f"R = {convert_to_fractions(simplified_ricci_scalar)}"
        
        # Zwracamy tylko te pola, które są faktycznie używane przez frontend
        result = {
            "success": True,
            "metric_display": metric_display,
            "christoffel_display": christoffel_display,
            "riemann_display": riemann_display,
            "ricci_display": ricci_display,
            "ricci_scalar_display": ricci_scalar_display,  # Tylko jedno pole dla skalara krzywizny
            "einstein_display": einstein_display,
            "weyl_display": weyl_display
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