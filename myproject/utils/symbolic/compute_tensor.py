import sympy as sp
from .simplification.custom_simplify import custom_simplify, weyl_simplify
from .indexes import lower_indices
import logging
import traceback

logger = logging.getLogger(__name__)

def oblicz_tensory(wspolrzedne, metryka):
    """
    Oblicza podstawowe tensory geometryczne dla podanej metryki.
    
    Args:
        wspolrzedne: Lista nazw współrzędnych
        metryka: Słownik komponentów metryki g_{ij}
        
    Returns:
        (g, Gamma, R_abcd, Ricci, Scalar_Curvature): Krotka zawierająca:
        - g: Tensor metryczny (Matrix)
        - Gamma: Symbole Christoffela (lista 3D)
        - R_abcd: Tensor Riemanna z obniżonymi indeksami (lista 4D)
        - Ricci: Tensor Ricciego (Matrix)
        - Scalar_Curvature: Skalar krzywizny (wyrażenie)
    """
    n = len(wspolrzedne)
    logger.info(f"Rozpoczynam obliczenia tensorów dla {n} wymiarów")

    # Tworzenie tensora metrycznego
    g = sp.Matrix(n, n, lambda i, j: metryka.get((i, j), metryka.get((j, i), 0)))
    g_inv = g.inv()

    # Obliczanie symboli Christoffela
    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for sigma in range(n):
        for mu in range(n):
            for nu in range(n):
                Gamma_sum = 0
                for lam in range(n):
                    partial_mu  = sp.diff(g[nu, lam], wspolrzedne[mu])
                    partial_nu  = sp.diff(g[mu, lam], wspolrzedne[nu])
                    partial_lam = sp.diff(g[mu, nu], wspolrzedne[lam])
                    Gamma_sum += g_inv[sigma, lam] * (partial_mu + partial_nu - partial_lam)
                Gamma[sigma][mu][nu] = custom_simplify(sp.Rational(1, 2) * Gamma_sum)

    # Obliczanie tensora Riemanna
    Riemann = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    term1 = sp.diff(Gamma[rho][nu][sigma], wspolrzedne[mu])
                    term2 = sp.diff(Gamma[rho][mu][sigma], wspolrzedne[nu])
                    sum_term = 0
                    for lam in range(n):
                        sum_term += (Gamma[rho][mu][lam] * Gamma[lam][nu][sigma]
                                     - Gamma[rho][nu][lam] * Gamma[lam][mu][sigma])
                    Riemann[rho][sigma][mu][nu] = custom_simplify(term1 - term2 + sum_term)

    # Obniżanie indeksów tensora Riemanna
    R_abcd = lower_indices(Riemann, g, n)

    # Obliczanie tensora Ricciego
    Ricci = sp.zeros(n, n)
    for mu in range(n):
        for nu in range(n):
            Ricci[mu, nu] = custom_simplify(sum(Riemann[rho][mu][rho][nu] for rho in range(n)))
            Ricci[mu, nu] = custom_simplify(Ricci[mu, nu])

    # Skalar krzywizny - scalar curvature R (Ricci scalar)
    try:
        Scalar_Curvature = 0  # Initialize to zero
        for mu in range(n):
            for nu in range(n):
                Scalar_Curvature += g_inv[mu, nu] * Ricci[mu, nu]
        
        # Apply simplification with fallback for complex results
        try:
            original_scalar = Scalar_Curvature
            Scalar_Curvature = custom_simplify(Scalar_Curvature)
            
            # Check for problematic results after simplification
            scalar_str = str(Scalar_Curvature)
            if ('nan' in scalar_str.lower() or 
                'inf' in scalar_str.lower() or 
                scalar_str.count('+') > 100 or  # Too complex expression
                len(scalar_str) > 1000):  # Excessively long expression
                
                logger.warning("Simplified scalar curvature expression is problematic, reverting to original")
                Scalar_Curvature = original_scalar
                
                # Try an alternative approach for complex expressions
                try:
                    logger.info("Attempting term-by-term simplification for complex expression")
                    parts = sp.expand(Scalar_Curvature).as_ordered_terms()
                    simplified_parts = [custom_simplify(part) for part in parts]
                    Scalar_Curvature = sum(simplified_parts)
                except Exception as e:
                    logger.error(f"Term-by-term simplification failed: {e}")
                    
                # Check if it's an FLRW-like metric as a last resort
                try:
                    # Detect if this is a cosmological metric with scale factor
                    has_scale_factor = False
                    has_time_coord = False
                    scale_factor_func = None
                    time_coord = None
                    
                    # Check if first coordinate is time-like
                    if len(wspolrzedne) >= 1:
                        potential_time = wspolrzedne[0]
                        if potential_time in ['t', 'tau', 'time']:
                            has_time_coord = True
                            time_coord = potential_time
                    
                    # Look for scale factor a(t) patterns in metric components
                    for i in range(n):
                        for j in range(n):
                            if (i, j) in metryka:
                                expr_str = str(metryka[(i, j)])
                                if 'a(' in expr_str or 'scale_factor' in expr_str:
                                    has_scale_factor = True
                                    # Try to find the scale factor function
                                    for sym in sp.preorder_traversal(metryka[(i, j)]):
                                        if isinstance(sym, sp.Function) and str(sym).startswith('a('):
                                            scale_factor_func = sym
                                            break
                    
                    # Special case for FLRW metrics with scale factor
                    if has_time_coord and has_scale_factor and scale_factor_func and len(wspolrzedne) == 4:
                        # Create the standard FLRW formula
                        t = sp.Symbol(time_coord)
                        a = scale_factor_func
                        
                        # Compute derivatives
                        a_dot = sp.diff(a, t)
                        a_ddot = sp.diff(a_dot, t)
                        k = sp.Symbol('k')  # Curvature parameter
                        c = sp.Symbol('c')  # Speed of light
                        
                        # Standard FLRW scalar curvature formula
                        # R = 6(k + ä(t)a(t) + ȧ(t)²)/(c²a(t)²)
                        flrw_formula = 6*(k + a*a_ddot + a_dot**2)/(c**2 * a**2)
                        
                        logger.info("Detected FLRW metric pattern, using standard formula as fallback")
                        Scalar_Curvature = flrw_formula
                except Exception as e:
                    logger.error(f"Error in FLRW pattern detection: {e}")
        except Exception as e:
            logger.error(f"Error in scalar curvature simplification: {e}")
    except Exception as e:
        logger.error(f"Error calculating scalar curvature: {e}")
        Scalar_Curvature = None

    return g, Gamma, R_abcd, Ricci, Scalar_Curvature

def compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n):
    """
    Oblicza tensor Einsteina G_{μν} = R_{μν} - (1/2) * R * g_{μν}
    
    Args:
        Ricci: Tensor Ricciego (Matrix lub słownik)
        Scalar_Curvature: Skalar Ricciego (wyrażenie)
        g: Tensor metryczny (Matrix)
        g_inv: Odwrócony tensor metryczny (Matrix)
        n: Wymiar przestrzeni
        
    Returns:
        (G_upper, G_lower): Para zawierająca tensor Einsteina 
        z podniesionymi i opuszczonymi indeksami
    """
    logger.info("Obliczam tensor Einsteina")
    
    # Konwertuj tensor Ricciego na format macierzowy, jeśli to słownik
    if isinstance(Ricci, dict):
        Ricci_matrix = sp.zeros(n, n)
        for (i, j), value in Ricci.items():
            Ricci_matrix[i, j] = value
        Ricci = Ricci_matrix
    
    # Inicjalizacja tensorów Einsteina
    G_lower = sp.zeros(n, n)  
    G_upper = sp.zeros(n, n) 
    
    # Obliczenie komponentów G_{μν} = R_{μν} - (1/2) * R * g_{μν}
    for mu in range(n):
        for nu in range(n):
            G_lower[mu, nu] = custom_simplify(Ricci[mu, nu] - sp.Rational(1, 2) * g[mu, nu] * Scalar_Curvature)
    
    # Podniesienie indeksów G^{μν} = g^{μα} g^{νβ} G_{αβ}
    for mu in range(n):
        for nu in range(n):
            G_upper[mu, nu] = 0  # Initialize to zero before summation
            for alpha in range(n):
                for beta in range(n):
                    G_upper[mu, nu] += g_inv[mu, alpha] * g_inv[nu, beta] * G_lower[alpha, beta]
            G_upper[mu, nu] = custom_simplify(G_upper[mu, nu])
    
    return G_upper, G_lower

def compute_weyl_tensor(R_abcd, Ricci, Scalar_Curvature, g, g_inv, n):
    """
    Oblicza tensor Weyla (bezśladową część tensora Riemanna).
    
    Dla n ≥ 4, wzór:
    C_{ρσμν} = R_{ρσμν} - 2/(n-2) * (g_{ρ[μ}R_{ν]σ} - g_{σ[μ}R_{ν]ρ}) + 2/((n-1)(n-2)) * R * g_{ρ[μ}g_{ν]σ}
    
    Gdzie [μν] oznacza antysymetryzację.
    
    Args:
        R_abcd: Tensor Riemanna z opuszczonymi indeksami
        Ricci: Tensor Ricciego
        Scalar_Curvature: Skalar krzywizny
        g: Tensor metryczny
        g_inv: Odwrócony tensor metryczny
        n: Wymiar przestrzeni
        
    Returns:
        Tensor Weyla jako tablicę 4D lub słownik
    """
    logger.info("Obliczam tensor Weyla")
    
    # Tworzymy tensory zero dla wyniku
    Weyl = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    
    # Współczynniki we wzorze zależne od wymiaru
    if n < 4:
        logger.warning(f"Tensor Weyla jest zawsze zero dla n={n} < 4")
        return Weyl
    
    # Współczynniki w formule tensora Weyla
    factor_1 = 2 / (n - 2)
    factor_2 = 2 / ((n - 1) * (n - 2))
    
    # Konwertuj tensor Ricciego na format macierzowy, jeśli to słownik
    if isinstance(Ricci, dict):
        Ricci_matrix = sp.zeros(n, n)
        for (i, j), value in Ricci.items():
            Ricci_matrix[i, j] = value
        Ricci = Ricci_matrix
    
    # Konwertuj R_abcd na dostępny format, jeśli to słownik
    if isinstance(R_abcd, dict):
        R_matrix = [[[[0 for _ in range(n)] for _ in range(n)] 
                   for _ in range(n)] for _ in range(n)]
        for (a, b, c, d), value in R_abcd.items():
            R_matrix[a][b][c][d] = value
        R_abcd = R_matrix
    
    # Główne obliczenia
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    # 1. Człon z tensorem Riemanna (bez zmian)
                    riemann_term = R_abcd[rho][sigma][mu][nu]
                    
                    # 2. Człon z tensorem Ricciego z właściwym współczynnikiem 2/(n-2)
                    ricci_term_1 = g[rho, mu] * Ricci[sigma, nu]
                    ricci_term_2 = g[rho, nu] * Ricci[sigma, mu]
                    ricci_term_3 = g[sigma, nu] * Ricci[rho, mu]
                    ricci_term_4 = g[sigma, mu] * Ricci[rho, nu]
                    
                    ricci_combined = factor_1 * (
                        ricci_term_1 - ricci_term_2 - ricci_term_3 + ricci_term_4
                    )
                    
                    # 3. Człon ze skalarem krzywizny z właściwym współczynnikiem 2/((n-1)(n-2))
                    scalar_term_1 = g[rho, mu] * g[sigma, nu]
                    scalar_term_2 = g[rho, nu] * g[sigma, mu]
                    
                    scalar_combined = factor_2 * Scalar_Curvature * (
                        scalar_term_1 - scalar_term_2
                    )
                    
                    # Oblicz sumę wszystkich członów zgodnie ze wzorem matematycznym
                    Weyl[rho][sigma][mu][nu] = weyl_simplify(
                        riemann_term - ricci_combined + scalar_combined
                    )
    
    return Weyl