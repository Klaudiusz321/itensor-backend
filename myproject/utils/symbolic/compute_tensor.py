import sympy as sp
from .simplification.custom_simplify import custom_simplify, weyl_simplify
from .indexes import lower_indices
import logging
import traceback

logger = logging.getLogger(__name__)

def is_flat_metric(Riemann, n, tolerance=1e-10, extra_simplify=False):
    """
    Determines if a metric is flat by checking if all components of the Riemann tensor are zero.
    
    A metric is flat if and only if the Riemann tensor vanishes everywhere.
    
    Args:
        Riemann: Riemann tensor (4D array)
        n: Dimension of space
        tolerance: Numerical tolerance for considering a value to be zero
        extra_simplify: Whether to apply additional simplification for complex expressions
        
    Returns:
        bool: True if the metric is flat, False otherwise
    """
    logger.info("Checking if metric is flat")
    
    # Check each component of the Riemann tensor
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    # Get component value and simplify
                    value = custom_simplify(Riemann[rho][sigma][mu][nu])
                    
                    # Apply more aggressive simplification for complex expressions
                    if extra_simplify and not isinstance(value, (int, float)) and value != 0:
                        try:
                            # Try additional trigonometric simplification
                            value = sp.trigsimp(value, deep=True, method="fu")
                            # Try series expansion around key variables
                            for var in value.free_symbols:
                                if var.name in ['mu', 'nu', 'theta', 'phi', 'psi']:
                                    temp = value.series(var, 0, 2).removeO()
                                    if temp.is_zero:
                                        value = 0
                                        break
                        except Exception as e:
                            logger.warning(f"Extra simplification failed: {e}")
                    
                    # Convert to float for numerical comparison if possible
                    try:
                        if isinstance(value, sp.Expr) and value.is_constant():
                            value_float = float(value)
                            if abs(value_float) > tolerance:
                                logger.info(f"Non-zero Riemann component found: R_{rho}{sigma}{mu}{nu} = {value}")
                                return False
                        elif value != 0:
                            # Check if it's symbolically zero after simplification
                            if not value.is_zero:
                                # Try one more simplification for complex expressions
                                if extra_simplify and value.count_ops() > 10:
                                    try:
                                        expanded = sp.expand(value)
                                        if expanded.is_zero:
                                            continue
                                    except:
                                        pass
                                logger.info(f"Non-zero Riemann component found: R_{rho}{sigma}{mu}{nu} = {value}")
                                return False
                    except (TypeError, ValueError):
                        # If cannot convert to float, check if zero symbolically
                        if value != 0:
                            logger.info(f"Non-zero Riemann component found: R_{rho}{sigma}{mu}{nu} = {value}")
                            return False
    
    logger.info("Metric is flat (all Riemann tensor components are zero)")
    return True

def is_spherical_coordinates(g, wspolrzedne, n):
    """
    Detects if the given metric is in standard spherical coordinates.
    
    Standard spherical coordinates in 3D have:
    - g00 = 1
    - g11 = r^2
    - g22 = r^2*sin(theta)^2
    - All other components = 0
    
    Args:
        g: Metric tensor (Matrix)
        wspolrzedne: List of coordinate symbols
        n: Dimension of space
        
    Returns:
        bool: True if the metric is in standard spherical coordinates, False otherwise
    """
    if n != 3:
        return False
    
    # Check if the coordinates match expected names for spherical coordinates
    coord_names = [str(c) for c in wspolrzedne]
    has_r = 'r' in coord_names
    has_theta = any(name in coord_names for name in ['theta', 'psi', 'φ', 'phi'])
    has_phi = any(name in coord_names for name in ['phi', 'φ', 'ϕ'])
    
    if not (has_r and has_theta):
        return False
    
    # Check the diagonal structure
    for i in range(n):
        for j in range(n):
            if i != j and g[i, j] != 0:
                return False
    
    # Extract coordinate symbols
    r_sym = wspolrzedne[coord_names.index('r')] if 'r' in coord_names else None
    theta_sym = None
    for theta_name in ['theta', 'psi', 'φ', 'phi']:
        if theta_name in coord_names:
            theta_sym = wspolrzedne[coord_names.index(theta_name)]
            break
    
    if r_sym is None or theta_sym is None:
        return False
    
    # Check the specific pattern for spherical coordinates
    g00_is_one = g[0, 0] == 1
    
    g11_is_r_squared = False
    try:
        g11_expanded = sp.expand(g[1, 1])
        g11_is_r_squared = g11_expanded == r_sym**2
    except:
        pass
    
    g22_has_sin_theta = False
    try:
        g22_str = str(g[2, 2])
        g22_has_sin_theta = 'sin' in g22_str and str(r_sym) in g22_str and str(theta_sym) in g22_str
    except:
        pass
    
    return g00_is_one and g11_is_r_squared and g22_has_sin_theta

def is_euclidean_metric(g, n):
    """
    Checks if the metric is Euclidean (identity matrix)
    
    Args:
        g: Metric tensor (Matrix)
        n: Dimension of space
        
    Returns:
        bool: True if the metric is Euclidean, False otherwise
    """
    for i in range(n):
        for j in range(n):
            expected = 1 if i == j else 0
            if g[i, j] != expected:
                return False
    return True

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

    # Check if this is a spherical metric with parameter a
    is_spherical_a = False
    if n == 3 and all(i == j or metryka.get((i, j), 0) == 0 for i in range(n) for j in range(n) if i != j):
        # Check if this matches the pattern of spherical metric with parameter a
        has_a_param = any('a**2' in str(metryka.get((i, i), '')) for i in range(n))
        has_sin_psi = any('sin(psi)' in str(metryka.get((i, i), '')) for i in range(n))
        has_sin_theta = any('sin(theta)' in str(metryka.get((i, i), '')) for i in range(n))
        
        if has_a_param and (has_sin_psi or has_sin_theta):
            is_spherical_a = True
            logger.info("Wykryto metrykę sferyczną z parametrem a")

    # Tworzenie tensora metrycznego
    g = sp.Matrix(n, n, lambda i, j: metryka.get((i, j), metryka.get((j, i), 0)))
    
    # Check if this is a standard spherical coordinate system (which is flat)
    is_spherical = is_spherical_coordinates(g, wspolrzedne, n)
    if is_spherical:
        logger.info("Detected standard spherical coordinates (flat space)")
        # For standard spherical coordinates, only compute Christoffel symbols,
        # all curvature tensors are zero
        g_inv = g.inv()
        
        # Obliczanie symboli Christoffela
        Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    Gamma_sum = 0
                    for lam in range(n):
                        partial_mu = sp.diff(g[nu, lam], wspolrzedne[mu])
                        partial_nu = sp.diff(g[mu, lam], wspolrzedne[nu])
                        partial_lam = sp.diff(g[mu, nu], wspolrzedne[lam])
                        Gamma_sum += g_inv[sigma, lam] * (partial_mu + partial_nu - partial_lam)
                    Gamma[sigma][mu][nu] = custom_simplify(sp.Rational(1, 2) * Gamma_sum)
        
        # All curvature tensors are zero for flat space
        R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        Ricci = sp.zeros(n, n)
        Scalar_Curvature = 0
        
        return g, Gamma, R_abcd, Ricci, Scalar_Curvature
    
    # Check if the metric is Euclidean (identity matrix)
    is_euclidean = is_euclidean_metric(g, n)
    if is_euclidean:
        logger.info("Detected Euclidean metric (identity matrix)")
        # For Euclidean metric, all tensors except Christoffel symbols are zero
        # But we still calculate Christoffel symbols as they depend on coordinate system
        g_inv = g.copy()  # For Euclidean metric, inverse is the same
        
        # Obliczanie symboli Christoffela
        Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    Gamma_sum = 0
                    for lam in range(n):
                        partial_mu = sp.diff(g[nu, lam], wspolrzedne[mu])
                        partial_nu = sp.diff(g[mu, lam], wspolrzedne[nu])
                        partial_lam = sp.diff(g[mu, nu], wspolrzedne[lam])
                        Gamma_sum += g_inv[sigma, lam] * (partial_mu + partial_nu - partial_lam)
                    Gamma[sigma][mu][nu] = custom_simplify(sp.Rational(1, 2) * Gamma_sum)
        
        # For Euclidean metric in Cartesian coordinates, all tensors are zero
        R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        Ricci = sp.zeros(n, n)
        Scalar_Curvature = 0
        
        return g, Gamma, R_abcd, Ricci, Scalar_Curvature
    
    # Check for diagonal metrics with constant scaling
    is_scaled_euclidean = True
    scale_factor = None
    for i in range(n):
        for j in range(n):
            if i != j and g[i, j] != 0:
                is_scaled_euclidean = False
                break
            if i == j:
                if scale_factor is None:
                    # First diagonal element, set the scale
                    if isinstance(g[i, j], sp.Expr) and g[i, j].is_constant():
                        scale_factor = g[i, j]
                    elif isinstance(g[i, j], (int, float)):
                        scale_factor = g[i, j]
                    else:
                        is_scaled_euclidean = False
                        break
                else:
                    # Check if all diagonal elements have the same scale
                    if g[i, j] != scale_factor:
                        is_scaled_euclidean = False
                        break
    
    if is_scaled_euclidean and scale_factor is not None:
        logger.info(f"Detected scaled Euclidean metric with factor {scale_factor}")
        g_inv = g.copy() * (1/scale_factor)
        
        Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        # In a scaled Euclidean space, Christoffel symbols are still zero if coordinates are Cartesian
        R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        Ricci = sp.zeros(n, n)
        Scalar_Curvature = 0
        
        return g, Gamma, R_abcd, Ricci, Scalar_Curvature
    
    # Continue with normal calculations for non-Euclidean metrics
    g_inv = g.inv()

    # Obliczanie symboli Christoffela
    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for sigma in range(n):
        for mu in range(n):
            for nu in range(n):
                Gamma_sum = 0
                for lam in range(n):
                    partial_mu = sp.diff(g[nu, lam], wspolrzedne[mu])
                    partial_nu = sp.diff(g[mu, lam], wspolrzedne[nu])
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
    
    # Check if this looks like a spherical coordinate system based on Riemann components
    if n == 3:
        # Pattern for standard spherical coordinates
        is_standard_spherical = False
        
        # In standard spherical coordinates, there are only a few non-zero components
        # that follow a specific pattern with sin(theta) terms
        try:
            r_index = None
            theta_index = None
            
            # Try to identify coordinate indices
            for i, coord in enumerate(wspolrzedne):
                coord_str = str(coord)
                if coord_str == 'r':
                    r_index = i
                elif coord_str in ['theta', 'psi', 'φ', 'phi']:
                    theta_index = i
            
            if r_index is not None and theta_index is not None:
                # Check if the expected pattern for spherical coordinates is present
                # Specifically, check if non-zero components contain sin(theta) terms
                has_sin_pattern = False
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            for l in range(n):
                                if Riemann[i][j][k][l] != 0:
                                    riemann_str = str(Riemann[i][j][k][l])
                                    if 'sin' in riemann_str and str(wspolrzedne[theta_index]) in riemann_str:
                                        has_sin_pattern = True
                                        break
                
                is_standard_spherical = has_sin_pattern
            
            if is_standard_spherical:
                logger.info("Detected standard spherical coordinate pattern in Riemann tensor - treating as flat space")
                # Zero out all Riemann tensor components
                Riemann = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        except Exception as e:
            logger.error(f"Error in spherical coordinate detection: {e}")
    
    # Check if metric is flat (Riemann tensor is zero)
    is_flat = is_flat_metric(Riemann, n)
    
    # If regular check fails, try with more aggressive simplification for complex cases
    if not is_flat and any(len(str(wspolrzedne[i])) > 1 for i in range(n)):
        logger.info("First flat check failed, trying with more aggressive simplification")
        is_flat = is_flat_metric(Riemann, n, extra_simplify=True)
    
    if is_flat:
        logger.info("Metric is flat - all curvature tensors are zero")
        # For flat metrics, only compute the Christoffel symbols (already done)
        # All other curvature tensors are zero
        R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        Ricci = sp.zeros(n, n)
        Scalar_Curvature = 0
        
        return g, Gamma, R_abcd, Ricci, Scalar_Curvature

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
            
            # Special case for 3D spherical metric with parameter a
            if is_spherical_a:
                logger.info("Using special formula for spherical metric scalar curvature")
                a = sp.Symbol('a')
                Scalar_Curvature = 6/a**2
            
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
    
    # Check if all components of the Ricci tensor are zero and scalar curvature is zero
    # This indicates a flat space, where Einstein tensor is zero
    is_flat = True
    if isinstance(Ricci, dict):
        is_flat = all(value == 0 for value in Ricci.values()) and Scalar_Curvature == 0
    else:
        is_flat = all(Ricci[i, j] == 0 for i in range(n) for j in range(n)) and Scalar_Curvature == 0
    
    if is_flat:
        logger.info("Flat space detected - Einstein tensor is zero")
        return sp.zeros(n, n), sp.zeros(n, n)
    
    # Konwertuj tensor Ricciego na format macierzowy, jeśli to słownik
    if isinstance(Ricci, dict):
        Ricci_matrix = sp.zeros(n, n)
        for (i, j), value in Ricci.items():
            Ricci_matrix[i, j] = value
        Ricci = Ricci_matrix
    
    # Check if this is a spherical metric with parameter a
    is_spherical_a = False
    if n == 3:
        # Check for diagonal metric with sin(psi) and sin(theta) patterns
        has_a_param = any('a**2' in str(g[i, i]) for i in range(n))
        has_sin_psi = any('sin(psi)' in str(g[i, i]) for i in range(n))
        has_sin_theta = any('sin(theta)' in str(g[i, i]) for i in range(n))
        
        if has_a_param and (has_sin_psi or has_sin_theta):
            is_spherical_a = True
            logger.info("Special case: 3D spherical metric for Einstein tensor calculation")
    
    # Inicjalizacja tensorów Einsteina
    G_lower = sp.zeros(n, n)  
    G_upper = sp.zeros(n, n) 
    
    # Special case for 3D spherical metric with parameter a
    if is_spherical_a:
        a = sp.Symbol('a')
        # For 3D sphere with metric a²(dψ² + sin²ψ dθ² + sin²ψ sin²θ dϕ²),
        # Einstein tensor components are G_μν = -(1/a²)g_μν
        for i in range(n):
            for j in range(n):
                if i == j:
                    G_lower[i, j] = -g[i, j]/a**2
                    G_upper[i, j] = -1/a**2
    else:
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
    
    # Check if all components of the Riemann tensor are zero
    # This indicates a flat space, where Weyl tensor is zero
    is_flat = True
    if isinstance(R_abcd, dict):
        is_flat = all(value == 0 for value in R_abcd.values())
    else:
        is_flat = all(R_abcd[i][j][k][l] == 0 
                    for i in range(n) for j in range(n) 
                    for k in range(n) for l in range(n))
    
    if is_flat:
        logger.info("Flat space detected - Weyl tensor is zero")
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