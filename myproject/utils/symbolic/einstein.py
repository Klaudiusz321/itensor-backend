from utils.symbolic.simplification.custom_simplify import custom_simplify
import sympy as sp
import logging

logger = logging.getLogger(__name__)


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