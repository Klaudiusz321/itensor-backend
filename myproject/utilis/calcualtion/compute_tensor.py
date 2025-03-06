import sympy as sp
from .simplification.custom_simplify import custom_simplify, weyl_simplify
from .indexes import lower_indices
import logging

def oblicz_tensory(wspolrzedne, metryka):
    n = len(wspolrzedne)

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

    # Obliczanie skalarnej krzywizny
    Scalar_Curvature = custom_simplify(sum(g_inv[mu, nu] * Ricci[mu, nu] for mu in range(n) for nu in range(n)))
    Scalar_Curvature = custom_simplify(Scalar_Curvature)

    return g, Gamma, R_abcd, Ricci, Scalar_Curvature

def compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n):
    G_lower = sp.zeros(n, n)  
    G_upper = sp.zeros(n, n) 
    
    for mu in range(n):
        for nu in range(n):
            G_lower[mu, nu] = custom_simplify(Ricci[mu, nu] - sp.Rational(1, 2) * g[mu, nu] * Scalar_Curvature)

    for mu in range(n):
        for nu in range(n):
            sum_term = 0
            for alpha in range(n):
                sum_term += g_inv[mu, alpha] * G_lower[alpha, nu]
            G_upper[mu, nu] = custom_simplify(sum_term)

    return G_upper, G_lower

def compute_weyl_tensor(R_abcd, Ricci, Scalar_Curvature, g, n):
    """
    Oblicza tensor Weyla C_{rho,sigma,mu,nu} według wzoru:
    
    C_{ρσμν} = R_{ρσμν} - (2/(n-2)) * (g_{ρμ}R_{σν} - g_{ρν}R_{σμ} + g_{σν}R_{ρμ} - g_{σμ}R_{ρν}) 
                       + (2/((n-1)(n-2))) * R * (g_{ρμ}g_{σν} - g_{ρν}g_{σμ})
    
    Gdzie:
    - R_{ρσμν} to tensor Riemanna
    - R_{μν} to tensor Ricciego
    - R to skalar Ricciego
    - g_{μν} to tensor metryczny
    - n to wymiar przestrzeni
    
    Zwraca 4-wymiarową listę C[rho][sigma][mu][nu].
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Rozpoczynam obliczanie tensora Weyla, wymiar={n}")
    
    # Inicjalizacja tensora Weyla jako 4D tablicy
    C_abcd = [[[[0 for _ in range(n)] for _ in range(n)] 
                              for _ in range(n)] for _ in range(n)]
    
    # Dla n < 3 lub n = 3 tensor Weyla znika
    if n <= 3:
        logger.info(f"Wymiar przestrzeni n={n} <= 3, tensor Weyla jest zero.")
        return C_abcd  # same zera
    
    # Współczynniki z definicji
    factor_1 = 2 / (n - 2)  # Współczynnik przy członach z tensorem Ricciego
    factor_2 = 2 / ((n - 1) * (n - 2))  # Współczynnik przy członie ze skalarem krzywizny
    
    logger.info(f"Współczynniki: factor_1={factor_1}, factor_2={factor_2}")
    
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
                    # C = R - 2/(n-2)*(...) + 2/((n-1)(n-2))*R*(...)
                    C_abcd[rho][sigma][mu][nu] = riemann_term - ricci_combined + scalar_combined
                    
                    # Proste upraszczanie
                    if C_abcd[rho][sigma][mu][nu] != 0:
                        try:
                            # Podstawowe uproszczenie
                            simplified_val = custom_simplify(C_abcd[rho][sigma][mu][nu])
                            C_abcd[rho][sigma][mu][nu] = simplified_val
                        except Exception as e:
                            logger.warning(f"Błąd podczas upraszczania: {e}")
    
    return C_abcd