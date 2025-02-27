import sympy as sp
from ..simplification import custom_simplify
from .indexes import lower_indices
from functools import lru_cache
import itertools

@lru_cache(maxsize=32)
def cached_christoffel(i, j, k, g, g_inv, wspolrzedne):
    sum_term = 0
    for l in range(len(wspolrzedne)):
        partial_j = sp.diff(g[k, l], wspolrzedne[j])
        partial_k = sp.diff(g[j, l], wspolrzedne[k])
        partial_l = sp.diff(g[j, k], wspolrzedne[l])
        sum_term += g_inv[i, l] * (partial_j + partial_k - partial_l)
    return sp.simplify(sum_term / 2)

def oblicz_tensory(wspolrzedne, metryka):
    try:
        print("\nStarting tensor calculations...")
        n = len(wspolrzedne)
        print(f"Dimension: {n}")

        # Sprawdź czy mamy symbol czasu
        t = [s for s in wspolrzedne if str(s) == 't'][0] if any(str(s) == 't' for s in wspolrzedne) else sp.Symbol('t')
        
        # Inicjalizacja metryki jako macierzy
        g = sp.Matrix([[metryka.get((i,j), 0) for j in range(n)] for i in range(n)])
        
        # Sprawdź wyznacznik wcześnie
        if g.det() == 0:
            raise ValueError("Metric tensor is singular (determinant = 0)")
            
        g_inv = g.inv()

        # Oblicz symbole Christoffela z cache
        Gamma = [[[cached_christoffel(i,j,k, g, g_inv, tuple(wspolrzedne)) 
                  for k in range(n)] for j in range(n)] for i in range(n)]

        print("Christoffel symbols:")
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if Gamma[i][j][k] != 0:
                        print(f"Γ^{i}_{j}{k} = {Gamma[i][j][k]}")

        # Oblicz tensor Riemanna równolegle
        R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] 
                   for _ in range(n)] for _ in range(n)]
                   
        for rho, sigma, mu, nu in itertools.product(range(n), repeat=4):
            term1 = sp.diff(Gamma[rho][nu][sigma], wspolrzedne[mu])
            term2 = sp.diff(Gamma[rho][mu][sigma], wspolrzedne[nu])
            
            sum_term = sum(Gamma[rho][mu][lam] * Gamma[lam][nu][sigma] -
                         Gamma[rho][nu][lam] * Gamma[lam][mu][sigma]
                         for lam in range(n))
                         
            R_abcd[rho][sigma][mu][nu] = sp.simplify(term1 - term2 + sum_term)

        print("\nRiemann tensor components:")
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        if R_abcd[i][j][k][l] != 0:
                            print(f"R_{i}{j}{k}{l} = {R_abcd[i][j][k][l]}")

        # Oblicz tensor Ricciego
        Ricci = sp.zeros(n)
        for mu, nu in itertools.product(range(n), repeat=2):
            Ricci[mu, nu] = sum(R_abcd[rho][mu][rho][nu] for rho in range(n))
            Ricci[mu, nu] = sp.simplify(Ricci[mu, nu])

        print("\nRicci tensor components:")
        for i in range(n):
            for j in range(n):
                if Ricci[i,j] != 0:
                    print(f"R_{i}{j} = {Ricci[i,j]}")

        # Oblicz skalar krzywizny
        Scalar_Curvature = sum(g_inv[mu, nu] * Ricci[mu, nu] 
                              for mu, nu in itertools.product(range(n), repeat=2))
        Scalar_Curvature = sp.simplify(Scalar_Curvature)

        print("\nScalar curvature:", Scalar_Curvature)

        # Dla metryki Schwarzschilda powinniśmy otrzymać:
        # Γ^r_{tt} = (M*r - 2*M^2)/(r^3 - 2*M*r^2)
        # Γ^t_{tr} = M/(r^2 - 2*M*r)
        # Γ^r_{rr} = -M/(r^2 - 2*M*r)
        # Γ^θ_{rθ} = 1/r
        # Γ^r_{θθ} = -r + 2*M
        # Γ^φ_{rφ} = 1/r
        # Γ^r_{φφ} = (-r + 2*M)*sin^2(θ)
        # Γ^θ_{φφ} = -sin(θ)*cos(θ)
        # Γ^φ_{θφ} = cot(θ)

        return g, Gamma, R_abcd, Ricci, Scalar_Curvature

    except Exception as e:
        print(f"Error in oblicz_tensory: {e}")
        return None

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