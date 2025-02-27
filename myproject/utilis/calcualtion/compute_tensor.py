import sympy as sp
from ..simplification import custom_simplify
from .indexes import lower_indices

def oblicz_tensory(wspolrzedne, metryka):
    try:
        print("\nStarting tensor calculations...")
        n = len(wspolrzedne)
        print(f"Dimension: {n}")

        # Sprawdź czy mamy symbol czasu
        t = [s for s in wspolrzedne if str(s) == 't'][0] if any(str(s) == 't' for s in wspolrzedne) else sp.Symbol('t')
        
        # Inicjalizacja metryki jako macierzy zerowej
        g = sp.zeros(n, n)
        
        # Wypełnij metrykę
        for i in range(n):
            for j in range(n):
                if (i, j) in metryka:
                    g[i, j] = metryka[(i, j)]
                elif (j, i) in metryka:
                    g[i, j] = metryka[(j, i)]
                else:
                    # Dla metryki diagonalnej, zakładamy 0 dla elementów pozadiagonalnych
                    g[i, j] = 0 if i != j else 1

        print("Metric tensor:")
        print(g)

        # Sprawdź czy macierz jest odwracalna
        if g.det() == 0:
            raise ValueError("Metric tensor is singular (determinant = 0)")

        g_inv = g.inv()
        print("Inverse metric tensor:")
        print(g_inv)

        # Oblicz symbole Christoffela
        Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    sum_term = 0
                    for lam in range(n):
                        partial_mu = sp.diff(g[nu, lam], wspolrzedne[mu])
                        partial_nu = sp.diff(g[mu, lam], wspolrzedne[nu])
                        partial_lam = sp.diff(g[mu, nu], wspolrzedne[lam])
                        sum_term += g_inv[sigma, lam] * (partial_mu + partial_nu - partial_lam)
                    Gamma[sigma][mu][nu] = custom_simplify(sp.Rational(1, 2) * sum_term)

        print("Christoffel symbols:")
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if Gamma[i][j][k] != 0:
                        print(f"Γ^{i}_{j}{k} = {Gamma[i][j][k]}")

        # Oblicz tensor Riemanna
        R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        term1 = custom_simplify(sp.diff(Gamma[rho][nu][sigma], wspolrzedne[mu]))
                        term2 = custom_simplify(sp.diff(Gamma[rho][mu][sigma], wspolrzedne[nu]))
                        sum_term = 0
                        for lam in range(n):
                            prod1 = custom_simplify(Gamma[rho][mu][lam] * Gamma[lam][nu][sigma])
                            prod2 = custom_simplify(Gamma[rho][nu][lam] * Gamma[lam][mu][sigma])
                            sum_term += (prod1 - prod2)
                        R_abcd[rho][sigma][mu][nu] = custom_simplify(term1 - term2 + sum_term)

        print("\nRiemann tensor components:")
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        if R_abcd[i][j][k][l] != 0:
                            print(f"R_{i}{j}{k}{l} = {R_abcd[i][j][k][l]}")

        # Oblicz tensor Ricciego
        Ricci = sp.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                sum_term = 0
                for rho in range(n):
                    sum_term += R_abcd[rho][mu][rho][nu]
                Ricci[mu, nu] = custom_simplify(sum_term)

        print("\nRicci tensor components:")
        for i in range(n):
            for j in range(n):
                if Ricci[i,j] != 0:
                    print(f"R_{i}{j} = {Ricci[i,j]}")

        # Oblicz skalar krzywizny
        Scalar_Curvature = 0
        for mu in range(n):
            for nu in range(n):
                term = custom_simplify(g_inv[mu, nu] * Ricci[mu, nu])
                Scalar_Curvature += term
        Scalar_Curvature = custom_simplify(Scalar_Curvature)

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