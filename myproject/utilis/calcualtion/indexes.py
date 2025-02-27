def generate_index_riemann(n):
    index = []
    for a in range(n):
        for b in range(a, n):
            for c in range(n):
                for d in range(c, n):
                    if (a * n + b) <= (c * n + d):
                        index.append((a, b, c, d))
    return index

def generate_index_ricci(n):
    index = []
    for i in range(n):
        for j in range(i, n):
            index.append((i, j))
    return index

def generate_index_christoffel(n):
    index = []
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                index.append((a, b, c))
    return index

def lower_indices(Riemann, g, n):
    R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    R_abcd[a][b][c][d] = sum(g[a, i] * Riemann[i][b][c][d] for i in range(n))
    return R_abcd

def raise_first_index_weyl(C, g_inv, n):
   
    C_raised = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    temp = 0
                    for lam in range(n):
                        temp += g_inv[rho, lam] * C[lam][sigma][mu][nu]
                    C_raised[rho][sigma][mu][nu] = (temp)
    return C_raised
