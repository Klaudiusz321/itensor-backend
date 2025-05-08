

class Indexes:
    def __init__(self, n):
        self.n = n

    def generate_index_christoffel(self):
        return [(a, b, c) for a in range(self.n) for b in range(self.n) for c in range(b, self.n)]

    def generate_index_riemann(self):
        index = [
            (a, b, c, d)
            for a in range(self.n)
            for b in range(a, self.n)
            for c in range(self.n)
            for d in range(c, self.n)
            if (a* self.n + b) <= (c* self.n + d)
        ]
        return index
    def generate_index_ricci(self):
        return [(i,j) for i in range(self.n) for j in range(self.n)]
    
    def lower_indices(self, tensor, g):
        lowered = [[[[0 for _ in range(self.n)] for _ in range(self.n)] 
                    for _ in range(self.n)] for _ in range(self.n)]
        
        for a in range(self.n):
            for b in range(self.n):
                for c in range(self.n):
                    for d in range(self.n):
                        lowered[a][b][c][d] = sum(g[a, i] * tensor[i][b][c][d] for i in range(self.n))
        
        return lowered
    def raise_indices(self, tensor, g_inv):
        raised = [[[[0 for _ in range(self.n)] for _ in range(self.n)] 
                   for _ in range(self.n)] for _ in range(self.n)]
        
        for rho in range(self.n):
            for sigma in range(self.n):
                for mu in range(self.n):
                    for nu in range(self.n):
                        raised[rho][sigma][mu][nu] = sum(
                            g_inv[rho, lam] * tensor[lam][sigma][mu][nu] for lam in range(self.n)
                        )
        
        return raised
    