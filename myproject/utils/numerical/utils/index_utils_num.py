

class IndexUtilsNum:
    def __init__(self, n):
        self.n = n
       
        

    def generate_index_christoffel(self):
        return [
        (a,b,c)
            for a in range(self.n)
            for b in range(self.n)
            for c in range(b, self.n)
        ]

    def generate_index_riemann(self):
        return [
            (a,b,c,d)
            for a in range(self.n)
            for b in range(a, self.n)
            for c in range(self.n)
            for d in range(c, self.n)
            if (a*self.n + b) <= (c*self.n + d)
        ]

    def generate_index_ricci(self):
        return [(i,j) for i in range(self.n) for j in range(i, self.n)]

