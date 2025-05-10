
class Formatting:
    
    def __init__(self, tensor_dict, Gamma, Ricci, G, n):
        self.tensor_dict = tensor_dict
        self.Gamma = Gamma
        self.Ricci = Ricci
        self.G = G
        self.n = n


# Helper functions for formatting tensor results
    def format_tensor_components(self):
        """Format tensor components for JSON response."""
        result = {}
        for (i, j), value in self.tensor_dict.items():
            if i <= j:  # Only include upper triangular part for symmetric tensors
                result[f"{i}{j}"] = str(value)
        return result

    def format_christoffel_symbols(self):
        """Format Christoffel symbols for JSON response."""
        result = {}
        for a in range(self.n):
            for b in range(self.n):
                for c in range(self.n):
                    if self.Gamma[a][b][c] != 0:
                        result[f"{a}_{{{b}{c}}}"] = str(self.Gamma[a][b][c])
        return result

    def format_ricci_tensor(self):
        """Format Ricci tensor for JSON response."""
        result = {}
        for i in range(self.n):
            for j in range(i, self.n):  # Use symmetry
                if self.Ricci[i, j] != 0:
                    result[f"{i}{j}"] = str(self.Ricci[i, j])
        return result

    def format_einstein_tensor(self):
        """Format Einstein tensor for JSON response."""
        result = {}
        for i in range(self.n):
            for j in range(i, self.n):  # Use symmetry
                if self.G[i, j] != 0:
                    result[f"{i}{j}"] = str(self.G[i, j])
        return result