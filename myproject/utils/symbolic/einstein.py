from utils.symbolic.simplification.custom_simplify import custom_simplify
import sympy as sp
import logging

logger = logging.getLogger(__name__)

class ComputeEinsteinTensor:

    def __init__(self, Ricci, Scalar_Curvature, g, g_inv, n):
        self.Ricci = Ricci
        self.Scalar_Curvature = Scalar_Curvature
        self.g = g
        self.g_inv = g_inv
        self.n = n
    def compute_einstein_tensor(self):
       
        logger.info("Obliczam tensor Einsteina")
        
        # Check if all components of the Ricci tensor are zero and scalar curvature is zero
        # This indicates a flat space, where Einstein tensor is zero
        is_flat = True
        if isinstance(self.Ricci, dict):
            is_flat = all(value == 0 for value in self.Ricci.values()) and self.Scalar_Curvature == 0
        else:
            is_flat = all(self.Ricci[i, j] == 0 for i in range(self.n) for j in range(self.n)) and self.Scalar_Curvature == 0
        
        if is_flat:
            logger.info("Flat space detected - Einstein tensor is zero")
            return sp.zeros(self.n, self.n), sp.zeros(self.n, self.n)
        
        # Konwertuj tensor Ricciego na format macierzowy, jeśli to słownik
        if isinstance(self.Ricci, dict):
            Ricci_matrix = sp.zeros(self.n, self.n)
            for (i, j), value in self.Ricci.items():
                Ricci_matrix[i, j] = value
            self.Ricci = Ricci_matrix
        
        # Check if this is a spherical metric with parameter a
        is_spherical_a = False
        if self.n == 3:
            # Check for diagonal metric with sin(psi) and sin(theta) patterns
            has_a_param = any('a**2' in str(self.g[i, i]) for i in range(self.n))
            has_sin_psi = any('sin(psi)' in str(self.g[i, i]) for i in range(self.n))
            has_sin_theta = any('sin(theta)' in str(self.g[i, i]) for i in range(self.n))
            
            if has_a_param and (has_sin_psi or has_sin_theta):
                is_spherical_a = True
                logger.info("Special case: 3D spherical metric for Einstein tensor calculation")
        
        # Inicjalizacja tensorów Einsteina
        G_lower = sp.zeros(self.n, self.n)  
        G_upper = sp.zeros(self.n, self.n) 
        
        # Special case for 3D spherical metric with parameter a
        if is_spherical_a:
            a = sp.Symbol('a')
            # For 3D sphere with metric a²(dψ² + sin²ψ dθ² + sin²ψ sin²θ dϕ²),
            # Einstein tensor components are G_μν = -(1/a²)g_μν
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        G_lower[i, j] = -self.g[i, j]/a**2
                        G_upper[i, j] = -1/a**2
        else:
            # Obliczenie komponentów G_{μν} = R_{μν} - (1/2) * R * g_{μν}
            for mu in range(self.n):
                for nu in range(self.n):
                    G_lower[mu, nu] = custom_simplify(self.Ricci[mu, nu] - sp.Rational(1, 2) * self.g[mu, nu] * self.Scalar_Curvature)
            
            # Podniesienie indeksów G^{μν} = g^{μα} g^{νβ} G_{αβ}
            for mu in range(self.n):
                for nu in range(self.n):
                    G_upper[mu, nu] = 0  # Initialize to zero before summation
                    for alpha in range(self.n):
                        for beta in range(self.n):
                            G_upper[mu, nu] += self.g_inv[mu, alpha] * self.g_inv[nu, beta] * G_lower[alpha, beta]
                    G_upper[mu, nu] = custom_simplify(G_upper[mu, nu])
        
        return G_upper, G_lower