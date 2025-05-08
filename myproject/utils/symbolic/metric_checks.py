from utils.symbolic.simplification.custom_simplify import custom_simplify
import sympy as sp
import logging

logger = logging.getLogger(__name__)

class MetricChecks:
    def __init__(self, Riemann, n, tolerance=1e-10, extra_simplify=False):
        self.Riemann = Riemann
        self.n = n
        self.tolerance = tolerance
        self.extra_simplify = extra_simplify
    def is_flat_metric(self):
       
        logger.info("Checking if metric is flat")
        
        # Check each component of the Riemann tensor
        for rho in range(self.n):
            for sigma in range(self.n):
                for mu in range(self.n):
                    for nu in range(self.n):
                        # Get component value and simplify
                        value = custom_simplify(self.Riemann[rho][sigma][mu][nu])
                        
                        # Apply more aggressive simplification for complex expressions
                        if self.extra_simplify and not isinstance(value, (int, float)) and value != 0:
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
                                if abs(value_float) > self.tolerance:
                                    logger.info(f"Non-zero Riemann component found: R_{rho}{sigma}{mu}{nu} = {value}")
                                    return False
                            elif value != 0:
                                # Check if it's symbolically zero after simplification
                                if not value.is_zero:
                                    # Try one more simplification for complex expressions
                                    if self.extra_simplify and value.count_ops() > 10:
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