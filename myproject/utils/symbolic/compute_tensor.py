import sympy as sp
import logging
from .simplification.custom_simplify import custom_simplify
from .utilis.indexes import Indexes

logger = logging.getLogger(__name__)

class ComputeTensor:
    def __init__(self, coords, metric, dimension, evaluation_point=None):
        """
        Constructor for the ComputeTensor class for symbolic calculations.
        
        Args:
            coords (list): List of coordinate names, e.g. ['r', 'theta'].
            metric (list or dict): Metric tensor representation.
            dimension (int): Space dimension.
            evaluation_point (dict, optional): Points for evaluating symbolic expressions.
        """
        from sympy import symbols, Matrix, sympify
        
        logger.info(f"Initializing ComputeTensor with dimension {dimension}")
        
        self.coords = coords
        self.dimension = dimension
        self.n = dimension  # used in class methods
        self.evaluation_point = evaluation_point or {}
        
        # Create symbolic variables for coordinates
        try:
            self.coord_symbols = symbols(coords)
            if isinstance(self.coord_symbols, sp.Symbol):
                self.coord_symbols = [self.coord_symbols]
        except Exception as e:
            logger.error(f"Error creating coordinate symbols: {str(e)}")
            raise ValueError(f"Invalid coordinate names: {coords}") from e
        
        # Convert metric to SymPy matrix
        try:
            if isinstance(metric, list):
                if all(isinstance(row, list) for row in metric):
                    # Metric as a list of lists (matrix)
                    # Convert string expressions to sympy objects
                    sympy_metric = []
                    for row in metric:
                        sympy_row = []
                        for item in row:
                            if isinstance(item, str):
                                sympy_row.append(sympify(item))
                            else:
                                sympy_row.append(item)
                        sympy_metric.append(sympy_row)
                    self.g = Matrix(sympy_metric)
                else:
                    # Metric as a flat list
                    self.g = Matrix(dimension, dimension, metric)
            else:
                # Assume metric is already in the correct format
                self.g = metric
        except Exception as e:
            logger.error(f"Error creating metric matrix: {str(e)}")
            raise ValueError(f"Invalid metric format: {metric}") from e
        
        # Calculate inverse metric
        try:
            self.g_inv = self.g.inv()
        except Exception as e:
            logger.error(f"Error calculating inverse metric: {str(e)}")
            raise ValueError("Could not calculate inverse metric. The metric might be singular.") from e
        
        # Initialize the index manager
        self.indexes_manager = Indexes(dimension)
        self.christoffel_indexes = self.indexes_manager.generate_index_christoffel()
        self.riemann_indexes = self.indexes_manager.generate_index_riemann()
        self.ricci_indexes = self.indexes_manager.generate_index_ricci()
        
        # Fields to be calculated later
        self.R_abcd = None
        self.Ricci = None
        self.Scalar_Curvature = None
        
        # Perform calculations
        try:
            _, _, self.R_abcd, self.Ricci, self.Scalar_Curvature = self.oblicz_tensory()
            logger.info("Successfully calculated all tensor values")
        except Exception as e:
            logger.error(f"Error in tensor calculation: {str(e)}")
            raise ValueError("Failed to calculate tensors") from e

    def to_dict(self):
        """
        Converts the computed tensors to a dictionary for JSON serialization.
        
        Returns:
            dict: A dictionary containing all computed tensor values.
        """
        # Convert SymPy matrix to a list of lists of strings
        def matrix_to_str_list(matrix):
            if hasattr(matrix, 'tolist'):
                return [[str(item) for item in row] for row in matrix.tolist()]
            elif isinstance(matrix, list):
                if all(isinstance(row, list) for row in matrix):
                    return [[str(item) for item in row] for row in matrix]
                else:
                    return [str(item) for item in matrix]
            else:
                return str(matrix)
        
        christoffel = self.oblicz_tensory()[1]
        
        result = {
            "success": True,
            "dimension": self.dimension,
            "coordinates": self.coords,
            "metric": matrix_to_str_list(self.g),
            "christoffel": [[[str(gamma) for gamma in row] for row in matrix] for matrix in christoffel],
            "riemann": [[[[str(r) for r in row] for row in matrix] for matrix in tensor] for tensor in self.R_abcd],
            "ricci": matrix_to_str_list(self.Ricci),
            "scalar": str(self.Scalar_Curvature)
        }
        
        return result

    def compute_all(self):
        """
        Calculates all tensors and returns results.
        
        Returns:
            tuple: (g, Gamma, R_abcd, Ricci, Scalar) - all calculated tensors.
        """
        try:
            g, Gamma, R_abcd, Ricci, Scalar = self.oblicz_tensory()
            return g, Gamma, R_abcd, Ricci, Scalar
        except Exception as e:
            logger.error(f"Error in compute_all: {str(e)}")
            raise ValueError("Failed to compute tensors") from e

    def lower_riemann_indices(self):
        """Lowers Riemann tensor indices using the metric."""
        self.R_abcd = self.indexes_manager.lower_indices(self.R_abcd, self.g)
        return self.R_abcd
    
    def raise_riemann_indices(self):
        """Raises Riemann tensor indices using the inverse metric."""
        self.R_abcd = self.indexes_manager.raise_indices(self.R_abcd, self.g_inv)
        return self.R_abcd

    def show_indexes(self):
        """Displays automatically generated indices."""
        print("Riemann indices:", self.riemann_indexes)
        print("Ricci indices:", self.ricci_indexes)
        print("Christoffel indices:", self.christoffel_indexes)

    def oblicz_tensory(self):
        """
        Calculates basic geometric tensors for the given metric.
        """
        n = self.n
        logger.info(f"Starting tensor calculations for {n} dimensions")

        # Calculate metric tensor
        g_inv = self.g.inv() if isinstance(self.g, sp.Matrix) else self.g_inv
        
        # Create symbolic variables if they don't exist yet
        if not hasattr(self, 'coord_symbols') or self.coord_symbols is None:
            self.coord_symbols = sp.symbols(self.coords)
        if isinstance(self.coord_symbols, sp.Symbol):
            self.coord_symbols = [self.coord_symbols]
            
        # Calculate Christoffel symbols
        Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    try:
                        Gamma_sum = 0
                        for lam in range(n):
                            partial_mu = sp.diff(self.g[nu, lam], self.coord_symbols[mu])
                            partial_nu = sp.diff(self.g[mu, lam], self.coord_symbols[nu])
                            partial_lam = sp.diff(self.g[mu, nu], self.coord_symbols[lam])
                            Gamma_sum += g_inv[sigma, lam] * (partial_mu + partial_nu - partial_lam)
                        Gamma[sigma][mu][nu] = custom_simplify(sp.Rational(1, 2) * Gamma_sum)
                    except Exception as e:
                        logger.error(f"Error calculating Christoffel symbol [{sigma}][{mu}][{nu}]: {str(e)}")
                        Gamma[sigma][mu][nu] = 0
        
        # Calculate Riemann tensor
        R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        try:
                            term1 = sp.diff(Gamma[rho][nu][sigma], self.coord_symbols[mu])
                            term2 = sp.diff(Gamma[rho][mu][sigma], self.coord_symbols[nu])
                            sum_term = sum(
                                Gamma[rho][mu][lam] * Gamma[lam][nu][sigma] 
                                - Gamma[rho][nu][lam] * Gamma[lam][mu][sigma]
                                for lam in range(n)
                            )
                            R_abcd[rho][sigma][mu][nu] = custom_simplify(term1 - term2 + sum_term)
                        except Exception as e:
                            logger.error(f"Error calculating Riemann component [{rho}][{sigma}][{mu}][{nu}]: {str(e)}")
                            R_abcd[rho][sigma][mu][nu] = 0

        # Lower Riemann tensor indices
        try:
            R_abcd = self.indexes_manager.lower_indices(R_abcd, self.g)
        except Exception as e:
            logger.error(f"Error lowering Riemann indices: {str(e)}")

        # Calculate Ricci tensor
        Ricci = sp.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                try:
                    Ricci[mu, nu] = sum(R_abcd[rho][mu][rho][nu] for rho in range(n))
                    Ricci[mu, nu] = custom_simplify(Ricci[mu, nu])
                except Exception as e:
                    logger.error(f"Error calculating Ricci component [{mu}][{nu}]: {str(e)}")
                    Ricci[mu, nu] = 0

        # Calculate Ricci scalar (scalar curvature)
        try:
            Scalar_Curvature = sum(
                g_inv[mu, nu] * Ricci[mu, nu] for mu in range(n) for nu in range(n)
            )
            Scalar_Curvature = custom_simplify(Scalar_Curvature)
        except Exception as e:
            logger.error(f"Error calculating scalar curvature: {str(e)}")
            Scalar_Curvature = 0

        return self.g, Gamma, R_abcd, Ricci, Scalar_Curvature
