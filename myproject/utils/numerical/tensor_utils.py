import numpy as np
import math
import logging
from typing import Dict, List, Union, Callable, Any, Tuple

logger = logging.getLogger(__name__)

def parse_metric_components(
    metric_data: List[List[Union[float, str]]], 
    coordinates: List[str]
) -> Dict[Tuple[int, int], Callable]:
    """
    Convert metric components from JSON input to callable functions.
    
    Args:
        metric_data: A nested list representing the metric tensor, with numeric values or string expressions
        coordinates: List of coordinate names
        
    Returns:
        Dictionary mapping (i,j) tuples to callable functions for each metric component
    """
    n = len(metric_data)
    metric_funcs = {}
    
    for i in range(n):
        for j in range(n):
            component = metric_data[i][j]
            
            # Handle None, null, or empty string values by treating them as zero
            if component is None or component == "" or component == "null":
                logger.warning(f"Empty or null metric component at position [{i}][{j}], using default value of 0.0")
                metric_funcs[(i, j)] = lambda coords, val=0.0: val
            elif isinstance(component, (int, float)):
                # For constant metric components, create a function that always returns this value
                metric_funcs[(i, j)] = lambda coords, val=component: val
            elif isinstance(component, str):
                # For string expressions, create a function that evaluates the expression
                metric_funcs[(i, j)] = create_component_function(component, coordinates)
            else:
                logger.error(f"Unsupported metric component type: {type(component)} at position [{i}][{j}]")
                raise ValueError(f"Unsupported metric component type: {type(component)} at position [{i}][{j}]")
    
    return metric_funcs

def create_component_function(expr: str, coordinates: List[str]) -> Callable:
    """
    Create a function from a string expression that can be evaluated with coordinate values.
    
    Args:
        expr: String representation of a mathematical expression (e.g., "r**2", "1/(1 - 2/r)")
        coordinates: List of coordinate names
        
    Returns:
        A function that takes coordinate values and returns the evaluated expression
        
    Raises:
        ValueError: If the expression has syntax errors or contains invalid operations
        
    Note:
        The expression must be a valid Python arithmetic expression. Examples:
        - Simple expressions: "1", "r**2", "sin(theta)"
        - Complex expressions: "1/(1 - 2/r)", "r**2 * sin(theta)**2"
        All parentheses must be properly closed and the expression must use valid mathematical operations.
        
        If the expression is empty or contains only whitespace, a function that returns 0.0 will be returned.
    """
    # Input validation - check if expression is empty or just whitespace
    if not expr or not str(expr).strip():
        logger.warning("Empty or whitespace-only metric component expression provided. Using default value of 0.0.")
        # Return a function that always returns 0.0 for empty expressions
        return lambda coords: 0.0
    
    # Try to convert numeric strings directly to floats for efficiency
    try:
        numeric_value = float(expr)
        # If we can convert it to a float, just return a constant function
        return lambda coords, val=numeric_value: val
    except (ValueError, TypeError):
        # Not a simple numeric value, continue with expression parsing
        pass
    
    # Import math functions for use in the evaluation
    import math
    
    # Create a safe evaluation namespace with mathematical functions
    # We'll combine numpy and math module functions for completeness
    safe_namespace = {
        "__builtins__": {},  # Restrict builtins for security
        # NumPy functions
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'pi': np.pi,
        'abs': np.abs,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'arcsinh': np.arcsinh,
        'arccosh': np.arccosh,
        'arctanh': np.arctanh,
        
        # Add additional math module functions
        'e': math.e,
        'ceil': math.ceil,
        'floor': math.floor,
        'degrees': math.degrees,
        'radians': math.radians,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'asinh': math.asinh,
        'acosh': math.acosh,
        'atanh': math.atanh,
        'pow': math.pow,
        'log10': math.log10,
        'log2': math.log2,
        'fabs': math.fabs,
    }
    
    # First validate syntax before creating the function
    try:
        compiled_expr = compile(expr, '<string>', 'eval')
    except SyntaxError as e:
        logger.error(f"Syntax error in expression '{expr}': {str(e)}")
        raise ValueError(f"Failed to parse expression: '{expr}'. Syntax error: {str(e)}") from e
    
    def component_func(coords):
        # Create a namespace with coordinate values
        namespace = {**safe_namespace}
        for i, name in enumerate(coordinates):
            namespace[name] = coords[i]
        
        # Safely evaluate the expression with enhanced error handling
        try:
            result = eval(compiled_expr, {"__builtins__": {}}, namespace)
            return float(result)
        except NameError as e:
            logger.error(f"Name error in expression '{expr}': {str(e)}")
            raise ValueError(f"Unknown variable or function in '{expr}': {str(e)}") from e
        except TypeError as e:
            logger.error(f"Type error in expression '{expr}': {str(e)}")
            raise ValueError(f"Type error in expression '{expr}': {str(e)}") from e
        except ZeroDivisionError as e:
            logger.error(f"Division by zero in expression '{expr}': {str(e)}")
            # This is an expected error in some cases (like at the Schwarzschild radius)
            # So we'll return infinity instead of raising an error
            return float('inf')
        except Exception as e:
            logger.error(f"Error evaluating expression '{expr}': {str(e)}")
            raise ValueError(f"Failed to evaluate expression '{expr}': {str(e)}") from e
    
    return component_func

def evaluate_metric_at_point(
    metric_funcs: Dict[Tuple[int, int], Callable], 
    coords: np.ndarray
) -> np.ndarray:
    """
    Evaluate the metric tensor at a specific coordinate point.
    
    Args:
        metric_funcs: Dictionary of metric component functions
        coords: Array of coordinate values
        
    Returns:
        Numpy array representing the metric tensor at the given point
    """
    n = int(np.sqrt(len(metric_funcs)))
    g = np.zeros((n, n))
    
    for (i, j), func in metric_funcs.items():
        g[i, j] = func(coords)
    
    return g

def invert_metric(g: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of the metric tensor.
    
    Args:
        g: Metric tensor as a numpy array
        
    Returns:
        Inverse metric tensor
    """
    try:
        g_inv = np.linalg.inv(g)
        return g_inv
    except np.linalg.LinAlgError as e:
        logger.error(f"Failed to invert metric: {str(e)}")
        raise ValueError("The metric tensor is singular and cannot be inverted")

def metric_derivatives(
    metric_funcs: Dict[Tuple[int, int], Callable], 
    coords: np.ndarray, 
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute partial derivatives of the metric tensor at a given point using finite differences.
    
    Args:
        metric_funcs: Dictionary of metric component functions
        coords: Coordinate values at which to evaluate derivatives
        h: Step size for finite difference approximation
        
    Returns:
        3D array where partials[k,i,j] = ∂g_ij/∂x^k
    """
    n = int(np.sqrt(len(metric_funcs)))
    partials = np.zeros((n, n, n))
    
    # Save the metric at the base point to avoid recomputation
    g0 = evaluate_metric_at_point(metric_funcs, coords)
    
    for k in range(n):  # For each coordinate direction
        coords_forward = coords.copy()
        coords_backward = coords.copy()
        coords_forward[k] += h
        coords_backward[k] -= h
        
        # Evaluate metric at offset points
        g_forward = evaluate_metric_at_point(metric_funcs, coords_forward)
        g_backward = evaluate_metric_at_point(metric_funcs, coords_backward)
        
        # Compute central difference approximation of the derivative
        partials[k] = (g_forward - g_backward) / (2 * h)
    
    return partials

def christoffel_symbols(
    metric_funcs: Dict[Tuple[int, int], Callable], 
    coords: np.ndarray
) -> np.ndarray:
    """
    Compute the Christoffel symbols (connection coefficients) at the given point.
    
    Formula: Γ^a_bc = 0.5 * g^ad * (∂_b g_dc + ∂_c g_db - ∂_d g_bc)
    
    Args:
        metric_funcs: Dictionary of metric component functions
        coords: Coordinate values at which to evaluate
        
    Returns:
        3D array Gamma[a,b,c] representing Γ^a_bc
    """
    n = int(np.sqrt(len(metric_funcs)))
    
    # Evaluate metric and its inverse at the point
    g = evaluate_metric_at_point(metric_funcs, coords)
    g_inv = invert_metric(g)
    
    # Compute metric derivatives
    partials = metric_derivatives(metric_funcs, coords)
    
    # Initialize Christoffel symbols array
    Gamma = np.zeros((n, n, n))
    
    # Compute Christoffel symbols using tensor operations
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    # Γ^a_bc = 0.5 * g^ad * (∂_b g_dc + ∂_c g_db - ∂_d g_bc)
                    Gamma[a, b, c] += 0.5 * g_inv[a, d] * (
                        partials[b, d, c] + partials[c, d, b] - partials[d, b, c]
                    )
    
    return Gamma

def christoffel_symbol_derivatives(
    Gamma: np.ndarray, 
    metric_funcs: Dict[Tuple[int, int], Callable], 
    coords: np.ndarray, 
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute derivatives of Christoffel symbols using finite differences.
    
    Args:
        Gamma: Christoffel symbols at the base point
        metric_funcs: Dictionary of metric component functions
        coords: Coordinate values at the base point
        h: Step size for finite differences
        
    Returns:
        4D array where dGamma[a,b,c,d] = ∂_d Γ^a_bc
    """
    n = Gamma.shape[0]
    dGamma = np.zeros((n, n, n, n))
    
    # Pre-compute Christoffel symbols at offset points along each coordinate
    Gamma_plus = []
    Gamma_minus = []
    
    for k in range(n):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        
        coords_plus[k] += h
        coords_minus[k] -= h
        
        Gamma_plus.append(christoffel_symbols(metric_funcs, coords_plus))
        Gamma_minus.append(christoffel_symbols(metric_funcs, coords_minus))
    
    # Compute derivatives using central differences
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    dGamma[a, b, c, d] = (Gamma_plus[d][a, b, c] - Gamma_minus[d][a, b, c]) / (2 * h)
    
    return dGamma

def riemann_tensor(
    metric_funcs: Dict[Tuple[int, int], Callable], 
    coords: np.ndarray
) -> np.ndarray:
    """
    Compute the Riemann curvature tensor R^a_bcd at the given point.
    
    Formula: R^a_bcd = ∂_c Γ^a_bd - ∂_d Γ^a_bc + Γ^a_ce Γ^e_bd - Γ^a_de Γ^e_bc
    
    Args:
        metric_funcs: Dictionary of metric component functions
        coords: Coordinate values at which to evaluate
        
    Returns:
        4D array R[a,b,c,d] representing R^a_bcd
    """
    n = int(np.sqrt(len(metric_funcs)))
    
    # Compute Christoffel symbols
    Gamma = christoffel_symbols(metric_funcs, coords)
    
    # Compute derivatives of Christoffel symbols
    dGamma = christoffel_symbol_derivatives(Gamma, metric_funcs, coords)
    
    # Initialize Riemann tensor
    R = np.zeros((n, n, n, n))
    
    # Compute Riemann tensor components
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    # First term: ∂_c Γ^a_bd
                    R[a, b, c, d] = dGamma[a, b, d, c]
                    
                    # Second term: -∂_d Γ^a_bc
                    R[a, b, c, d] -= dGamma[a, b, c, d]
                    
                    # Third and fourth terms: Γ^a_ce Γ^e_bd - Γ^a_de Γ^e_bc
                    for e in range(n):
                        R[a, b, c, d] += Gamma[a, c, e] * Gamma[e, b, d] - Gamma[a, d, e] * Gamma[e, b, c]
    
    return R

def ricci_tensor(R: np.ndarray) -> np.ndarray:
    """
    Compute the Ricci tensor by contracting the Riemann tensor.
    
    Formula: R_bd = R^a_bad
    
    Args:
        R: Riemann tensor as a 4D array
        
    Returns:
        2D array representing the Ricci tensor
    """
    n = R.shape[0]
    Ricci = np.zeros((n, n))
    
    # Contract the Riemann tensor
    for b in range(n):
        for d in range(n):
            for a in range(n):
                Ricci[b, d] += R[a, b, a, d]
    
    return Ricci

def ricci_scalar(Ricci: np.ndarray, g_inv: np.ndarray) -> float:
    """
    Compute the Ricci scalar (curvature scalar) by contracting the Ricci tensor with the inverse metric.
    
    Formula: R = g^ij R_ij
    
    Args:
        Ricci: Ricci tensor as a 2D array
        g_inv: Inverse metric tensor
        
    Returns:
        Ricci scalar (scalar curvature)
    """
    n = Ricci.shape[0]
    scalar = 0.0
    
    # Contract Ricci tensor with inverse metric
    for i in range(n):
        for j in range(n):
            scalar += g_inv[i, j] * Ricci[i, j]
    
    return scalar

def einstein_tensor(Ricci: np.ndarray, R_scalar: float, g: np.ndarray) -> np.ndarray:
    """
    Compute the Einstein tensor.
    
    Formula: G_ij = R_ij - 0.5 * R * g_ij
    
    Args:
        Ricci: Ricci tensor
        R_scalar: Ricci scalar
        g: Metric tensor
        
    Returns:
        Einstein tensor as a 2D array
    """
    n = Ricci.shape[0]
    G = np.zeros((n, n))
    
    # G_ij = R_ij - 0.5 * R * g_ij
    for i in range(n):
        for j in range(n):
            G[i, j] = Ricci[i, j] - 0.5 * R_scalar * g[i, j]
    
    return G

def weyl_tensor(R: np.ndarray, Ricci: np.ndarray, R_scalar: float, g: np.ndarray) -> np.ndarray:
    """
    Compute the Weyl tensor (trace-free part of the Riemann tensor).
    
    Formula (in 4D): C_abcd = R_abcd - (g_ac R_bd - g_ad R_bc - g_bc R_ad + g_bd R_ac) / 2 + R * (g_ac g_bd - g_ad g_bc) / 6
    
    Args:
        R: Riemann tensor with mixed indices R^a_bcd
        Ricci: Ricci tensor
        R_scalar: Ricci scalar
        g: Metric tensor
        
    Returns:
        Weyl tensor as a 4D array with mixed indices C^a_bcd
    """
    n = g.shape[0]
    
    # Convert Riemann tensor to fully covariant form
    R_cov = np.zeros((n, n, n, n))
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    for e in range(n):
                        R_cov[a, b, c, d] += g[a, e] * R[e, b, c, d]
    
    # Initialize Weyl tensor (in covariant form)
    C_cov = np.zeros((n, n, n, n))
    
    # Compute the Weyl tensor
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    # Start with the Riemann tensor
                    C_cov[a, b, c, d] = R_cov[a, b, c, d]
                    
                    # Subtract the semi-traceless part
                    C_cov[a, b, c, d] -= (g[a, c] * Ricci[b, d] - g[a, d] * Ricci[b, c] - 
                                          g[b, c] * Ricci[a, d] + g[b, d] * Ricci[a, c]) / 2.0
                    
                    # Add the scalar curvature part
                    C_cov[a, b, c, d] += R_scalar * (g[a, c] * g[b, d] - g[a, d] * g[b, c]) / 6.0
    
    # Convert back to mixed form (C^a_bcd)
    C = np.zeros((n, n, n, n))
    g_inv = invert_metric(g)
    
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    for e in range(n):
                        C[a, b, c, d] += g_inv[a, e] * C_cov[e, b, c, d]
    
    return C

def calculate_all_tensors(
    metric_data: List[List[Union[float, str]]],
    coordinates: List[str],
    evaluation_point: List[float]
) -> Dict[str, Any]:
    """
    Calculate all tensor quantities at the given evaluation point.
    
    Args:
        metric_data: Nested list of metric components
        coordinates: List of coordinate names
        evaluation_point: Values of coordinates at which to evaluate
        
    Returns:
        Dictionary containing all calculated tensors
    """
    n = len(coordinates)
    if len(evaluation_point) != n:
        raise ValueError(f"Evaluation point must have {n} coordinates, got {len(evaluation_point)}")
    
    # Parse metric components into callable functions
    metric_funcs = parse_metric_components(metric_data, coordinates)
    
    # Convert evaluation point to numpy array
    coords = np.array(evaluation_point, dtype=float)
    
    # Calculate metric and inverse metric
    g = evaluate_metric_at_point(metric_funcs, coords)
    g_inv = invert_metric(g)
    
    # Calculate Christoffel symbols
    Gamma = christoffel_symbols(metric_funcs, coords)
    
    # Calculate Riemann tensor
    R = riemann_tensor(metric_funcs, coords)
    
    # Calculate Ricci tensor
    Ricci = ricci_tensor(R)
    
    # Calculate Ricci scalar
    R_scalar = ricci_scalar(Ricci, g_inv)
    
    # Calculate Einstein tensor
    G = einstein_tensor(Ricci, R_scalar, g)
    
    # Calculate Weyl tensor
    C = weyl_tensor(R, Ricci, R_scalar, g)
    
    # Return all calculated values
    result = {
        "metric": g.tolist(),
        "inverse_metric": g_inv.tolist(),
        "christoffel_symbols": format_tensor_components(Gamma),
        "riemann_tensor": format_tensor_components(R),
        "ricci_tensor": Ricci.tolist(),
        "ricci_scalar": float(R_scalar),
        "einstein_tensor": G.tolist(),
        "weyl_tensor": format_tensor_components(C),
        "dimension": n,
        "coordinates": coordinates,
        "evaluation_point": evaluation_point
    }
    
    return result

def format_tensor_components(tensor: np.ndarray) -> Dict[str, float]:
    """
    Format a tensor's non-zero components as a dictionary with indices as keys.
    
    Args:
        tensor: NumPy array representing a tensor
        
    Returns:
        Dictionary with string indices as keys and component values as values
    """
    components = {}
    shape = tensor.shape
    
    # Create a function to iterate over all indices of the tensor
    def iterate_indices(shape, current_indices=None):
        if current_indices is None:
            current_indices = []
        
        if len(current_indices) == len(shape):
            # We have a complete set of indices
            value = tensor[tuple(current_indices)]
            if abs(value) > 1e-10:  # Filter out near-zero values
                key = ",".join(map(str, current_indices))
                components[key] = float(value)
            return
        
        # Recursively build all index combinations
        for i in range(shape[len(current_indices)]):
            iterate_indices(shape, current_indices + [i])
    
    iterate_indices(shape)
    return components 