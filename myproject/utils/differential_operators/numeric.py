"""
Numerical implementation of differential operators in curvilinear coordinates.

This module provides functionality to compute:
1. Discretization of fields on grids in curvilinear coordinates
2. Numerical evaluation of gradient, divergence, curl, Laplacian, and d'Alembertian operators
3. Finite difference methods for derivatives in curvilinear coordinates

All calculations are performed numerically using NumPy.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)

def create_grid(coords_ranges: Dict[str, Dict[str, float]], 
                dimensions: List[int]) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    Create a grid in the specified coordinate system.
    
    Args:
        coords_ranges: Dictionary mapping coordinate names to their ranges (min, max)
        dimensions: List of number of points along each dimension
        
    Returns:
        Tuple of (grid_points, spacing) where:
        - grid_points is a list of arrays containing the coordinate values along each axis
        - spacing is a dictionary mapping coordinate names to spacing values
    """
    grid_points = []
    spacing = {}
    
    for i, (coord_name, coord_range) in enumerate(coords_ranges.items()):
        min_val = coord_range['min']
        max_val = coord_range['max']
        n_points = dimensions[i]
        
        # Special handling for spherical coordinates to avoid singular points
        if coord_name.lower() == 'r':
            # Ensure r is not zero (avoiding coordinate singularity)
            if min_val <= 0:
                logger.warning(f"Minimum value for r must be positive. Adjusting from {min_val} to 1e-6")
                min_val = 1e-6
        
        elif coord_name.lower() in ['theta', 'θ']:
            # Avoid theta = 0 and theta = pi (to avoid sin(theta) = 0)
            if min_val <= 0:
                logger.warning(f"Minimum value for theta must be positive. Adjusting from {min_val} to 1e-6")
                min_val = 1e-6
            if max_val >= np.pi:
                logger.warning(f"Maximum value for theta must be less than π. Adjusting from {max_val} to π-1e-6")
                max_val = np.pi - 1e-6
        
        # Create evenly spaced points
        coord_points = np.linspace(min_val, max_val, n_points)
        grid_points.append(coord_points)
        
        # Calculate spacing
        spacing[coord_name] = (max_val - min_val) / (n_points - 1)
    
    return grid_points, spacing

def discretize_field(field_func: Callable, grid: List[np.ndarray]) -> np.ndarray:
    """
    Discretize a scalar field function onto a grid.
    
    Args:
        field_func: Function that takes coordinate values and returns the field value
        grid: List of arrays containing the coordinate values along each axis
        
    Returns:
        Array of field values at each grid point
    """
    # Create meshgrid for evaluation
    mesh = np.meshgrid(*grid, indexing='ij')
    grid_shape = [len(g) for g in grid]
    
    # Try different approaches based on function signature
    try:
        # First approach: try vectorized evaluation with stacked coordinates
        coords = np.stack([m.flatten() for m in mesh], axis=-1)
        try:
            values = field_func(coords)
            return values.reshape(grid_shape)
        except (TypeError, ValueError):
            # This might fail if field_func expects separate arguments for each coordinate
            pass
        
        # Second approach: try calling with unpacked mesh
        try:
            values = field_func(*mesh)
            if isinstance(values, np.ndarray) and values.shape == tuple(grid_shape):
                return values
        except (TypeError, ValueError):
            # This might fail if field_func expects a single point at a time
            pass
        
        # Fall back to loop-based evaluation
        logger.info("Falling back to loop-based field discretization")
        values = np.zeros(grid_shape)
        
        # Use nested loops to evaluate at each point
        for idx in np.ndindex(*grid_shape):
            point = [grid[d][idx[d]] for d in range(len(grid))]
            try:
                values[idx] = field_func(point)  # Try passing a list of coordinates
            except (TypeError, ValueError):
                try:
                    values[idx] = field_func(*point)  # Try unpacking the coordinates
                except Exception as e:
                    logger.error(f"Error evaluating field function: {e}")
                    values[idx] = np.nan
        
        return values
    
    except Exception as e:
        logger.error(f"Error during field discretization: {e}")
        # Return a grid of NaN values as a fallback
        return np.full(grid_shape, np.nan)

def discretize_vector_field(field_funcs: List[Callable], grid: List[np.ndarray]) -> List[np.ndarray]:
    """
    Discretize a vector field function onto a grid.
    
    Args:
        field_funcs: List of functions, one for each vector component
        grid: List of arrays containing the coordinate values along each axis
        
    Returns:
        List of arrays, one for each vector component, with values at each grid point
    """
    return [discretize_field(func, grid) for func in field_funcs]

def compute_partial_derivative(field: np.ndarray, grid: List[np.ndarray], 
                              direction: int, order: int = 2) -> np.ndarray:
    """
    Compute partial derivative of a field using finite differences.
    
    Args:
        field: Array of field values at grid points
        grid: List of arrays containing the coordinate values along each axis
        direction: Direction (axis) along which to differentiate
        order: Order of accuracy (1: forward/backward, 2: central)
        
    Returns:
        Array of derivative values
    """
    # Get the coordinate spacing
    dx = grid[direction][1] - grid[direction][0]
    
    # Use np.gradient for central differences (2nd order accurate)
    if order == 2:
        # Using numpy's gradient which handles non-uniform spacing
        # When computing gradient along a specific axis, we only need to provide
        # the coordinate array for that axis, not all axes
        grad = np.gradient(field, grid[direction], axis=direction, edge_order=2)
        return grad
    
    # For 1st order, implement forward and backward differences
    elif order == 1:
        # Forward difference at the beginning, backward at the end, central in the middle
        result = np.zeros_like(field)
        
        slices_before = [slice(None)] * field.ndim
        slices_at = [slice(None)] * field.ndim
        slices_after = [slice(None)] * field.ndim
        
        # Forward difference for first point
        slices_at[direction] = slice(0, 1)
        slices_after[direction] = slice(1, 2)
        result[tuple(slices_at)] = (field[tuple(slices_after)] - field[tuple(slices_at)]) / dx
        
        # Central difference for middle points
        slices_before[direction] = slice(0, -2)
        slices_at[direction] = slice(1, -1)
        slices_after[direction] = slice(2, None)
        result[tuple(slices_at)] = (field[tuple(slices_after)] - field[tuple(slices_before)]) / (2 * dx)
        
        # Backward difference for last point
        slices_before[direction] = slice(-2, -1)
        slices_at[direction] = slice(-1, None)
        result[tuple(slices_at)] = (field[tuple(slices_at)] - field[tuple(slices_before)]) / dx
        
        return result
    
    else:
        raise ValueError(f"Order {order} not supported for finite differences")

def compute_christoffel_on_grid(metric_funcs: List[List[Callable]], grid: List[np.ndarray]) -> np.ndarray:
    """
    Compute Christoffel symbols on a grid.
    
    Args:
        metric_funcs: List of callables for metric components
        grid: List of arrays containing the coordinate values along each axis
        
    Returns:
        Array of Christoffel symbols at each grid point
    """
    import warnings
    n = len(grid)
    
    # Create meshgrid for evaluation
    mesh = np.meshgrid(*grid, indexing='ij')
    grid_shape = [len(g) for g in grid]
    
    # Try to get a sample metric value to check if it's a constant metric
    try:
        sample_point = [grid[d][0] for d in range(n)]
        sample_metric = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if callable(metric_funcs[i][j]):
                    sample_metric[i, j] = metric_funcs[i][j](sample_point)
                else:
                    # If not callable, assume it's a constant value
                    sample_metric[i, j] = metric_funcs[i][j]
        
        # Check if all metric functions return constant values
        is_constant_metric = True
        for idx in [0, -1]:  # Check at first and last points
            test_point = [grid[d][idx if idx < len(grid[d]) else -1] for d in range(n)]
            for i in range(n):
                for j in range(n):
                    if callable(metric_funcs[i][j]):
                        if abs(metric_funcs[i][j](test_point) - sample_metric[i, j]) > 1e-10:
                            is_constant_metric = False
                            break
            if not is_constant_metric:
                break
        
        if is_constant_metric:
            logger.info("Detected constant metric tensor")
            # For constant metric, simplified computation
            metric = sample_metric
            det = np.linalg.det(metric)
            if abs(det) < 1e-10:
                warnings.warn(f"Singular metric detected (det={det}). Using pseudoinverse.")
                metric_inv = np.linalg.pinv(metric)
            else:
                try:
                    metric_inv = np.linalg.inv(metric)
                except np.linalg.LinAlgError:
                    # Add a small regularization if needed
                    warnings.warn("Error inverting metric. Using pseudoinverse.")
                    metric_inv = np.linalg.pinv(metric)
            
            # For Cartesian coordinates with identity metric, all Christoffel symbols are zero
            if np.allclose(metric, np.eye(n)):
                logger.info("Identity metric detected, all Christoffel symbols are zero")
                return np.zeros(grid_shape + (n, n, n))
            
            # For constant non-Cartesian metric, compute derivatives
            metric_deriv = np.zeros((n, n, n))
            
            # For constant metric, all derivatives should be zero
            # But we'll compute them anyway for completeness
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        try:
                            # Try to compute symbolic derivative for more accuracy
                            # For now, just assume zero derivatives for constant metric
                            metric_deriv[i, j, k] = 0.0
                        except:
                            metric_deriv[i, j, k] = 0.0
            
            # Compute Christoffel symbols
            Gamma = np.zeros(grid_shape + (n, n, n))
            
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        sum_term = 0.0
                        for l in range(n):
                            term1 = metric_deriv[l, j, i]  # ∂_i g_lj
                            term2 = metric_deriv[l, i, j]  # ∂_j g_li
                            term3 = metric_deriv[i, j, l]  # ∂_l g_ij
                            sum_term += metric_inv[k, l] * (term1 + term2 - term3)
                        
                        # Fill the entire grid with the same value
                        for idx in np.ndindex(*grid_shape):
                            Gamma[idx][k][i][j] = 0.5 * sum_term
            
            return Gamma
    
    except Exception as e:
        logger.warning(f"Error checking for constant metric: {e}")
        # Continue with regular computation
    
    # Initialize arrays for metric and its derivatives
    metric = np.zeros(grid_shape + (n, n))
    metric_inv = np.zeros_like(metric)
    metric_deriv = np.zeros(grid_shape + (n, n, n))  # Additional index for derivative direction
    
    # Compute metric at each grid point
    for idx in np.ndindex(*grid_shape):
        point = [grid[d][idx[d]] for d in range(n)]
        
        # Evaluate metric components at this point
        g_point = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    if callable(metric_funcs[i][j]):
                        g_point[i, j] = metric_funcs[i][j](point)
                    else:
                        g_point[i, j] = metric_funcs[i][j]
                except Exception as e:
                    logger.warning(f"Error evaluating metric at {point}: {e}")
                    # Use a reasonable default (identity metric component)
                    g_point[i, j] = 1.0 if i == j else 0.0
        
        # Check if the metric is singular and use appropriate inversion method
        try:
            det = np.linalg.det(g_point)
            if abs(det) < 1e-10:
                warnings.warn(f"Singular matrix encountered at grid point {idx}. Using pseudoinverse.")
                metric[idx] = g_point
                metric_inv[idx] = np.linalg.pinv(g_point)
            else:
                metric[idx] = g_point
                metric_inv[idx] = np.linalg.inv(g_point)
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Inversion error at grid point {idx}: {e}")
            metric[idx] = g_point
            # Use pseudoinverse for more robust handling of singular or near-singular matrices
            metric_inv[idx] = np.linalg.pinv(g_point)
    
    # Compute metric derivatives using finite differences
    for d in range(n):  # Derivative direction
        for i in range(n):
            for j in range(n):
                metric_component = metric[..., i, j]
                metric_deriv[..., i, j, d] = compute_partial_derivative(metric_component, grid, d)
    
    # Initialize Christoffel symbols array
    Gamma = np.zeros(grid_shape + (n, n, n))  # Indexed as Gamma[x,y,z,...][k][i][j]
    
    # Compute Christoffel symbols at each grid point using the formula:
    # Γ^k_ij = (1/2) g^kl (∂_i g_lj + ∂_j g_li - ∂_l g_ij)
    for idx in np.ndindex(*grid_shape):
        g_inv = metric_inv[idx]
        g_deriv = metric_deriv[idx]
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    sum_term = 0
                    for l in range(n):
                        term1 = g_deriv[l, j, i]  # ∂_i g_lj
                        term2 = g_deriv[l, i, j]  # ∂_j g_li
                        term3 = g_deriv[i, j, l]  # ∂_l g_ij
                        sum_term += g_inv[k, l] * (term1 + term2 - term3)
                    
                    Gamma[idx][k][i][j] = 0.5 * sum_term
    
    return Gamma

def evaluate_gradient(scalar_field: np.ndarray, metric_inverse: np.ndarray, 
                     grid: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute the gradient of a scalar field numerically.
    
    Args:
        scalar_field: Array of scalar field values
        metric_inverse: Array of inverse metric values at each point
        grid: List of arrays containing the coordinate values along each axis
        
    Returns:
        List of arrays for gradient components (contravariant)
    """
    n = len(grid)
    grid_shape = scalar_field.shape
    
    # Compute partial derivatives
    partial_derivs = [compute_partial_derivative(scalar_field, grid, d) for d in range(n)]
    
    # Initialize gradient components (contravariant)
    gradient = [np.zeros(grid_shape) for _ in range(n)]
    
    # Determine the shape of metric_inverse to handle different cases correctly
    if metric_inverse.ndim == 2:
        # Case 1: Constant metric (shape: (3, 3)) - apply to all grid points
        for i in range(n):
            for j in range(n):
                # Contract the inverse metric with partial derivatives
                gradient[i] += metric_inverse[i, j] * partial_derivs[j]
    
    elif metric_inverse.ndim == n + 2:
        # Case 2: Position-dependent metric (shape: (N_r, N_theta, N_phi, 3, 3))
        # Use vectorized operations with proper broadcasting
        for i in range(n):
            for j in range(n):
                # For each component (i,j) of the metric inverse, multiply with the j-th derivative
                # and accumulate to the i-th gradient component
                gradient[i] += metric_inverse[..., i, j] * partial_derivs[j]
    
    else:
        # Case 3: Partially position-dependent metric or other shape
        # Use explicit nested loops for point-by-point contraction
        # This is the most general but slowest approach
        import warnings
        warnings.warn(f"Metric inverse has unexpected shape {metric_inverse.shape}. "
                     f"Using explicit point-wise contraction, which may be slow.")
        
        # Create indexing arrays for each dimension
        idx_arrays = np.indices(grid_shape)
        
        # Perform contraction at each grid point
        for idx in np.ndindex(*grid_shape):
            # Extract metric inverse at this grid point
            if metric_inverse.ndim >= n:
                # If metric has grid dimensions, extract the 3x3 matrix at this point
                try:
                    # Try to extract the metric at this grid point
                    g_inv = metric_inverse[idx[:n]]
                except (IndexError, TypeError):
                    # If the indexing fails, try another approach based on the shape
                    if metric_inverse.shape[:n] == (1,) * n:
                        # If the metric has size 1 in each grid dimension, extract the first element
                        g_inv = metric_inverse[(0,) * n]
                    else:
                        # Fall back to identity matrix if all else fails
                        g_inv = np.eye(n)
                        warnings.warn(f"Could not extract metric at index {idx}. Using identity.")
            else:
                # If metric doesn't have grid dimensions, use it directly
                g_inv = metric_inverse
            
            # Compute the gradient at this point by contracting with partial derivatives
            for i in range(n):
                grad_i_at_point = 0
                for j in range(n):
                    grad_i_at_point += g_inv[i, j] * partial_derivs[j][idx]
                gradient[i][idx] = grad_i_at_point
    
    return gradient

def evaluate_divergence(vector_field: List[np.ndarray], metric: np.ndarray, 
                       grid: List[np.ndarray], is_contravariant: bool = True) -> np.ndarray:
    """
    Compute the divergence of a vector field numerically.
    
    Args:
        vector_field: List of arrays for vector components
        metric: Array of metric values at each point
        grid: List of arrays containing the coordinate values along each axis
        is_contravariant: Whether the vector field has contravariant components
        
    Returns:
        Array of divergence values
    """
    import warnings
    n = len(grid)
    grid_shape = vector_field[0].shape
    
    # If vector is not contravariant, convert it
    if not is_contravariant:
        # Compute metric inverse for raising indices
        if metric.ndim == 2:  # constant metric
            # Check determinant before inversion
            det = np.linalg.det(metric)
            if abs(det) < 1e-10:
                warnings.warn(f"Singular constant metric detected (det={det}). Using pseudoinverse.")
                metric_inverse = np.linalg.pinv(metric)
            else:
                try:
                    metric_inverse = np.linalg.inv(metric)
                except np.linalg.LinAlgError as e:
                    warnings.warn(f"Error inverting constant metric: {e}. Using pseudoinverse.")
                    metric_inverse = np.linalg.pinv(metric)
                
            vector_contravariant = [np.zeros(grid_shape) for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    vector_contravariant[i] += metric_inverse[i, j] * vector_field[j]
        else:  # position-dependent metric
            metric_inverse = np.zeros(grid_shape + (n, n))
            for idx in np.ndindex(*grid_shape):
                try:
                    # Check determinant before inversion
                    det = np.linalg.det(metric[idx])
                    if abs(det) < 1e-10:
                        if not np.isclose(abs(det), 0):  # Only warn if not exactly zero
                            warnings.warn(f"Small determinant at grid point {idx}: {det}. Using regularized inverse.")
                        # Add small regularization for numerical stability
                        metric_at_point = metric[idx].copy()
                        for i in range(n):
                            metric_at_point[i, i] += 1e-10
                        metric_inverse[idx] = np.linalg.inv(metric_at_point)
                    else:
                        metric_inverse[idx] = np.linalg.inv(metric[idx])
                except np.linalg.LinAlgError as e:
                    warnings.warn(f"Inversion error at grid point {idx}: {e}. Using pseudoinverse.")
                    # Use pseudoinverse for more robust handling of singular matrices
                    metric_inverse[idx] = np.linalg.pinv(metric[idx])
            
            # Raise indices to get contravariant components
            vector_contravariant = [np.zeros(grid_shape) for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    vector_contravariant[i] += metric_inverse[..., i, j] * vector_field[j]
    else:
        vector_contravariant = vector_field
    
    # Compute metric determinant
    g_det = np.zeros(grid_shape)
    
    if metric.ndim == 2:  # constant metric
        g_det.fill(np.linalg.det(metric))
    else:  # position-dependent metric
        for idx in np.ndindex(*grid_shape):
            try:
                g_det[idx] = np.linalg.det(metric[idx])
            except:
                # If there's an error computing determinant, use a small positive value
                g_det[idx] = 1e-10
    
    # Ensure determinant is not too close to zero
    g_det = np.maximum(g_det, 1e-10)
    g_det_sqrt = np.sqrt(np.abs(g_det))
    
    # Compute divergence using the formula:
    # div V = (1/√|g|) ∂_i(√|g| V^i)
    divergence = np.zeros(grid_shape)
    
    for i in range(n):
        # Compute √|g| * V^i
        weighted_component = g_det_sqrt * vector_contravariant[i]
        
        # Compute partial derivative and add to divergence
        div_term = compute_partial_derivative(weighted_component, grid, i)
        divergence += div_term
    
    # Divide by √|g|, avoiding division by zero
    g_det_sqrt_safe = np.maximum(g_det_sqrt, 1e-10)
    divergence /= g_det_sqrt_safe
    
    return divergence

def evaluate_curl(vector_field: List[np.ndarray], metric: np.ndarray, grid: List[np.ndarray], 
                 christoffel: Optional[np.ndarray] = None, is_contravariant: bool = True) -> List[np.ndarray]:
    """
    Compute the curl of a vector field numerically (3D only).
    
    Args:
        vector_field: List of 3 arrays for vector components
        metric: Array of metric values at each point
        grid: List of arrays containing the coordinate values along each axis
        christoffel: Optional pre-computed Christoffel symbols
        is_contravariant: Whether the vector field has contravariant components
        
    Returns:
        List of 3 arrays for curl components (contravariant)
    """
    n = len(grid)
    
    if n != 3:
        raise ValueError("Curl operator is only defined in 3D space")
    
    grid_shape = vector_field[0].shape
    
    # Convert to covariant components if needed
    vector_covariant = [np.zeros(grid_shape) for _ in range(3)]
    
    if is_contravariant:
        # Lower indices: A_i = g_{ij} * V^j
        if metric.ndim == 2:  # constant metric
            for i in range(3):
                for j in range(3):
                    vector_covariant[i] += metric[i, j] * vector_field[j]
        else:  # position-dependent metric
            for i in range(3):
                for j in range(3):
                    vector_covariant[i] += metric[..., i, j] * vector_field[j]
    else:
        vector_covariant = vector_field.copy()
    
    # Compute covariant derivatives of the vector
    # ∇_j A_k = ∂_j A_k - Γ^m_{jk} A_m
    
    # First compute partial derivatives
    partials = [[compute_partial_derivative(vector_covariant[k], grid, j) 
                 for k in range(3)] for j in range(3)]
    
    # Compute Christoffel symbols if not provided
    if christoffel is None:
        if metric.ndim == 2:  # constant metric
            # For constant metric, Christoffel symbols are zero if it's Cartesian
            # Otherwise, they need to be computed
            if np.allclose(metric, np.eye(3)):
                christoffel = np.zeros(grid_shape + (3, 3, 3))
            else:
                # Create metric functions for constant metric
                def make_metric_func(i, j, metric_val):
                    def metric_func(point):
                        return metric_val[i, j]
                    return metric_func
                
                metric_funcs = [[make_metric_func(i, j, metric) for j in range(3)] for i in range(3)]
                christoffel = compute_christoffel_on_grid(metric_funcs, grid)
                
        else:  # position-dependent metric
            # Create metric functions from the array
            def make_metric_func(i, j):
                def metric_func(point):
                    # Find the closest grid point
                    indices = []
                    for d in range(3):
                        idx = np.abs(grid[d] - point[d]).argmin()
                        indices.append(idx)
                    return metric[tuple(indices)][i, j]
                return metric_func
            
            metric_funcs = [[make_metric_func(i, j) for j in range(3)] for i in range(3)]
            christoffel = compute_christoffel_on_grid(metric_funcs, grid)
    
    # Compute covariant derivatives
    cov_derivs = np.zeros(grid_shape + (3, 3))  # [x,y,z,...][j][k]
    
    # Iterate over all grid points
    for idx in np.ndindex(*grid_shape):
        for j in range(3):
            for k in range(3):
                cov_derivs[idx][j][k] = partials[j][k][idx]
                
                # Subtract Christoffel terms
                for m in range(3):
                    if christoffel.ndim == 3 + 3:  # constant Christoffel
                        cov_derivs[idx][j][k] -= christoffel[m, j, k] * vector_covariant[m][idx]
                    else:  # position-dependent Christoffel
                        cov_derivs[idx][j][k] -= christoffel[idx][m][j][k] * vector_covariant[m][idx]
    
    # Compute metric determinant for the Levi-Civita tensor
    g_det = np.zeros(grid_shape)
    
    if metric.ndim == 2:  # constant metric
        g_det.fill(np.linalg.det(metric))
    else:  # position-dependent metric
        for idx in np.ndindex(*grid_shape):
            g_det[idx] = np.linalg.det(metric[idx])
    
    # Ensure determinant is not too close to zero
    g_det = np.maximum(g_det, 1e-10)
    g_det_sqrt = np.sqrt(np.abs(g_det))
    
    # Compute curl using the formula:
    # (∇×V)^i = (1/√|g|) * ε^{ijk} * ∇_j A_k
    # where ε^{ijk} is the Levi-Civita symbol
    curl = [np.zeros(grid_shape) for _ in range(3)]
    
    # Define the Levi-Civita tensor in 3D
    def levi_civita(i, j, k):
        if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            return 1
        elif (i, j, k) in [(0, 2, 1), (2, 1, 0), (1, 0, 2)]:
            return -1
        else:
            return 0
    
    for idx in np.ndindex(*grid_shape):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    curl[i][idx] += levi_civita(i, j, k) * cov_derivs[idx][j][k] / g_det_sqrt[idx]
    
    return curl

def evaluate_laplacian(scalar_field: np.ndarray, metric: np.ndarray, metric_inverse: np.ndarray, 
                      grid: List[np.ndarray]) -> np.ndarray:
    """
    Compute the Laplacian of a scalar field numerically.
    
    Args:
        scalar_field: Array of scalar field values
        metric: Array of metric values at each point
        metric_inverse: Array of inverse metric values
        grid: List of arrays containing the coordinate values along each axis
        
    Returns:
        Array of Laplacian values
    """
    import warnings
    n = len(grid)
    grid_shape = scalar_field.shape
    
    # Compute metric determinant
    g_det = np.zeros(grid_shape)
    
    if metric.ndim == 2:  # constant metric
        try:
            det = np.linalg.det(metric)
            if abs(det) < 1e-10:
                warnings.warn(f"Singular constant metric detected (det={det}). Using 1e-10 instead.")
                g_det.fill(1e-10)
            else:
                g_det.fill(det)
        except Exception as e:
            warnings.warn(f"Error computing determinant of constant metric: {e}")
            g_det.fill(1e-10)
    else:  # position-dependent metric
        for idx in np.ndindex(*grid_shape):
            try:
                det = np.linalg.det(metric[idx])
                if abs(det) < 1e-10:
                    if not np.isclose(abs(det), 0):  # Only warn if not exactly zero
                        warnings.warn(f"Small determinant at grid point {idx}: {det}. Using 1e-10 instead.")
                    g_det[idx] = 1e-10
                else:
                    g_det[idx] = det
            except Exception as e:
                warnings.warn(f"Error computing determinant at grid point {idx}: {e}")
                g_det[idx] = 1e-10
    
    # Ensure determinant is not too close to zero
    g_det = np.maximum(g_det, 1e-10)
    g_det_sqrt = np.sqrt(np.abs(g_det))
    
    # Compute Laplacian using the formula:
    # ∇²f = (1/√|g|) ∂_i(√|g| g^{ij} ∂_j f)
    
    # First compute partial derivatives of f
    partial_f = [compute_partial_derivative(scalar_field, grid, j) for j in range(n)]
    
    # Compute g^{ij} ∂_j f for each i
    weighted_derivs = [np.zeros(grid_shape) for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if metric_inverse.ndim == 2:  # constant metric
                weighted_derivs[i] += metric_inverse[i, j] * partial_f[j]
            else:  # position-dependent metric
                weighted_derivs[i] += metric_inverse[..., i, j] * partial_f[j]
    
    # Multiply by √|g|
    for i in range(n):
        weighted_derivs[i] *= g_det_sqrt
    
    # Take derivatives of these weighted terms
    div_terms = [compute_partial_derivative(weighted_derivs[i], grid, i) for i in range(n)]
    
    # Sum and divide by √|g|
    laplacian = np.zeros(grid_shape)
    for i in range(n):
        laplacian += div_terms[i]
    
    # Divide by √|g|, avoiding division by zero
    g_det_sqrt_safe = np.maximum(g_det_sqrt, 1e-10)
    laplacian /= g_det_sqrt_safe
    
    return laplacian

def evaluate_dalembert(scalar_field: np.ndarray, metric: np.ndarray, metric_inverse: np.ndarray, 
                      grid: List[np.ndarray]) -> np.ndarray:
    """
    Compute the d'Alembertian (wave operator) of a scalar field numerically.
    
    Args:
        scalar_field: Array of scalar field values
        metric: Array of metric values at each point
        metric_inverse: Array of inverse metric values
        grid: List of arrays containing the coordinate values along each axis
        
    Returns:
        Array of d'Alembertian values
    """
    n = len(grid)
    
    if n != 4:
        raise ValueError("d'Alembertian operator is defined for 4D spacetime")
    
    # The d'Alembertian is the Laplacian generalized to 4D spacetime
    return evaluate_laplacian(scalar_field, metric, metric_inverse, grid)

def apply_boundary_condition(field: np.ndarray, grid: List[np.ndarray], 
                            boundary_type: str, boundary_values: Dict[str, Any] = None) -> np.ndarray:
    """
    Apply boundary conditions to a field on a grid.
    
    Args:
        field: Array of field values
        grid: List of arrays containing the coordinate values along each axis
        boundary_type: Type of boundary condition ('dirichlet', 'neumann', 'periodic')
        boundary_values: Boundary values for Dirichlet or derivative values for Neumann
        
    Returns:
        Updated field array with boundary conditions applied
    """
    n = len(grid)
    grid_shape = field.shape
    result = field.copy()
    
    # Process each boundary (two per dimension)
    for dim in range(n):
        # Get the appropriate slices for the boundaries in this dimension
        lower_slice = [slice(None)] * n
        upper_slice = [slice(None)] * n
        
        lower_slice[dim] = 0
        upper_slice[dim] = -1
        
        # Apply appropriate boundary condition
        if boundary_type.lower() == 'dirichlet':
            # Set fixed values at the boundaries
            if boundary_values is not None:
                if f'dim{dim}_lower' in boundary_values:
                    result[tuple(lower_slice)] = boundary_values[f'dim{dim}_lower']
                if f'dim{dim}_upper' in boundary_values:
                    result[tuple(upper_slice)] = boundary_values[f'dim{dim}_upper']
        
        elif boundary_type.lower() == 'neumann':
            # Set fixed derivatives at the boundaries
            # For simplicity, use first-order forward/backward differences
            dx = grid[dim][1] - grid[dim][0]
            
            one_in_slice = [slice(None)] * n
            one_in_slice[dim] = 1
            
            last_but_one_slice = [slice(None)] * n
            last_but_one_slice[dim] = -2
            
            if boundary_values is not None:
                if f'dim{dim}_lower' in boundary_values:
                    # f[0] = f[1] - dx * derivative
                    result[tuple(lower_slice)] = result[tuple(one_in_slice)] - dx * boundary_values[f'dim{dim}_lower']
                
                if f'dim{dim}_upper' in boundary_values:
                    # f[-1] = f[-2] + dx * derivative
                    result[tuple(upper_slice)] = result[tuple(last_but_one_slice)] + dx * boundary_values[f'dim{dim}_upper']
        
        elif boundary_type.lower() == 'periodic':
            # Copy values from opposite boundaries
            result[tuple(lower_slice)] = result[tuple(upper_slice)]
            
            second_to_last_slice = [slice(None)] * n
            second_to_last_slice[dim] = -2
            
            first_slice = [slice(None)] * n
            first_slice[dim] = 1
            
            result[tuple(upper_slice)] = result[tuple(first_slice)]
            
            # Also update the second-to-last point for better derivative calculation
            result[tuple(second_to_last_slice)] = result[tuple(lower_slice)]
        
        else:
            raise ValueError(f"Unsupported boundary condition type: {boundary_type}")
    
    return result

def interpolate_field(field: np.ndarray, grid: List[np.ndarray], point: List[float]) -> float:
    """
    Interpolate a field value at an arbitrary point within the grid.
    
    Args:
        field: Array of field values
        grid: List of arrays containing the coordinate values along each axis
        point: List of coordinate values at which to interpolate
        
    Returns:
        Interpolated field value at the specified point
    """
    n = len(grid)
    
    # Find the indices of the surrounding grid points
    indices = []
    weights = []
    
    for i in range(n):
        # Find the index of the closest grid point below the target
        idx = np.searchsorted(grid[i], point[i]) - 1
        idx = max(0, min(idx, len(grid[i]) - 2))  # Ensure within bounds
        
        # Calculate the fractional distance between grid points
        x0 = grid[i][idx]
        x1 = grid[i][idx + 1]
        w = (point[i] - x0) / (x1 - x0)  # Weight for linear interpolation
        
        indices.append((idx, idx + 1))
        weights.append((1 - w, w))
    
    # Perform multilinear interpolation
    result = 0.0
    
    # Iterate over all combinations of surrounding grid points
    for idx_tuple in np.ndindex(*[(2,)] * n):
        # Compute the weight for this combination
        weight_product = 1.0
        for dim, (idx_choice, weight_choice) in enumerate(zip(idx_tuple, weights)):
            weight_product *= weight_choice[idx_choice]
        
        # Compute the field value at this grid point
        grid_indices = [indices[dim][idx_choice] for dim, idx_choice in enumerate(idx_tuple)]
        field_value = field[tuple(grid_indices)]
        
        # Add weighted contribution
        result += weight_product * field_value
    
    return result

if __name__ == "__main__":
    print("Numerical Differential Operators - Demo\n")
    
    # Example 1: Create a grid in spherical coordinates
    print("1. Creating a grid in spherical coordinates...")
    coords_ranges = {
        'r': {'min': 1.0, 'max': 2.0},
        'theta': {'min': 0.0, 'max': np.pi},
        'phi': {'min': 0.0, 'max': 2 * np.pi}
    }
    dimensions = [10, 10, 10]  # 10 points in each dimension
    
    grid, spacing = create_grid(coords_ranges, dimensions)
    print(f"Grid created with dimensions: {[len(g) for g in grid]}")
    print(f"Spacing: {spacing}")
    
    # Example 2: Define a simple scalar field and discretize it
    print("\n2. Discretizing a scalar field on the grid...")
    
    # Define a simple scalar field: f(r, theta, phi) = r^2 * cos(theta)
    def scalar_field_func(point):
        r, theta, phi = point
        return r**2 * np.cos(theta)
    
    # Discretize the field on the grid
    scalar_field = discretize_field(scalar_field_func, grid)
    print(f"Scalar field shape: {scalar_field.shape}")
    print(f"Value at [0,0,0]: {scalar_field[0,0,0]}")
    
    # Example 3: Compute metric tensor for spherical coordinates
    print("\n3. Computing metric tensor for spherical coordinates...")
    
    # Define the metric tensor for spherical coordinates
    def metric_func(point):
        r, theta, phi = point
        g = np.zeros((3, 3))
        g[0, 0] = 1.0
        g[1, 1] = r**2
        g[2, 2] = r**2 * np.sin(theta)**2
        return g
    
    # Initialize metric tensor on the grid
    metric = np.zeros(scalar_field.shape + (3, 3))
    for i in range(len(grid[0])):
        for j in range(len(grid[1])):
            for k in range(len(grid[2])):
                r = grid[0][i]
                theta = grid[1][j]
                phi = grid[2][k]
                metric[i, j, k] = metric_func([r, theta, phi])
    
    # Compute inverse metric
    metric_inverse = np.zeros_like(metric)
    for i in range(len(grid[0])):
        for j in range(len(grid[1])):
            for k in range(len(grid[2])):
                metric_inverse[i, j, k] = np.linalg.inv(metric[i, j, k])
    
    print("Metric at [0,0,0]:")
    print(metric[0,0,0])
    
    # Example 4: Compute gradient of the scalar field
    print("\n4. Computing gradient of the scalar field...")
    grad = evaluate_gradient(scalar_field, metric_inverse, grid)
    print(f"Gradient shape: {len(grad)} components of shape {grad[0].shape}")
    print("Gradient at [0,0,0]:")
    for i in range(3):
        print(f"∇f[{i}] = {grad[i][0,0,0]}")
    
    # Example 5: Compute divergence of a vector field
    print("\n5. Computing divergence of a simple vector field...")
    
    # Define a simple vector field: V = [r, 0, 0]
    vector_field = [np.zeros_like(scalar_field) for _ in range(3)]
    for i in range(len(grid[0])):
        for j in range(len(grid[1])):
            for k in range(len(grid[2])):
                vector_field[0][i, j, k] = grid[0][i]  # r component
    
    div = evaluate_divergence(vector_field, metric, grid)
    print(f"Divergence shape: {div.shape}")
    print(f"Divergence at [0,0,0]: {div[0,0,0]}")
    
    # Example 6: Compute Laplacian of the scalar field
    print("\n6. Computing Laplacian of the scalar field...")
    laplacian = evaluate_laplacian(scalar_field, metric, metric_inverse, grid)
    print(f"Laplacian shape: {laplacian.shape}")
    print(f"Laplacian at [0,0,0]: {laplacian[0,0,0]}")
    
    print("\nTo see more examples and operations, modify the code in numeric.py") 