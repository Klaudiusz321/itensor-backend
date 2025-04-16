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
    Create a grid in the specified coordinate system with robust singularity handling.
    
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
    
    # Log grid creation details
    logger.info(f"Creating grid with dimensions: {dimensions}")
    logger.info(f"Coordinate ranges: {coords_ranges}")
    
    # Check and validate dimensions
    if len(coords_ranges) != len(dimensions):
        logger.warning(f"Mismatch between coordinate ranges ({len(coords_ranges)}) "
                      f"and dimensions ({len(dimensions)})")
    
    # Check for known coordinate system types based on coordinate names
    coord_names = list(coords_ranges.keys())
    
    # Check for spherical-like coordinates
    is_spherical = any('r' in c.lower() for c in coord_names) and \
                  any('theta' in c.lower() or 'θ' in c for c in coord_names) and \
                  any('phi' in c.lower() or 'φ' in c for c in coord_names)
    
    # Check for cylindrical-like coordinates
    is_cylindrical = any('r' in c.lower() for c in coord_names) and \
                    any('phi' in c.lower() or 'φ' in c for c in coord_names) and \
                    any('z' in c.lower() for c in coord_names)
    
    if is_spherical:
        logger.info("Detected spherical-like coordinate system")
    elif is_cylindrical:
        logger.info("Detected cylindrical-like coordinate system")
    else:
        logger.info("Using general coordinate system")
    
    # Set safety margins for singularity handling
    SAFETY_MARGIN = 1e-5
    
    for i, (coord_name, coord_range) in enumerate(coords_ranges.items()):
        min_val = coord_range['min']
        max_val = coord_range['max']
        
        # Ensure we have enough dimensions for this coordinate
        n_points = dimensions[i] if i < len(dimensions) else dimensions[-1]
        
        # Singularity handling with detailed logging
        if coord_name.lower() == 'r':
            # Ensure r is not zero (avoiding coordinate singularity)
            if min_val <= 0:
                original_min = min_val
                min_val = SAFETY_MARGIN
                logger.warning(f"Minimum value for r must be positive. Adjusting from {original_min} to {min_val}")
        
        elif coord_name.lower() in ['theta', 'θ']:
            # Avoid theta = 0 and theta = pi (to avoid sin(theta) = 0)
            if min_val <= 0:
                original_min = min_val
                min_val = SAFETY_MARGIN
                logger.warning(f"Minimum value for theta must be positive. Adjusting from {original_min} to {min_val}")
            if max_val >= np.pi:
                original_max = max_val
                max_val = np.pi - SAFETY_MARGIN
                logger.warning(f"Maximum value for theta must be less than π. Adjusting from {original_max} to {max_val}")
        
        # Special handling for angular coordinates
        if coord_name.lower() in ['phi', 'φ', 'theta', 'θ']:
            # Ensure periodic boundary conditions are respected
            if coord_name.lower() in ['phi', 'φ'] and (max_val - min_val) > 2*np.pi + SAFETY_MARGIN:
                logger.warning(f"Phi range exceeds 2π: [{min_val}, {max_val}]. Consider using a 2π range.")
            
            # For periodic coordinates, ensure we don't double-count endpoints
            if coord_name.lower() in ['phi', 'φ'] and np.isclose(max_val - min_val, 2*np.pi, rtol=1e-3):
                # Use endpoint=False to avoid double-counting the endpoints for full periodic range
                logger.info(f"Using periodic grid for {coord_name} (not including endpoint)")
                coord_points = np.linspace(min_val, max_val, n_points, endpoint=False)
            else:
                coord_points = np.linspace(min_val, max_val, n_points)
        else:
            # Standard coordinates
            coord_points = np.linspace(min_val, max_val, n_points)
        
        grid_points.append(coord_points)
        
        # Calculate spacing
        spacing[coord_name] = (max_val - min_val) / (n_points - 1)
        
        # Log grid details
        logger.info(f"Created {coord_name} grid with {n_points} points from {min_val} to {max_val}")
        logger.info(f"  {coord_name} spacing: {spacing[coord_name]}")
        
        # Additional checks for very small or very large grid spacings
        if spacing[coord_name] < 1e-10:
            logger.warning(f"Very small grid spacing for {coord_name}: {spacing[coord_name]}")
        elif spacing[coord_name] > 1e3:
            logger.warning(f"Very large grid spacing for {coord_name}: {spacing[coord_name]}")
    
    # Verify grid quality and provide diagnostics
    grid_size = 1
    for dim in dimensions:
        grid_size *= dim
    
    if grid_size > 10**7:
        logger.warning(f"Very large grid with {grid_size} total points. This may cause memory issues.")
    
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
    # Check dimensions and add debug information
    field_shape = field.shape
    expected_ndim = len(grid)
    
    # Print debug information about the field and grid
    logger.debug(f"Field shape: {field_shape}, Grid dimensions: {expected_ndim}, Direction: {direction}")
    
    # Handle case where field has fewer dimensions than expected
    if field.ndim < expected_ndim:
        logger.warning(f"Field has {field.ndim} dimensions but expected at least {expected_ndim}. " 
                      f"This may cause issues when computing derivatives along axis {direction}.")
        
        # Reshape the field to match the expected dimensions if needed
        if direction >= field.ndim:
            logger.info(f"Reshaping field with shape {field_shape} to have at least {direction+1} dimensions")
            
            # Create a new shape with the right number of dimensions
            new_shape = list(field_shape) + [1] * (direction + 1 - field.ndim)
            reshaped_field = field.reshape(new_shape)
            logger.info(f"Reshaped field to {reshaped_field.shape}")
            
            # Use the reshaped field for derivative computation
            field = reshaped_field
    
    # Get the coordinate spacing
    dx = grid[direction][1] - grid[direction][0]
    
    # Use np.gradient for central differences (2nd order accurate)
    if order == 2:
        # Using numpy's gradient which handles non-uniform spacing
        try:
            # When computing gradient along a specific axis, we only need to provide
            # the coordinate array for that axis, not all axes
            grad = np.gradient(field, grid[direction], axis=direction, edge_order=2)
            return grad
        except np.exceptions.AxisError as e:
            # Handle the axis error more gracefully
            logger.error(f"Axis error when computing gradient: {e}")
            logger.error(f"Field shape: {field.shape}, attempting derivative along axis {direction}")
            
            # If the field has fewer dimensions than direction, reshape it
            if field.ndim <= direction:
                # Create a new shape with enough dimensions
                new_shape = [1] * (direction + 1)
                if field.ndim > 0:
                    for i in range(field.ndim):
                        new_shape[i] = field.shape[i]
                
                reshaped_field = field.reshape(new_shape)
                logger.warning(f"Reshaped field from {field.shape} to {reshaped_field.shape} for axis {direction}")
                
                # Try again with the reshaped field
                try:
                    grad = np.gradient(reshaped_field, grid[direction], axis=direction, edge_order=2)
                    return grad
                except Exception as reshape_error:
                    logger.error(f"Error after reshaping: {reshape_error}")
            
            # If still failing after reshape or for other reasons
            try:
                # Fall back to simpler method - compute along existing dimensions only
                if field.ndim > 0 and direction < field.ndim:
                    return np.gradient(field, grid[direction], axis=direction)
                else:
                    # For scalar or completely incompatible dimensions
                    return np.zeros_like(field)
            except Exception as fallback_error:
                logger.error(f"Fallback gradient method failed: {fallback_error}")
                return np.zeros_like(field)
    
    # For 1st order, implement forward and backward differences
    elif order == 1:
        # Forward difference at the beginning, backward at the end, central in the middle
        result = np.zeros_like(field)
        
        # Handle 1D case differently
        if field.ndim == 1 and direction == 0:
            # Forward difference for first point
            result[0] = (field[1] - field[0]) / dx
            
            # Central difference for middle points
            result[1:-1] = (field[2:] - field[:-2]) / (2 * dx)
            
            # Backward difference for last point
            result[-1] = (field[-1] - field[-2]) / dx
            
            return result
        
        # For multi-dimensional arrays, use slicing with tuple indexing
        try:
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
        except (IndexError, ValueError) as e:
            logger.error(f"Error in 1st order derivative: {e}")
            logger.error(f"Field shape: {field.shape}, direction: {direction}")
            
            # Fall back to zeros if all else fails
            return np.zeros_like(field)
    
    else:
        raise ValueError(f"Order {order} not supported for finite differences")

def compute_christoffel_on_grid(metric_funcs: List[List[Callable]], grid: List[np.ndarray]) -> np.ndarray:
    """
    Compute Christoffel symbols on a grid with robust handling of near-singular metrics.
    
    Args:
        metric_funcs: List of callables for metric components
        grid: List of arrays containing the coordinate values along each axis
        
    Returns:
        Array of Christoffel symbols at each grid point
    """
    import warnings
    from numpy.linalg import LinAlgError
    
    n = len(grid)
    
    # Create meshgrid for evaluation
    mesh = np.meshgrid(*grid, indexing='ij')
    grid_shape = [len(g) for g in grid]
    
    # Log basic information about the calculation
    logger.info(f"Computing Christoffel symbols on grid with shape {grid_shape}")
    
    # Check that metric_funcs has the right dimensions
    if len(metric_funcs) != n or any(len(row) != n for row in metric_funcs):
        logger.error(f"Metric functions array has incorrect dimensions: {len(metric_funcs)}x{len(metric_funcs[0] if metric_funcs else 0)}, expected {n}x{n}")
        logger.warning("Attempting to continue with available metric components")
    
    # Try to get a sample metric value to check if it's a constant metric
    try:
        sample_point = [grid[d][0] for d in range(n)]
        sample_metric = np.zeros((n, n))
        
        # Track which components are callable vs. constants
        component_types = np.zeros((n, n), dtype=int)  # 0 for constant, 1 for callable
        
        for i in range(n):
            for j in range(n):
                if i < len(metric_funcs) and j < len(metric_funcs[i]):
                    if callable(metric_funcs[i][j]):
                        sample_metric[i, j] = metric_funcs[i][j](sample_point)
                        component_types[i, j] = 1
                    else:
                        # If not callable, assume it's a constant value
                        try:
                            sample_metric[i, j] = float(metric_funcs[i][j])
                        except (TypeError, ValueError):
                            logger.warning(f"Metric component [{i},{j}] is neither callable nor convertible to float, using 0.0")
                            sample_metric[i, j] = 0.0
                else:
                    # Handle missing components gracefully
                    logger.warning(f"Metric component [{i},{j}] is missing, using 0.0")
                    sample_metric[i, j] = 0.0
        
        # Ensure metric is symmetric
        if not np.allclose(sample_metric, sample_metric.T, rtol=1e-5):
            logger.warning("Metric tensor is not symmetric at sample point. This may cause incorrect results.")
            # Make it symmetric by averaging
            sample_metric = 0.5 * (sample_metric + sample_metric.T)
        
        # Check if all metric functions return constant values
        is_constant_metric = True
        num_test_points = min(3, min(len(g) for g in grid))  # Test at most 3 points
        
        for idx in range(num_test_points):
            idx_scaled = idx * (len(grid[0]) - 1) // (num_test_points - 1)  # Distribute across grid
            test_point = [grid[d][min(idx_scaled, len(grid[d])-1)] for d in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i < len(metric_funcs) and j < len(metric_funcs[i]) and callable(metric_funcs[i][j]):
                        try:
                            metric_val = metric_funcs[i][j](test_point)
                            if abs(metric_val - sample_metric[i, j]) > 1e-8:
                                is_constant_metric = False
                                break
                        except Exception as e:
                            logger.warning(f"Error evaluating metric function at {test_point}: {e}")
                            is_constant_metric = False
                            break
                if not is_constant_metric:
                    break
            if not is_constant_metric:
                break
        
        # Check metric determinant
        try:
            det = np.linalg.det(sample_metric)
            if abs(det) < 1e-10:
                logger.warning(f"Near-singular metric detected at sample point (det={det:.2e})")
                
                # Analyze eigenvalues to understand the singularity
                try:
                    eigenvals = np.linalg.eigvals(sample_metric)
                    logger.info(f"Metric eigenvalues at sample point: {eigenvals}")
                    min_eigval = np.min(np.abs(eigenvals))
                    if min_eigval < 1e-10:
                        logger.warning(f"Small eigenvalue detected: {min_eigval:.2e}")
                except Exception as e:
                    logger.warning(f"Could not compute eigenvalues: {e}")
                
                # Attempt to regularize if needed
                if abs(det) < 1e-15:
                    logger.info("Regularizing sample metric to avoid singularity")
                    for i in range(n):
                        sample_metric[i, i] += 1e-8
                    det = np.linalg.det(sample_metric)
                    logger.info(f"After regularization, det = {det:.2e}")
        except LinAlgError as e:
            logger.warning(f"Linear algebra error with sample metric: {e}")
            is_constant_metric = False
        
        if is_constant_metric:
            logger.info("Detected constant metric tensor - using optimized computation")
            # For constant metric, simplified computation
            metric = sample_metric
            
            try:
                det = np.linalg.det(metric)
                if abs(det) < 1e-10:
                    logger.warning(f"Singular metric detected (det={det:.2e}). Using pseudoinverse.")
                    metric_inv = np.linalg.pinv(metric)
                else:
                    try:
                        metric_inv = np.linalg.inv(metric)
                    except LinAlgError:
                        # Add a small regularization if needed
                        logger.warning("Error inverting metric. Using pseudoinverse.")
                        metric_inv = np.linalg.pinv(metric)
                
                # For Cartesian coordinates with identity metric, all Christoffel symbols are zero
                if np.allclose(metric, np.eye(n), rtol=1e-5, atol=1e-5):
                    logger.info("Identity metric detected, all Christoffel symbols are zero")
                    return np.zeros(grid_shape + (n, n, n))
                
                # For constant non-Cartesian metric, compute derivatives
                metric_deriv = np.zeros((n, n, n))
                
                # For constant metric, all derivatives should be zero
                # Initialize Christoffel symbols array
                Gamma = np.zeros(grid_shape + (n, n, n))
                
                for k in range(n):
                    for i in range(n):
                        for j in range(n):
                            # For constant metric, Christoffel symbols are constant everywhere
                            gamma_value = 0.0  # Will remain zero for constant metric
                            
                            # Fill the entire grid with the same value
                            for idx in np.ndindex(*grid_shape):
                                Gamma[idx][k][i][j] = gamma_value
                
                logger.info("Constant metric computation complete")
                return Gamma
            
            except Exception as e:
                logger.warning(f"Error in constant metric computation: {e}")
                logger.info("Falling back to position-dependent computation")
                # Continue with regular computation
    
    except Exception as e:
        logger.warning(f"Error checking for constant metric: {e}")
        logger.info("Proceeding with position-dependent metric computation")
    
    # Position-dependent metric computation
    logger.info("Computing position-dependent Christoffel symbols")
    
    # Initialize arrays for metric and its derivatives
    metric = np.zeros(grid_shape + (n, n))
    metric_inv = np.zeros_like(metric)
    metric_deriv = np.zeros(grid_shape + (n, n, n))  # Additional index for derivative direction
    
    # Track metrics that couldn't be inverted
    inversion_failures = 0
    regularization_count = 0
    
    # Compute metric at each grid point
    for idx in np.ndindex(*grid_shape):
        point = [grid[d][idx[d]] for d in range(n)]
        
        # Evaluate metric components at this point
        g_point = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    if i < len(metric_funcs) and j < len(metric_funcs[i]):
                        if callable(metric_funcs[i][j]):
                            g_point[i, j] = metric_funcs[i][j](point)
                        else:
                            # Use the constant value
                            g_point[i, j] = metric_funcs[i][j]
                    else:
                        # Use delta_ij (identity) for missing components
                        g_point[i, j] = 1.0 if i == j else 0.0
                except Exception as e:
                    # If there's an error evaluating the metric, use a reasonable default
                    logger.warning(f"Error evaluating metric[{i},{j}] at {point}: {e}")
                    g_point[i, j] = 1.0 if i == j else 0.0
        
        # Ensure metric is symmetric
        g_point = 0.5 * (g_point + g_point.T)
        
        # Check if the metric is singular and use appropriate inversion method
        try:
            det = np.linalg.det(g_point)
            
            # Debug info for first few points and problematic points
            if idx == (0,) * len(grid_shape) or abs(det) < 1e-10:
                eigenvals = np.linalg.eigvals(g_point)
                logger.debug(f"Metric at {idx}: det={det:.2e}, eigenvalues={eigenvals}")
            
            if abs(det) < 1e-10:
                # Try to regularize first
                regularized = g_point.copy()
                for i in range(n):
                    regularized[i, i] += 1e-8
                
                reg_det = np.linalg.det(regularized)
                if reg_det > 1e-10:
                    logger.debug(f"Regularized metric at {idx}: det improved from {det:.2e} to {reg_det:.2e}")
                    metric[idx] = regularized
                    metric_inv[idx] = np.linalg.inv(regularized)
                    regularization_count += 1
                else:
                    # If regularization doesn't help enough, use pseudoinverse
                    logger.warning(f"Singular matrix at grid point {idx} (det={det:.2e}). Using pseudoinverse.")
                    metric[idx] = g_point
                    metric_inv[idx] = np.linalg.pinv(g_point)
                    inversion_failures += 1
            else:
                metric[idx] = g_point
                metric_inv[idx] = np.linalg.inv(g_point)
        except LinAlgError as e:
            inversion_failures += 1
            logger.warning(f"Inversion error at grid point {idx}: {e}")
            metric[idx] = g_point
            # Use pseudoinverse for more robust handling of singular or near-singular matrices
            metric_inv[idx] = np.linalg.pinv(g_point)
    
    # Log summary of metric computation
    if inversion_failures > 0:
        logger.warning(f"{inversion_failures} grid points required pseudoinverse (out of {np.prod(grid_shape)})")
    if regularization_count > 0:
        logger.info(f"{regularization_count} grid points were regularized to avoid singularity")
    
    # Compute metric derivatives using finite differences
    for d in range(n):  # Derivative direction
        for i in range(n):
            for j in range(n):
                metric_component = metric[..., i, j]
                try:
                    metric_deriv[..., i, j, d] = compute_partial_derivative(metric_component, grid, d)
                except Exception as e:
                    logger.error(f"Error computing metric derivative [{i},{j}] in direction {d}: {e}")
                    # Use zeros as fallback
                    metric_deriv[..., i, j, d] = np.zeros_like(metric_component)
    
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
    
    # Verify Christoffel symbols
    # Check for NaN or Inf values
    if np.any(~np.isfinite(Gamma)):
        logger.error("Non-finite values detected in Christoffel symbols")
        # Replace NaN/Inf with zeros
        Gamma = np.nan_to_num(Gamma)
    
    # Check for unusually large values that might indicate numerical issues
    max_abs_gamma = np.max(np.abs(Gamma))
    if max_abs_gamma > 1e6:
        logger.warning(f"Very large Christoffel symbol detected: {max_abs_gamma:.2e}")
    
    logger.info("Position-dependent Christoffel symbol computation complete")
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
    orig_field_shape = scalar_field.shape
    
    # Create a 1D version of the grid for 1D fields
    grid_1d = [grid[0]]
    
    # Check if field dimensions match grid dimensions
    if scalar_field.ndim < n:
        logger.warning(f"Scalar field has {scalar_field.ndim} dimensions but grid has {n} dimensions")
        logger.info(f"Original field shape: {orig_field_shape}")
        
        # For 1D field with higher dimensional grid, handle specially
        if scalar_field.ndim == 1 and n > 1:
            # Option 1: Match with broadcast-compatible shape (more efficient)
            # Compute just the derivatives along the first dimension
            logger.info(f"Computing gradient for 1D field along first dimension only")
            
            # Initialize gradient components with zeros
            gradient = [np.zeros(orig_field_shape) if i == 0 else np.zeros(1) for i in range(n)]
            
            # Compute partial derivative only along first dimension
            try:
                # Use only the first dimension of the grid
                dx_1d = compute_partial_derivative(scalar_field, grid_1d, 0)
                gradient[0] = dx_1d
                
                # Contract with metric inverse (first row only) since other components are zero
                if metric_inverse.ndim == 2:
                    # If constant metric, apply first row to the gradient
                    for j in range(n):
                        if j == 0:  # Only the first partial derivative is non-zero
                            gradient[0] = metric_inverse[0, 0] * dx_1d
                        else:
                            # Other components remain zero
                            pass
                else:
                    # For position-dependent metric, this is more complex
                    # We'll leave this as zeros for now
                    logger.warning("Position-dependent metric with 1D field not fully supported")
                
                logger.info(f"Gradient computed for 1D field. Component shapes: {[g.shape for g in gradient]}")
                return gradient
            except Exception as e:
                logger.error(f"Error computing 1D gradient: {e}")
                # Continue to general approach
        
        # Option 2: Reshape field to full dimensionality (more general but potentially misleading)
        logger.info(f"Attempting to reshape field to match grid dimensions")
        
        # Create meshgrid indices to match grid shape
        grid_shape = tuple([len(g) for g in grid])
        
        if np.prod(orig_field_shape) == 1:
            # If scalar, broadcast to full grid shape
            scalar_field_full = np.full(grid_shape, scalar_field.item())
        elif scalar_field.ndim == 1 and len(scalar_field) == len(grid[0]):
            # If 1D array matching first dimension, broadcast along other dimensions
            scalar_field_full = np.zeros(grid_shape)
            
            # Create arrays for properly indexing the full grid
            indices = np.meshgrid(*[np.arange(len(g)) for g in grid], indexing='ij')
            
            # Fill the full field using broadcasting
            for idx in np.ndindex(grid_shape):
                # Use the value from the original field's first dimension
                i = idx[0]  # Index along first dimension
                if i < len(scalar_field):
                    scalar_field_full[idx] = scalar_field[i]
                else:
                    scalar_field_full[idx] = 0.0
        else:
            # Generic case - create a field with the right shape but possibly nonsensical values
            scalar_field_full = np.zeros(grid_shape)
            
            logger.warning(f"Unable to meaningfully reshape {orig_field_shape} to {grid_shape}")
            logger.warning("This may produce incorrect gradient values")
            
            # Fill with zeros to avoid errors, but results may not be meaningful
        
        logger.info(f"Reshaped/broadcast field to {scalar_field_full.shape}")
        scalar_field = scalar_field_full
    
    # Now proceed with the regular gradient calculation
    grid_shape = scalar_field.shape
    
    # Compute partial derivatives
    try:
        partial_derivs = [compute_partial_derivative(scalar_field, grid, d) for d in range(n)]
    except Exception as e:
        logger.error(f"Error computing partial derivatives: {e}")
        # Fall back to zeros
        return [np.zeros(grid_shape) for _ in range(n)]

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
                    try:
                        grad_i_at_point += g_inv[i, j] * partial_derivs[j][idx]
                    except IndexError:
                        # If we can't index into partial_derivs correctly, use the first value as fallback
                        warnings.warn(f"Error accessing derivative at index {idx}. Using first value as fallback.")
                        grad_i_at_point += g_inv[i, j] * partial_derivs[j].flat[0]
                
                gradient[i][idx] = grad_i_at_point
    
    # If original field was 1D and we want to maintain consistent dimensions
    if len(orig_field_shape) == 1 and orig_field_shape != scalar_field.shape:
        logger.info(f"Reshaping gradient components back to match original field shape: {orig_field_shape}")
        
        # Only keep the first component for 1D fields
        # Reshape first component back to original shape
        first_component = gradient[0]
        
        # Extract just the values along the first dimension for the first component
        if first_component.ndim > 1:
            # Take values along the first axis, with zeros elsewhere
            idx = (slice(None),) + (0,) * (first_component.ndim - 1)
            first_component = first_component[idx]
            
        # Truncate if needed to match original length
        if len(first_component) > orig_field_shape[0]:
            first_component = first_component[:orig_field_shape[0]]
            
        # For other components, just return zeros
        gradient = [first_component]
        for i in range(1, n):
            gradient.append(np.zeros(1))
    
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
    grid_shape = [len(g) for g in grid]
    
    # Check and reshape vector field components if needed
    reshaped_vector_field = []
    for i, component in enumerate(vector_field):
        comp_shape = component.shape
        
        # Log information about component shape
        logger.debug(f"Vector component {i} shape: {comp_shape}, grid dimensions: {grid_shape}")
        
        if component.ndim < n:
            logger.warning(f"Vector component {i} has {component.ndim} dimensions but grid has {n} dimensions")
            
            # For 1D field with higher dimensional grid, reshape to be compatible
            if component.ndim == 1:
                # Check if first dimension is compatible
                if len(component) == grid_shape[0]:
                    # Create a broadcast-compatible shape
                    broadcast_shape = list(grid_shape)
                    broadcast_shape[0] = len(component)
                    
                    # Reshape to add singleton dimensions
                    new_shape = [len(component)] + [1] * (n - 1)
                    reshaped = component.reshape(new_shape)
                    
                    # Broadcast to fill grid
                    try:
                        broadcasted = np.broadcast_to(reshaped, broadcast_shape)
                        logger.info(f"Reshaped and broadcast component {i} from {comp_shape} to {broadcast_shape}")
                        reshaped_vector_field.append(broadcasted)
                        continue
                    except Exception as e:
                        logger.error(f"Failed to broadcast component: {e}")
            
            # Generic approach for other dimension mismatches
            try:
                # Create a new array of the right shape filled with zeros
                full_shape_component = np.zeros(grid_shape, dtype=component.dtype)
                
                # Copy values where possible
                if component.ndim == 1 and len(component) <= grid_shape[0]:
                    # Fill first dimension, zeros elsewhere
                    slices = [slice(None)] + [0] * (n - 1)
                    full_shape_component[tuple(slices)[:len(component)]] = component
                else:
                    # Try to copy values from the original component
                    slice_list = []
                    for d in range(min(component.ndim, n)):
                        if d < component.ndim and d < len(grid_shape):
                            dim_size = min(component.shape[d], grid_shape[d])
                            slice_list.append(slice(0, dim_size))
                        else:
                            slice_list.append(0)
                    
                    target_slices = tuple(slice_list)
                    source_slices = tuple(slice_list[:component.ndim])
                    full_shape_component[target_slices] = component[source_slices]
                
                logger.info(f"Reshaped component {i} from {comp_shape} to {grid_shape}")
                reshaped_vector_field.append(full_shape_component)
            except Exception as e:
                logger.error(f"Error reshaping component {i}: {e}")
                # If all else fails, use the original component
                reshaped_vector_field.append(component)
        else:
            # Component already has the right number of dimensions
            if component.shape != tuple(grid_shape):
                logger.warning(f"Component {i} shape {component.shape} doesn't match grid shape {grid_shape}")
                # Try to reshape or pad if possible
                try:
                    if np.prod(component.shape) == np.prod(grid_shape):
                        # If same number of elements, reshape
                        reshaped = component.reshape(grid_shape)
                        logger.info(f"Reshaped component {i} to match grid shape")
                        reshaped_vector_field.append(reshaped)
                    else:
                        # If different number of elements, we can't simply reshape
                        reshaped_vector_field.append(component)
                except Exception as e:
                    logger.error(f"Failed to reshape component {i}: {e}")
                    reshaped_vector_field.append(component)
            else:
                reshaped_vector_field.append(component)
    
    # Use the reshaped vector field from now on
    grid_shape = reshaped_vector_field[0].shape
    
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
                    vector_contravariant[i] += metric_inverse[i, j] * reshaped_vector_field[j]
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
                    vector_contravariant[i] += metric_inverse[..., i, j] * reshaped_vector_field[j]
    else:
        vector_contravariant = reshaped_vector_field
    
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
        try:
            div_term = compute_partial_derivative(weighted_component, grid, i)
            divergence += div_term
        except Exception as e:
            logger.error(f"Error computing partial derivative for component {i}: {e}")
            # Try a more robust approach if the standard method fails
            try:
                # For 1D grids, use simple finite differences
                if len(grid[i]) > 2:
                    # Get the coordinate spacing
                    dx = grid[i][1] - grid[i][0]
                    
                    # Create slices for neighboring points
                    slices_before = [slice(None)] * weighted_component.ndim
                    slices_at = [slice(None)] * weighted_component.ndim
                    slices_after = [slice(None)] * weighted_component.ndim
                    
                    # Central difference for interior points
                    slices_before[i] = slice(0, -2)
                    slices_at[i] = slice(1, -1)
                    slices_after[i] = slice(2, None)
                    
                    interior_derivative = (weighted_component[tuple(slices_after)] - 
                                          weighted_component[tuple(slices_before)]) / (2 * dx)
                    
                    # Forward/backward differences for boundaries
                    boundary_derivative = np.zeros_like(weighted_component)
                    
                    # Forward difference for first point
                    slices_at[i] = 0
                    slices_after[i] = 1
                    boundary_derivative[tuple(slices_at)] = (weighted_component[tuple(slices_after)] - 
                                                            weighted_component[tuple(slices_at)]) / dx
                    
                    # Backward difference for last point
                    slices_before[i] = -2
                    slices_at[i] = -1
                    boundary_derivative[tuple(slices_at)] = (weighted_component[tuple(slices_at)] - 
                                                            weighted_component[tuple(slices_before)]) / dx
                    
                    # Combine interior and boundary derivatives
                    div_term = np.zeros_like(weighted_component)
                    interior_slices = [slice(None)] * weighted_component.ndim
                    interior_slices[i] = slice(1, -1)
                    div_term[tuple(interior_slices)] = interior_derivative
                    
                    # Add boundary derivatives
                    boundary_slices_first = [slice(None)] * weighted_component.ndim
                    boundary_slices_first[i] = 0
                    boundary_slices_last = [slice(None)] * weighted_component.ndim
                    boundary_slices_last[i] = -1
                    
                    div_term[tuple(boundary_slices_first)] = boundary_derivative[tuple(boundary_slices_first)]
                    div_term[tuple(boundary_slices_last)] = boundary_derivative[tuple(boundary_slices_last)]
                    
                    # Add to divergence
                    divergence += div_term
                else:
                    logger.warning(f"Grid dimension {i} has too few points for derivative calculation")
            except Exception as fallback_error:
                logger.error(f"Fallback derivative method also failed: {fallback_error}")
                # If all else fails, just skip this component's contribution
                pass
    
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