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

def compute_partial_derivative(field: np.ndarray,
                               grid: List[np.ndarray],
                               direction: int,
                               order: int = 2) -> np.ndarray:
    """
    Compute ∂_direction field on a (possibly non‑uniform) grid.
    
    This version vectorizes across all other axes and
    delegates to np.gradient for 2nd‑order central differences.
    """
    expected_ndim = len(grid)
    # ── 1) Expand lower‑dimensional fields to full rank ──────────────────────
    if field.ndim < expected_ndim:
        # append singleton dims at the end
        new_shape = field.shape + (1,) * (expected_ndim - field.ndim)
        logger.debug(f"Reshaping field {field.shape} → {new_shape}")
        field = field.reshape(new_shape)

    # ── 2) 2nd‑order central differences via numpy ───────────────────────────
    if order == 2:
        try:
            # Only need the coordinate array for this axis
            return np.gradient(field,
                               grid[direction],
                               axis=direction,
                               edge_order=2)
        except Exception as e:
            logger.error(f"np.gradient failed on axis {direction}: {e}")
            # fall through to 1st‑order below

    # ── 3) 1st‑order forward/backward stencil fallback ──────────────────────
    if order == 1 or order == 2:
        dx = np.diff(grid[direction])
        # uniform‑spacing assumption for simplicity:
        if not np.allclose(dx, dx[0]):
            logger.warning("Non‑uniform spacing detected; using average dx")
        dx0 = dx.mean()

        # prepare result container
        deriv = np.zeros_like(field)

        # build slices
        slc0 = [slice(None)] * expected_ndim
        slc1 = [slice(None)] * expected_ndim

        # forward diff at start
        slc0[direction] = 0
        slc1[direction] = 1
        deriv[tuple(slc0)] = (field[tuple(slc1)] - field[tuple(slc0)]) / dx0

        # backward diff at end
        slc0[direction] = -1
        slc1[direction] = -2
        deriv[tuple(slc0)] = (field[tuple(slc0)] - field[tuple(slc1)]) / dx0

        # central diff in the bulk
        slc0[direction] = slice(2, None)
        slc1[direction] = slice(None, -2)
        mid = [slice(None)] * expected_ndim
        mid[direction] = slice(1, -1)
        deriv[tuple(mid)] = (field[tuple(slc0)] - field[tuple(slc1)]) / (2 * dx0)

        return deriv

    # ── 4) unsupported order ─────────────────────────────────────────────────
    raise ValueError(f"Unsupported finite‑difference order: {order}")

import numpy as np
import logging
from typing import List, Callable

logger = logging.getLogger(__name__)

def compute_metric_derivatives(metric: np.ndarray,
                               grid: List[np.ndarray]) -> np.ndarray:
    """
    Vectorized finite-difference: returns ∂_d g_{ij} for all i,j and grid directions d.
    metric: shape (..., n, n)
    returns: shape (..., n, n, n)
    """
    # gradient returns list of arrays, one per grid axis
    derivs = np.gradient(metric, *grid,
                         axis=tuple(range(metric.ndim - 2)),
                         edge_order=2)
    # stack along new last axis
    return np.stack(derivs, axis=-1)


def compute_christoffel_on_grid(metric_funcs: List[List[Callable]],
                                grid: List[np.ndarray]) -> np.ndarray:
    """
    Compute Christoffel symbols Γ^k_{ij} on a curvilinear grid.

    metric_funcs: n×n list of callables f(point)→g_{ij}
    grid: list of arrays for each coordinate axis

    Returns: array shape grid_shape+(n,n,n)
    """
    n = len(grid)
    grid_shape = tuple(len(axis) for axis in grid)

    # 1) Build metric array at all points: shape grid_shape+(n,n)
    metric = np.zeros(grid_shape + (n, n))
    for idx in np.ndindex(grid_shape):
        point = [grid[d][idx[d]] for d in range(n)]
        g = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                comp = metric_funcs[i][j] if i < len(metric_funcs) and j < len(metric_funcs[i]) else None
                if callable(comp):
                    g[i, j] = comp(point)
                else:
                    g[i, j] = float(comp) if comp is not None else (1.0 if i == j else 0.0)
        metric[idx] = 0.5 * (g + g.T)  # enforce symmetry

    # 2) Invert metric at each point
    metric_inv = np.zeros_like(metric)
    for idx in np.ndindex(grid_shape):
        g = metric[idx]
        try:
            metric_inv[idx] = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            metric_inv[idx] = np.linalg.pinv(g)

    # 3) Compute metric derivatives: shape grid_shape+(n,n,n)
    metric_deriv = compute_metric_derivatives(metric, grid)

    # 4) Build S_{l,i,j} = ∂_i g_{l j} + ∂_j g_{l i} - ∂_l g_{i j}
    # metric_deriv[..., i, j, d] = ∂_d g_{i j}
    # We need axes: (..., l, i, j) for S
    # transpose metric_deriv to (..., l, j, i, d)
    md = metric_deriv
    # create S with shape grid_shape+(n,n,n)
    # use broadcasting of tensor contractions via einsum
    # S[..., l, i, j] = md[..., l, j, i] + md[..., l, i, j] - md[..., i, j, l]
    S = (
        md.transpose(*range(md.ndim-3), 2,3,1) +  # (..., l, i, j)
        md.transpose(*range(md.ndim-3), 2,1,3) -  # (..., l, j, i)
        md.transpose(*range(md.ndim-3), 1,2,3)    # (..., i, j, l)
    )

    # 5) Contract: Γ[...,k,i,j] = 0.5 * g^{k l} S[...,l,i,j]
    Gamma = 0.5 * np.einsum('...kl,...lij->...kij', metric_inv, S)

    # 6) Cleanup any non-finite values
    if not np.all(np.isfinite(Gamma)):
        logger.warning("Non-finite Christoffel values found; replacing with zero.")
        Gamma = np.nan_to_num(Gamma)

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

import numpy as np
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

def evaluate_divergence(vector_field: List[np.ndarray],
                        metric: np.ndarray,
                        grid: List[np.ndarray],
                        is_contravariant: bool = True) -> np.ndarray:
    """
    Compute divergence of a vector field on a curvilinear grid:
        div V = (1/√|g|) ∂_i(√|g| V^i)

    This version is fully vectorized using NumPy.
    """
    # Number of dimensions
    n = len(grid)
    # Stack components: shape (..., n)
    V = np.stack(vector_field, axis=-1)

    # If covariant, raise indices to contravariant
    if not is_contravariant:
        # metric shape: (..., n, n) or (n, n)
        if metric.ndim == 2:
            inv = np.linalg.pinv(metric)
            V = np.einsum('ij,...j->...i', inv, V)
        else:
            # position-dependent: shape (..., n, n)
            inv = np.linalg.inv(metric)
            V = np.einsum('...ij,...j->...i', inv, V)

    # Compute metric determinant and sqrt |g|
    if metric.ndim == 2:
        det = np.linalg.det(metric)
        sqrtg = np.sqrt(max(det, 1e-10))
    else:
        det = np.linalg.det(metric)
        det = np.maximum(det, 1e-10)
        sqrtg = np.sqrt(det)

    # Weighted vector: √|g| * V^i
    W = sqrtg[..., None] * V  # shape (..., n)

    # Compute all partial derivatives ∂_i W^i
    # gradient returns a list of length n, each shape (..., n)
    partials = np.gradient(W, *grid, axis=tuple(range(W.ndim - 1)), edge_order=2)

    # Sum diagonal terms: ∑_i ∂_i W^i
    div_num = sum(partials[i][..., i] for i in range(n))

    # Final divergence
    divergence = div_num / sqrtg
    return divergence


def evaluate_curl(vector_field: List[np.ndarray],
                  metric: np.ndarray,
                  grid: List[np.ndarray],
                  christoffel: Optional[np.ndarray] = None,
                  is_contravariant: bool = True) -> List[np.ndarray]:
    """
    Compute the curl of a 3D vector field on a curvilinear grid:
        (curl V)^i = (1/√|g|) ε^{ijk} (∇_j A_k)
    where A_k = g_{km} V^m and ∇_j A_k = ∂_j A_k - Γ^m_{jk} A_m.
    """
    # ==== 1) Stack components and handle index positions ====  
    n = 3
    V = np.stack(vector_field, axis=-1)  # shape (...,3)

    # Lower indices if needed
    if is_contravariant:
        # metric: [...,i,j] or (i,j)
        if metric.ndim == 2:
            A = np.einsum('ij,...j->...i', metric, V)
        else:
            A = np.einsum('...ij,...j->...i', metric, V)
    else:
        A = V.copy()

    # ==== 2) Compute covariant derivatives ∂_j A_k via one gradient call ====  
    # A has shape grid_shape+(3)
    axes = tuple(range(len(grid)))  # spatial axes
    derivs = np.gradient(A, *grid, axis=axes, edge_order=2)
    # derivs[j] has shape grid_shape+(3) giving ∂_j A_k
    D = np.stack(derivs, axis=-2)  # shape grid_shape+(3,3) with D[...,j,k]

    # ==== 3) Subtract Christoffel terms: ∇_j A_k = D_jk - Γ^m_{jk} A_m ====  
    if christoffel is None:
        # Assume christoffel zeros if not provided
        cov_derivs = D
    else:
        # Γ shape: grid_shape+(m,j,k)
        cov_de = np.einsum('...mjk,...m->...jk', christoffel, A)
        cov_derivs = D - cov_de  # shape grid_shape+(3,3)

    # ==== 4) Compute √|g| ====  
    if metric.ndim == 2:
        det = np.linalg.det(metric)
        sqrtg = np.sqrt(max(det, 1e-10))
    else:
        det = np.linalg.det(metric)
        det = np.maximum(det, 1e-10)
        sqrtg = np.sqrt(det)

    # ==== 5) Contract with Levi-Civita: (curl V)^i = 1/√g ε^{ijk} (∇_j A_k) ====  
    # Define ε^{ijk}
    eps = np.zeros((3,3,3), dtype=int)
    eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1
    eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1

    # Contract: result[...,i] = ε^{i j k} cov_derivs[...,j,k]
    curl = np.einsum('ijk,...jk->...i', eps, cov_derivs) / sqrtg[..., None]

    # ==== 6) Return as list of components ====  
    return [curl[..., i] for i in range(3)]

def evaluate_laplacian(scalar_field: np.ndarray,
                       metric: np.ndarray,
                       metric_inverse: Union[np.ndarray, List[np.ndarray]],
                       grid: List[np.ndarray]) -> np.ndarray:
    """
    Compute the Laplacian of a scalar field on a curvilinear grid:
        ∇² f = (1/√|g|) ∂_i (√|g| g^{ij} ∂_j f)

    Fully vectorized using NumPy.

    Args:
        scalar_field: array shape grid_shape
        metric: constant (n,n) or array shape grid_shape+(n,n)
        metric_inverse: constant (n,n) or array shape grid_shape+(n,n)
        grid: list of coordinate arrays

    Returns:
        Array of Laplacian values shape grid_shape
    """
    n = len(grid)
    # 1) Compute partial derivatives ∂_j f
    partials = np.gradient(scalar_field, *grid,
                           axis=tuple(range(scalar_field.ndim)),
                           edge_order=2)
    # Stack into (..., n)
    P = np.stack(partials, axis=-1)

    # 2) g^{ij} ∂_j f for each i
    if metric_inverse.ndim == 2:
        G = np.einsum('ij,...j->...i', metric_inverse, P)
    else:
        G = np.einsum('...ij,...j->...i', metric_inverse, P)

    # 3) Compute √|g|
    if metric.ndim == 2:
        det = np.linalg.det(metric)
        sqrtg = np.sqrt(max(det, 1e-10))
    else:
        det = np.linalg.det(metric)
        det = np.maximum(det, 1e-10)
        sqrtg = np.sqrt(det)

    # 4) Weighted vector: √|g| * G^i
    W = sqrtg[..., None] * G  # shape (..., n)

    # 5) Divergence ∂_i W^i via single gradient call
    partials_W = np.gradient(W, *grid,
                             axis=tuple(range(W.ndim - 1)),
                             edge_order=2)
    # Sum diagonal terms
    lap_num = sum(partials_W[i][..., i] for i in range(n))

    # 6) Final Laplacian
    laplacian = lap_num / sqrtg
    return laplacian

def evaluate_dalembert(scalar_field: np.ndarray,
                       metric: np.ndarray,
                       metric_inverse: np.ndarray,
                       grid: List[np.ndarray]) -> np.ndarray:
    """
    Alias for the 4D Laplacian (wave operator).
    """
    if len(grid) != 4:
        raise ValueError("d'Alembertian is defined only in 4D spacetime")
    return evaluate_laplacian(scalar_field, metric, metric_inverse, grid)


def apply_boundary_condition(field: np.ndarray,
                             grid: List[np.ndarray],
                             boundary_type: str,
                             boundary_values: Dict[str, Any] = None) -> np.ndarray:
    """
    Apply Dirichlet, Neumann, or periodic BCs with minimal overhead.
    """
    out = field.copy()
    ndim = out.ndim
    btype = boundary_type.lower()

    if btype == 'dirichlet' and boundary_values:
        for d in range(ndim):
            low = [slice(None)]*ndim; low[d] = 0
            high= [slice(None)]*ndim; high[d]= -1
            lk, uk = f'dim{d}_lower', f'dim{d}_upper'
            if lk in boundary_values:
                out[tuple(low)]  = boundary_values[lk]
            if uk in boundary_values:
                out[tuple(high)] = boundary_values[uk]

    elif btype == 'neumann' and boundary_values:
        for d in range(ndim):
            dx = grid[d][1] - grid[d][0]
            idx0 = [slice(None)]*ndim; idx0[d] = 0
            idx1 = [slice(None)]*ndim; idx1[d] = 1
            idxm2= [slice(None)]*ndim; idxm2[d]= -2
            idxm1= [slice(None)]*ndim; idxm1[d]= -1
            lk, uk = f'dim{d}_lower', f'dim{d}_upper'
            if lk in boundary_values:
                out[tuple(idx0)] = out[tuple(idx1)] - dx * boundary_values[lk]
            if uk in boundary_values:
                out[tuple(idxm1)]= out[tuple(idxm2)] + dx * boundary_values[uk]

    elif btype == 'periodic':
        # wrap via pad+slice
        pads = [(1,1)]*ndim
        out = np.pad(out, pads, mode='wrap')
        slices = tuple(slice(1,-1) for _ in range(ndim))
        out = out[slices]

    else:
        raise ValueError(f"Unsupported or incomplete BC: {boundary_type}")

    return out


def interpolate_field(field: np.ndarray,
                      grid: List[np.ndarray],
                      point: List[float]) -> float:
    """
    Pure–NumPy multilinear interpolation at one point.
    """
    n = len(grid)
    # 1) locate bracketing indices and weights
    idx0 = [0]*n
    w    = [0.0]*n
    for d in range(n):
        xi = grid[d]
        p  = point[d]
        i  = np.searchsorted(xi, p) - 1
        i  = min(max(i, 0), len(xi)-2)
        idx0[d] = i
        x0, x1 = xi[i], xi[i+1]
        w[d]    = (p - x0) / (x1 - x0)

    # 2) sum over 2^n corners
    result = 0.0
    for corner in range(1 << n):
        weight = 1.0
        loc    = [0]*n
        for d in range(n):
            bit = (corner >> d) & 1
            weight *= w[d] if bit else (1 - w[d])
            loc[d]   = idx0[d] + bit
        result += weight * field[tuple(loc)]

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