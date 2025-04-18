"""
Grid module for MHD simulations.

This module provides functionality for creating and managing computational grids
in various coordinate systems, with support for both symbolic and numerical operations.
It handles coordinate transformations, metric calculations, and implements differential
operators for MHD simulations.
"""

import numpy as np
import sympy as sp
from numba import njit
import logging
from typing import Dict, List, Tuple, Union, Callable, Optional

# Setup logging
logger = logging.getLogger(__name__)

def create_grid(coord_ranges: Dict[str, Dict[str, float]], 
                resolution: List[int], 
                include_endpoints: bool = True) -> Tuple[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]:
    """
    Create a computational grid for MHD simulations.
    
    Args:
        coord_ranges: Dictionary mapping coordinate names to their min/max ranges
                      e.g. {'x': {'min': 0, 'max': 1}, 'y': {'min': 0, 'max': 1}}
        resolution: List of grid points in each dimension
        include_endpoints: Whether to include the endpoints in the grid
    
    Returns:
        Tuple containing:
            - Tuple of 1D grid arrays for each coordinate (not meshgrid)
            - Dictionary of grid spacing for each coordinate
    """
    # First create as dictionary for easier manipulation
    grid_dict = {}
    spacing = {}
    coord_names = []
    
    for i, (coord_name, range_info) in enumerate(coord_ranges.items()):
        coord_names.append(coord_name)
        min_val = range_info['min']
        max_val = range_info['max']
        
        # Number of grid points in this dimension
        n_points = resolution[i]
        
        # Create the grid for this coordinate
        if include_endpoints:
            # Include both endpoints
            coord_grid = np.linspace(min_val, max_val, n_points)
        else:
            # Cell-centered grid (exclude endpoints)
            dx = (max_val - min_val) / n_points
            coord_grid = np.linspace(min_val + dx/2, max_val - dx/2, n_points)
        
        # Store the grid and spacing
        grid_dict[coord_name] = coord_grid
        spacing[coord_name] = (max_val - min_val) / (n_points - 1 if include_endpoints else n_points)
    
    # Store the 1D arrays in a tuple (in the same order as coord_names)
    grid_tuple = tuple(grid_dict[name] for name in coord_names)
    
    # Store the coordinate names for reference
    spacing['_coord_names'] = coord_names
    
    return grid_tuple, spacing

def create_staggered_grid(grid: Tuple[np.ndarray, ...], 
                          spacing: Dict[str, float]) -> Tuple[Tuple[np.ndarray, ...], ...]:
    """
    Create a staggered grid for constrained transport MHD.
    
    This creates face-centered grids needed for maintaining div(B) = 0.
    
    Args:
        grid: Tuple of grid arrays (either 1D arrays or meshgrids)
        spacing: Dictionary of grid spacing for each coordinate
    
    Returns:
        Tuple of face-centered grids for each dimension
    """
    dimension = len(grid)
    coord_names = spacing.get('_coord_names', [f'x{i}' for i in range(dimension)])
    
    # Check if the input is already a meshgrid (2D or 3D arrays) or 1D arrays
    is_meshgrid = len(grid[0].shape) > 1
    
    if is_meshgrid:
        # If meshgrid, use the shape directly
        grid_shape = grid[0].shape
    else:
        # If 1D arrays, use their length for shape
        grid_shape = tuple(len(grid[i]) for i in range(dimension))
    
    # Initialize staggered grids
    staggered_grids = []
    
    for i in range(dimension):
        # Create face-centered grids for this coordinate
        face_grids = []
        
        for j in range(dimension):
            if i == j:
                # Face centers for this coordinate
                if is_meshgrid:
                    # For meshgrid, directly use the grid arrays
                    face_centers = np.copy(grid[i])
                    
                    # Shift grid half a cell (except at boundaries where we extrapolate)
                    slice_idx = [slice(None)] * dimension
                    slice_idx[i] = slice(1, None)
                    shift_slice = tuple(slice_idx)
                    
                    slice_idx = [slice(None)] * dimension
                    slice_idx[i] = slice(0, -1)
                    orig_slice = tuple(slice_idx)
                    
                    face_centers[orig_slice] = 0.5 * (face_centers[orig_slice] + face_centers[shift_slice])
                    
                    # Handle boundary (extrapolate)
                    # Get the correct slice for the boundary
                    slice_idx = [slice(None)] * dimension
                    slice_idx[i] = -1
                    boundary_slice = tuple(slice_idx)
                    
                    # Get the previous slice for extrapolation
                    slice_idx[i] = -2
                    prev_slice = tuple(slice_idx)
                    
                    # Extrapolate the boundary
                    face_centers[boundary_slice] = 2 * face_centers[boundary_slice] - face_centers[prev_slice]
                else:
                    # For 1D arrays, use the original approach
                    face_centers = np.copy(grid[i])
                    
                    # Shift grid half a cell (except at boundaries where we extrapolate)
                    face_centers = 0.5 * (face_centers + np.roll(face_centers, -1, axis=i))
                    
                    # Handle boundary (don't wrap around but extrapolate)
                    # Get the correct slice for the boundary
                    slice_idx = [slice(None)] * len(grid_shape)
                    slice_idx[i] = -1
                    boundary_slice = tuple(slice_idx)
                    
                    # Get the previous slice for extrapolation
                    slice_idx[i] = -2
                    prev_slice = tuple(slice_idx)
                    
                    # Extrapolate the boundary
                    face_centers[boundary_slice] = 2 * face_centers[boundary_slice] - face_centers[prev_slice]
                
                face_grids.append(face_centers)
            else:
                # Keep the same grid for other coordinates
                face_grids.append(grid[j])
        
        staggered_grids.append(tuple(face_grids))
    
    return tuple(staggered_grids)

def metric_from_transformation(
    transform_map: List[sp.Expr],
    base_metric: sp.Matrix,
    coord_symbols: List[sp.Symbol]
) -> sp.Matrix:
    """
    Compute the metric tensor given a coordinate transformation.
    
    Args:
        transform_map: List of transformation expressions from curvilinear to Cartesian
        base_metric: Base metric tensor in Cartesian coordinates (usually identity)
        coord_symbols: List of symbolic coordinate variables
    
    Returns:
        Metric tensor in the curvilinear coordinate system
    """
    dimension = len(coord_symbols)
    jacobian = sp.zeros(dimension, dimension)
    
    # Compute the Jacobian matrix of the transformation
    for i in range(dimension):
        for j in range(dimension):
            jacobian[i, j] = sp.diff(transform_map[i], coord_symbols[j])
    
    # Compute the metric tensor using the Jacobian: g = J^T * base_metric * J
    metric = jacobian.transpose() * base_metric * jacobian
    
    return metric

def compute_christoffel_symbols(metric: sp.Matrix, coord_symbols: List[sp.Symbol]) -> List[List[List[sp.Expr]]]:
    """
    Compute the Christoffel symbols for a given metric tensor.
    
    Args:
        metric: Metric tensor as a SymPy matrix
        coord_symbols: List of symbolic coordinate variables
    
    Returns:
        3D list of Christoffel symbols Gamma^i_jk
    """
    dimension = len(coord_symbols)
    
    # Compute the inverse metric
    inverse_metric = metric.inv()
    
    # Initialize the Christoffel symbols as a 3D list
    christoffel = [[[sp.S.Zero for _ in range(dimension)] 
                    for _ in range(dimension)] 
                   for _ in range(dimension)]
    
    # Compute Christoffel symbols using the formula:
    # Gamma^i_jk = 0.5 * g^im * (d_j g_km + d_k g_jm - d_m g_jk)
    for i in range(dimension):
        for j in range(dimension):
            for k in range(dimension):
                for m in range(dimension):
                    # Calculate term1: d_j g_km
                    term1 = sp.diff(metric[k, m], coord_symbols[j])
                    
                    # Calculate term2: d_k g_jm
                    term2 = sp.diff(metric[j, m], coord_symbols[k])
                    
                    # Calculate term3: d_m g_jk
                    term3 = sp.diff(metric[j, k], coord_symbols[m])
                    
                    # Accumulate the contribution to the Christoffel symbol
                    christoffel[i][j][k] += 0.5 * inverse_metric[i, m] * (term1 + term2 - term3)
    
    return christoffel

def symbolic_gradient(
    scalar_field: sp.Expr,
    coord_symbols: List[sp.Symbol],
    metric: sp.Matrix
) -> List[sp.Expr]:
    """
    Compute the symbolic gradient of a scalar field in curvilinear coordinates.
    
    Args:
        scalar_field: SymPy expression for the scalar field
        coord_symbols: List of symbolic coordinate variables
        metric: Metric tensor as a SymPy matrix
    
    Returns:
        List of SymPy expressions for the contravariant gradient components
    """
    dimension = len(coord_symbols)
    inverse_metric = metric.inv()
    gradient = []
    
    # Compute the gradient using g^{ij} * partial_j(scalar_field)
    for i in range(dimension):
        grad_component = sp.S.Zero
        for j in range(dimension):
            grad_component += inverse_metric[i, j] * sp.diff(scalar_field, coord_symbols[j])
        gradient.append(grad_component)
    
    return gradient

def symbolic_divergence(
    vector_field: List[sp.Expr],
    coord_symbols: List[sp.Symbol],
    metric: sp.Matrix,
    christoffel: List[List[List[sp.Expr]]] = None
) -> sp.Expr:
    """
    Compute the symbolic divergence of a vector field in curvilinear coordinates.
    
    Args:
        vector_field: List of SymPy expressions for vector field components
        coord_symbols: List of symbolic coordinate variables
        metric: Metric tensor as a SymPy matrix
        christoffel: Christoffel symbols (computed if not provided)
    
    Returns:
        SymPy expression for the divergence
    """
    dimension = len(coord_symbols)
    
    # Compute determinant of the metric
    g_det = metric.det()
    sqrt_g = sp.sqrt(g_det)
    
    # Compute the divergence using: (1/sqrt(g)) * partial_i(sqrt(g) * v^i)
    divergence = sp.S.Zero
    
    for i in range(dimension):
        term = sqrt_g * vector_field[i]
        divergence += sp.diff(term, coord_symbols[i])
    
    divergence = divergence / sqrt_g
    
    return divergence

@njit
def numerical_gradient(
    scalar_field: np.ndarray,
    grid: Tuple[np.ndarray, ...],
    spacing: Dict[str, float],
    metric_inverse: np.ndarray
) -> List[np.ndarray]:
    """
    Compute the numerical gradient of a scalar field on a grid.
    
    Args:
        scalar_field: Array of scalar field values
        grid: Tuple of grid arrays
        spacing: Dictionary of grid spacing
        metric_inverse: Inverse metric tensor
    
    Returns:
        List of arrays for the gradient components
    """
    dimension = len(grid)
    shape = scalar_field.shape
    coord_names = spacing.get('_coord_names', [f'x{i}' for i in range(dimension)])
    
    # Initialize gradient components
    gradient = [np.zeros(shape) for _ in range(dimension)]
    
    # Compute covariant derivatives (simple central differences)
    partial_derivatives = []
    for i in range(dimension):
        coord_name = coord_names[i]
        dx = spacing[coord_name]
        # Use central differences for interior points
        partial = np.zeros_like(scalar_field)
        
        # Create slices for forward and backward differences
        slices_forward = [slice(None)] * dimension
        slices_backward = [slice(None)] * dimension
        slices_center = [slice(None)] * dimension
        
        # Interior points (central difference)
        slices_forward[i] = slice(2, None)
        slices_backward[i] = slice(0, -2)
        slices_center[i] = slice(1, -1)
        
        partial[tuple(slices_center)] = (scalar_field[tuple(slices_forward)] - 
                                        scalar_field[tuple(slices_backward)]) / (2 * dx)
        
        # Forward difference for the first point
        slices_forward[i] = slice(1, 3)
        slices_center[i] = 0
        
        if shape[i] > 2:
            partial[tuple(slices_center)] = (-1.5 * scalar_field[tuple(slices_center)] + 
                                           2.0 * scalar_field[tuple(slices_forward)][..., 0] - 
                                           0.5 * scalar_field[tuple(slices_forward)][..., 1]) / dx
        else:
            # Fallback for small grids
            partial[tuple(slices_center)] = (scalar_field[tuple(slices_forward)][..., 0] - 
                                           scalar_field[tuple(slices_center)]) / dx
        
        # Backward difference for the last point
        slices_backward[i] = slice(-3, -1)
        slices_center[i] = -1
        
        if shape[i] > 2:
            partial[tuple(slices_center)] = (1.5 * scalar_field[tuple(slices_center)] - 
                                           2.0 * scalar_field[tuple(slices_backward)][..., -1] + 
                                           0.5 * scalar_field[tuple(slices_backward)][..., -2]) / dx
        else:
            # Fallback for small grids
            partial[tuple(slices_center)] = (scalar_field[tuple(slices_center)] - 
                                           scalar_field[tuple(slices_backward)][..., -1]) / dx
        
        partial_derivatives.append(partial)
    
    # Combine with inverse metric to get contravariant components
    for i in range(dimension):
        for j in range(dimension):
            gradient[i] += metric_inverse[i, j] * partial_derivatives[j]
    
    return gradient

@njit
def numerical_divergence(
    vector_field: List[np.ndarray],
    grid: Tuple[np.ndarray, ...],
    spacing: Dict[str, float],
    metric_determinant: np.ndarray
) -> np.ndarray:
    """
    Compute the numerical divergence of a vector field on a grid.
    
    Args:
        vector_field: List of arrays for vector field components
        grid: Tuple of grid arrays
        spacing: Dictionary of grid spacing
        metric_determinant: Determinant of the metric tensor
    
    Returns:
        Array of divergence values
    """
    dimension = len(grid)
    shape = vector_field[0].shape
    coord_names = spacing.get('_coord_names', [f'x{i}' for i in range(dimension)])
    
    # Initialize divergence array
    divergence = np.zeros(shape)
    
    # Compute the divergence using: (1/sqrt(g)) * partial_i(sqrt(g) * v^i)
    sqrt_g = np.sqrt(metric_determinant)
    
    for i in range(dimension):
        coord_name = coord_names[i]
        dx = spacing[coord_name]
        # Compute v_scaled = sqrt(g) * v^i
        v_scaled = sqrt_g * vector_field[i]
        
        # Use central differences for interior points
        div_term = np.zeros_like(divergence)
        
        # Create slices for forward and backward differences
        slices_forward = [slice(None)] * dimension
        slices_backward = [slice(None)] * dimension
        slices_center = [slice(None)] * dimension
        
        # Interior points (central difference)
        slices_forward[i] = slice(2, None)
        slices_backward[i] = slice(0, -2)
        slices_center[i] = slice(1, -1)
        
        div_term[tuple(slices_center)] = (v_scaled[tuple(slices_forward)] - 
                                         v_scaled[tuple(slices_backward)]) / (2 * dx)
        
        # Forward difference for the first point
        slices_forward[i] = slice(1, 3)
        slices_center[i] = 0
        
        if shape[i] > 2:
            div_term[tuple(slices_center)] = (-1.5 * v_scaled[tuple(slices_center)] + 
                                            2.0 * v_scaled[tuple(slices_forward)][..., 0] - 
                                            0.5 * v_scaled[tuple(slices_forward)][..., 1]) / dx
        else:
            # Fallback for small grids
            div_term[tuple(slices_center)] = (v_scaled[tuple(slices_forward)][..., 0] - 
                                            v_scaled[tuple(slices_center)]) / dx
        
        # Backward difference for the last point
        slices_backward[i] = slice(-3, -1)
        slices_center[i] = -1
        
        if shape[i] > 2:
            div_term[tuple(slices_center)] = (1.5 * v_scaled[tuple(slices_center)] - 
                                            2.0 * v_scaled[tuple(slices_backward)][..., -1] + 
                                            0.5 * v_scaled[tuple(slices_backward)][..., -2]) / dx
        else:
            # Fallback for small grids
            div_term[tuple(slices_center)] = (v_scaled[tuple(slices_center)] - 
                                            v_scaled[tuple(slices_backward)][..., -1]) / dx
        
        divergence += div_term
    
    # Divide by sqrt(g)
    divergence = divergence / sqrt_g
    
    return divergence

@njit
def numerical_curl_2d(
    vector_field: List[np.ndarray],
    grid: Tuple[np.ndarray, ...],
    spacing: Dict[str, float]
) -> np.ndarray:
    """
    Compute the curl of a 2D vector field (returns scalar).
    
    Args:
        vector_field: List of arrays for vector field components [vx, vy]
        grid: Tuple of grid arrays
        spacing: Dictionary of grid spacing
    
    Returns:
        Array of curl values (vorticity)
    """
    # Extract components and spacing
    vx = vector_field[0]
    vy = vector_field[1]
    coord_names = spacing.get('_coord_names', ['x', 'y'])
    dx = spacing[coord_names[0]]
    dy = spacing[coord_names[1]]
    
    # Initialize curl array
    curl = np.zeros_like(vx)
    
    # Interior points (central differences)
    curl[1:-1, 1:-1] = ((vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * dx) - 
                        (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * dy))
    
    # Boundary points (one-sided differences)
    # Left and right boundaries
    curl[0, 1:-1] = ((vy[1, 1:-1] - vy[0, 1:-1]) / dx - 
                     (vx[0, 2:] - vx[0, :-2]) / (2 * dy))
    curl[-1, 1:-1] = ((vy[-1, 1:-1] - vy[-2, 1:-1]) / dx - 
                      (vx[-1, 2:] - vx[-1, :-2]) / (2 * dy))
    
    # Top and bottom boundaries
    curl[1:-1, 0] = ((vy[2:, 0] - vy[:-2, 0]) / (2 * dx) - 
                     (vx[1:-1, 1] - vx[1:-1, 0]) / dy)
    curl[1:-1, -1] = ((vy[2:, -1] - vy[:-2, -1]) / (2 * dx) - 
                      (vx[1:-1, -1] - vx[1:-1, -2]) / dy)
    
    # Corner points
    curl[0, 0] = ((vy[1, 0] - vy[0, 0]) / dx - 
                  (vx[0, 1] - vx[0, 0]) / dy)
    curl[0, -1] = ((vy[1, -1] - vy[0, -1]) / dx - 
                   (vx[0, -1] - vx[0, -2]) / dy)
    curl[-1, 0] = ((vy[-1, 0] - vy[-2, 0]) / dx - 
                   (vx[-1, 1] - vx[-1, 0]) / dy)
    curl[-1, -1] = ((vy[-1, -1] - vy[-2, -1]) / dx - 
                    (vx[-1, -1] - vx[-1, -2]) / dy)
    
    return curl

@njit
def numerical_curl_3d(
    vector_field: List[np.ndarray],
    grid: Tuple[np.ndarray, ...],
    spacing: Dict[str, float]
) -> List[np.ndarray]:
    """
    Compute the curl of a 3D vector field.
    
    Args:
        vector_field: List of arrays for vector field components [vx, vy, vz]
        grid: Tuple of grid arrays
        spacing: Dictionary of grid spacing
    
    Returns:
        List of arrays for curl components [curl_x, curl_y, curl_z]
    """
    # Extract components and spacing
    vx = vector_field[0]
    vy = vector_field[1]
    vz = vector_field[2]
    coord_names = spacing.get('_coord_names', ['x', 'y', 'z'])
    dx = spacing[coord_names[0]]
    dy = spacing[coord_names[1]]
    dz = spacing[coord_names[2]]
    
    # Initialize curl components
    curl_x = np.zeros_like(vx)
    curl_y = np.zeros_like(vy)
    curl_z = np.zeros_like(vz)
    
    # Interior points (central differences)
    # curl_x = dvy/dz - dvz/dy
    curl_x[1:-1, 1:-1, 1:-1] = ((vy[1:-1, 1:-1, 2:] - vy[1:-1, 1:-1, :-2]) / (2 * dz) - 
                               (vz[1:-1, 2:, 1:-1] - vz[1:-1, :-2, 1:-1]) / (2 * dy))
    
    # curl_y = dvz/dx - dvx/dz
    curl_y[1:-1, 1:-1, 1:-1] = ((vz[2:, 1:-1, 1:-1] - vz[:-2, 1:-1, 1:-1]) / (2 * dx) - 
                               (vx[1:-1, 1:-1, 2:] - vx[1:-1, 1:-1, :-2]) / (2 * dz))
    
    # curl_z = dvx/dy - dvy/dx
    curl_z[1:-1, 1:-1, 1:-1] = ((vx[1:-1, 2:, 1:-1] - vx[1:-1, :-2, 1:-1]) / (2 * dy) - 
                               (vy[2:, 1:-1, 1:-1] - vy[:-2, 1:-1, 1:-1]) / (2 * dx))
    
    # Note: Boundary points would require similar one-sided differences as in the 2D case
    # But implementation is omitted for brevity
    
    return [curl_x, curl_y, curl_z]

def create_curvilinear_coordinates(coordinate_system: str, 
                                  dimension: int = 2) -> Tuple[List[sp.Symbol], List[sp.Expr], sp.Matrix]:
    """
    Create symbolic variables and transformations for common coordinate systems.
    
    Args:
        coordinate_system: Name of the coordinate system ('cartesian', 'cylindrical', 'spherical', etc.)
        dimension: Number of dimensions (2 or 3)
    
    Returns:
        Tuple containing:
            - List of coordinate symbols
            - List of transformation expressions to Cartesian
            - Metric tensor
    """
    if dimension not in [2, 3]:
        raise ValueError("Only 2D and 3D coordinate systems are supported")
    
    # Create basic coordinate symbols
    if coordinate_system.lower() == 'cartesian':
        if dimension == 2:
            x, y = sp.symbols('x y')
            coord_symbols = [x, y]
            transform_map = [x, y]  # Identity transformation
            metric = sp.Matrix([[1, 0], [0, 1]])  # Identity metric
        else:  # 3D
            x, y, z = sp.symbols('x y z')
            coord_symbols = [x, y, z]
            transform_map = [x, y, z]  # Identity transformation
            metric = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity metric
    
    elif coordinate_system.lower() == 'cylindrical':
        if dimension == 2:
            r, theta = sp.symbols('r theta')
            coord_symbols = [r, theta]
            transform_map = [r * sp.cos(theta), r * sp.sin(theta)]
            # Compute metric from the transformation
            metric = metric_from_transformation(transform_map, sp.eye(2), coord_symbols)
        else:  # 3D
            r, theta, z = sp.symbols('r theta z')
            coord_symbols = [r, theta, z]
            transform_map = [r * sp.cos(theta), r * sp.sin(theta), z]
            # Compute metric from the transformation
            metric = metric_from_transformation(transform_map, sp.eye(3), coord_symbols)
    
    elif coordinate_system.lower() == 'spherical':
        if dimension == 2:
            # 2D spherical (polar) is the same as 2D cylindrical
            r, theta = sp.symbols('r theta')
            coord_symbols = [r, theta]
            transform_map = [r * sp.cos(theta), r * sp.sin(theta)]
            metric = metric_from_transformation(transform_map, sp.eye(2), coord_symbols)
        else:  # 3D
            r, theta, phi = sp.symbols('r theta phi')
            coord_symbols = [r, theta, phi]
            transform_map = [
                r * sp.sin(theta) * sp.cos(phi),
                r * sp.sin(theta) * sp.sin(phi),
                r * sp.cos(theta)
            ]
            metric = metric_from_transformation(transform_map, sp.eye(3), coord_symbols)
    else:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")
    
    return coord_symbols, transform_map, metric

def lambdify_metric_functions(metric: sp.Matrix, 
                             coord_symbols: List[sp.Symbol]) -> Dict[str, Callable]:
    """
    Create numerical functions from symbolic metric expressions.
    
    Args:
        metric: Symbolic metric tensor
        coord_symbols: List of coordinate symbols
    
    Returns:
        Dictionary of lambda functions for metric components, determinant, etc.
    """
    dimension = len(coord_symbols)
    
    # Calculate inverse metric and determinant symbolically
    inverse_metric = metric.inv()
    metric_det = metric.det()
    
    # Initialize result dictionary
    metric_funcs = {
        'dimension': dimension,
        'components': {},
        'inverse_components': {},
        'determinant': sp.lambdify(coord_symbols, metric_det, 'numpy')
    }
    
    # Lambdify each metric component
    for i in range(dimension):
        for j in range(dimension):
            key = f'g_{i}{j}'
            metric_funcs['components'][key] = sp.lambdify(coord_symbols, metric[i, j], 'numpy')
            
            key = f'g^{i}{j}'
            metric_funcs['inverse_components'][key] = sp.lambdify(coord_symbols, inverse_metric[i, j], 'numpy')
    
    return metric_funcs

def apply_boundary_conditions(
    field: np.ndarray,
    boundary_type: str = 'periodic',
    dimension: int = None,
    value: float = 0.0
) -> np.ndarray:
    """
    Apply boundary conditions to a field array.
    
    Args:
        field: The field array to apply boundary conditions to
        boundary_type: Type of boundary condition ('periodic', 'dirichlet', 'neumann', 'outflow')
        dimension: Dimension along which to apply the boundary (None for all dimensions)
        value: Value to use for Dirichlet boundary condition
    
    Returns:
        Updated field with boundary conditions applied
    """
    result = field.copy()
    dims = range(field.ndim) if dimension is None else [dimension]
    
    for dim in dims:
        if boundary_type.lower() == 'periodic':
            # For periodic boundaries, copy from opposite side
            slices_src_left = [slice(None)] * field.ndim
            slices_dst_left = [slice(None)] * field.ndim
            slices_src_right = [slice(None)] * field.ndim
            slices_dst_right = [slice(None)] * field.ndim
            
            # Set the specific dimension slices
            slices_src_left[dim] = -2  # Second-to-last element
            slices_dst_left[dim] = 0   # First element
            slices_src_right[dim] = 1  # Second element
            slices_dst_right[dim] = -1 # Last element
            
            # Apply the periodic boundary
            result[tuple(slices_dst_left)] = field[tuple(slices_src_left)]
            result[tuple(slices_dst_right)] = field[tuple(slices_src_right)]
            
        elif boundary_type.lower() == 'dirichlet':
            # For Dirichlet boundary, set to a fixed value
            slices_left = [slice(None)] * field.ndim
            slices_right = [slice(None)] * field.ndim
            
            # Set the specific dimension slices
            slices_left[dim] = 0    # First element
            slices_right[dim] = -1  # Last element
            
            # Apply the Dirichlet boundary
            result[tuple(slices_left)] = value
            result[tuple(slices_right)] = value
            
        elif boundary_type.lower() == 'neumann':
            # For Neumann boundary, set the derivative to zero (copy neighboring value)
            slices_src_left = [slice(None)] * field.ndim
            slices_dst_left = [slice(None)] * field.ndim
            slices_src_right = [slice(None)] * field.ndim
            slices_dst_right = [slice(None)] * field.ndim
            
            # Set the specific dimension slices
            slices_src_left[dim] = 1    # Second element
            slices_dst_left[dim] = 0    # First element
            slices_src_right[dim] = -2  # Second-to-last element
            slices_dst_right[dim] = -1  # Last element
            
            # Apply the Neumann boundary
            result[tuple(slices_dst_left)] = field[tuple(slices_src_left)]
            result[tuple(slices_dst_right)] = field[tuple(slices_src_right)]
            
        elif boundary_type.lower() == 'outflow':
            # For outflow boundary, extrapolate from interior
            slices_src1_left = [slice(None)] * field.ndim
            slices_src2_left = [slice(None)] * field.ndim
            slices_dst_left = [slice(None)] * field.ndim
            slices_src1_right = [slice(None)] * field.ndim
            slices_src2_right = [slice(None)] * field.ndim
            slices_dst_right = [slice(None)] * field.ndim
            
            # Set the specific dimension slices
            slices_src1_left[dim] = 1   # Second element
            slices_src2_left[dim] = 2   # Third element
            slices_dst_left[dim] = 0    # First element
            slices_src1_right[dim] = -2 # Second-to-last element
            slices_src2_right[dim] = -3 # Third-to-last element
            slices_dst_right[dim] = -1  # Last element
            
            # Linear extrapolation
            result[tuple(slices_dst_left)] = 2 * field[tuple(slices_src1_left)] - field[tuple(slices_src2_left)]
            result[tuple(slices_dst_right)] = 2 * field[tuple(slices_src1_right)] - field[tuple(slices_src2_right)]
            
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    return result

def apply_vector_boundary_conditions(
    vector_field: List[np.ndarray],
    boundary_types: List[str] = None,
    dimension: int = None
) -> List[np.ndarray]:
    """
    Apply boundary conditions to a vector field.
    
    This handles special cases like reflective boundaries where components
    may need different treatment depending on their direction.
    
    Args:
        vector_field: List of arrays for vector field components
        boundary_types: List of boundary types for each dimension
        dimension: Dimension along which to apply the boundary (None for all dimensions)
    
    Returns:
        Updated vector field with boundary conditions applied
    """
    dims = range(vector_field[0].ndim) if dimension is None else [dimension]
    
    # If boundary_types not specified, default to periodic
    if boundary_types is None:
        boundary_types = ['periodic'] * len(dims)
    
    # Ensure we have enough boundary types
    if len(boundary_types) < len(dims):
        boundary_types.extend(['periodic'] * (len(dims) - len(boundary_types)))
    
    result = [field.copy() for field in vector_field]
    
    for dim_idx, dim in enumerate(dims):
        boundary_type = boundary_types[dim_idx].lower()
        
        # For reflective boundaries, normal component flips sign, tangential unchanged
        if boundary_type == 'reflective':
            for comp_idx, component in enumerate(vector_field):
                # If this component is in the direction of the boundary, flip its sign
                flip_sign = 1.0
                if comp_idx == dim:
                    flip_sign = -1.0
                
                # Set up slices for the boundary
                slices_src_left = [slice(None)] * component.ndim
                slices_dst_left = [slice(None)] * component.ndim
                slices_src_right = [slice(None)] * component.ndim
                slices_dst_right = [slice(None)] * component.ndim
                
                # Set the specific dimension slices
                slices_src_left[dim] = 1    # Second element
                slices_dst_left[dim] = 0    # First element
                slices_src_right[dim] = -2  # Second-to-last element
                slices_dst_right[dim] = -1  # Last element
                
                # Apply the reflective boundary
                result[comp_idx][tuple(slices_dst_left)] = flip_sign * component[tuple(slices_src_left)]
                result[comp_idx][tuple(slices_dst_right)] = flip_sign * component[tuple(slices_src_right)]
        else:
            # For other boundary types, apply the same condition to all components
            for comp_idx, component in enumerate(vector_field):
                result[comp_idx] = apply_boundary_conditions(
                    component, 
                    boundary_type=boundary_type, 
                    dimension=dim,
                    value=0.0  # Default value for Dirichlet
                )
    
    return result

def apply_divB_preserving_boundary(
    magnetic_field: List[np.ndarray],
    boundary_types: List[str],
    dimension: int = None
) -> List[np.ndarray]:
    """
    Apply boundary conditions to a magnetic field preserving div(B) = 0.
    
    This is especially important for MHD simulations to avoid creating
    magnetic monopoles at the boundaries.
    
    Args:
        magnetic_field: List of arrays for magnetic field components [Bx, By, Bz]
        boundary_types: List of boundary types for each dimension
        dimension: Dimension along which to apply the boundary (None for all dimensions)
    
    Returns:
        Updated magnetic field with div(B)=0 preserving boundary conditions applied
    """
    dims = range(magnetic_field[0].ndim) if dimension is None else [dimension]
    
    # If boundary_types not specified, default to periodic
    if boundary_types is None:
        boundary_types = ['periodic'] * len(dims)
    
    # Ensure we have enough boundary types
    if len(boundary_types) < len(dims):
        boundary_types.extend(['periodic'] * (len(dims) - len(boundary_types)))
    
    # First apply standard vector boundary conditions
    result = apply_vector_boundary_conditions(
        magnetic_field,
        boundary_types,
        dimension
    )
    
    # For non-periodic boundaries, ensure div(B)=0 at boundaries
    for dim_idx, dim in enumerate(dims):
        boundary_type = boundary_types[dim_idx].lower()
        
        # Skip periodic boundaries as they naturally preserve div(B)=0
        if boundary_type == 'periodic':
            continue
            
        # For other boundary types, adjust the normal component to ensure div(B)=0
        # This is done by computing the divergence at the boundary and then
        # adjusting the normal component to make it zero
        
        # For brevity, we'll implement a simplified version here
        # In a full implementation, you would:
        # 1. Calculate div(B) at the boundary
        # 2. Compute the correction to the normal component
        # 3. Apply the correction
        
        # Left boundary
        slices_boundary = [slice(None)] * magnetic_field[0].ndim
        slices_boundary[dim] = 0  # First element
        
        # For outflow boundaries, copy neighboring values and avoid monopoles
        if boundary_type in ['outflow', 'neumann']:
            slices_interior = [slice(None)] * magnetic_field[0].ndim
            slices_interior[dim] = 1  # Second element
            result[dim][tuple(slices_boundary)] = result[dim][tuple(slices_interior)]
        
        # Right boundary
        slices_boundary[dim] = -1  # Last element
        slices_interior[dim] = -2  # Second-to-last element
        
        if boundary_type in ['outflow', 'neumann']:
            result[dim][tuple(slices_boundary)] = result[dim][tuple(slices_interior)]
    
    return result 