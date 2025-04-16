"""
Coordinate transformation utilities for differential operators.

This module provides functionality to:
1. Transform between different coordinate systems
2. Compute metric tensors from coordinate transformations
3. Map fields between different coordinate systems

Supports both symbolic (using SymPy) and numerical (using NumPy) calculations.
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Tuple, Union, Callable, Optional

# Implementation plan (to be implemented):

# 1. COORDINATE TRANSFORMATIONS
# - cartesian_to_curvilinear(transform_functions, coords)
#   * Defines a mapping from Cartesian (x,y,z) to curvilinear coordinates
#   * transform_functions is a list of transformation expressions
#   * Returns the transformation map and its Jacobian

# - curvilinear_to_cartesian(inverse_transform_functions, coords)
#   * Defines a mapping from curvilinear back to Cartesian coordinates
#   * Useful for visualizing results in Cartesian space

# 2. METRIC COMPUTATION
# - metric_from_transformation(transform_map, base_metric, coords)
#   * Computes the metric tensor in curvilinear coordinates
#   * Uses the Jacobian of the transformation: g_ij = Σ_kl (∂x^k/∂u^i)(∂x^l/∂u^j)g_kl
#   * base_metric is typically the Cartesian metric (identity for Euclidean, Minkowski for spacetime)

# - jacobian_matrix(transform_map, coords)
#   * Computes the Jacobian matrix of a coordinate transformation
#   * J_ij = ∂x^i/∂u^j

# 3. COMMON COORDINATE SYSTEMS
# - spherical_coordinates(r, theta, phi)
#   * Standard spherical coordinate system
#   * x = r sin(θ) cos(φ), y = r sin(θ) sin(φ), z = r cos(θ)

# - cylindrical_coordinates(rho, phi, z)
#   * Standard cylindrical coordinate system
#   * x = ρ cos(φ), y = ρ sin(φ), z = z

# - general_ellipsoidal_coordinates(a, b, c, lambda1, lambda2, lambda3)
#   * Ellipsoidal coordinate system with semi-axes a, b, c
#   * More general than spherical or cylindrical

# 4. FIELD TRANSFORMATIONS
# - transform_scalar_field(field, old_coords, new_coords, transform_map)
#   * Transforms a scalar field from one coordinate system to another
#   * Scalar fields transform trivially: f'(u) = f(x(u))

# - transform_vector_field(field, old_coords, new_coords, transform_map, is_contravariant=True)
#   * Transforms a vector field from one coordinate system to another
#   * Uses the Jacobian for the transformation

# - transform_tensor_field(field, old_coords, new_coords, transform_map, contravariant_indices, covariant_indices)
#   * Transforms a general tensor field between coordinate systems
#   * Uses appropriate transformation rules for each index

# These utilities will enable working with any curvilinear coordinate system
# by either directly specifying the metric or deriving it from a coordinate transformation. 

def cartesian_to_curvilinear(transform_functions: List[sp.Expr], 
                            coords: List[sp.Symbol]) -> Tuple[Dict[sp.Symbol, sp.Expr], sp.Matrix]:
    """
    Define a mapping from Cartesian coordinates to curvilinear coordinates.
    
    Args:
        transform_functions: List of expressions defining the transformation from curvilinear to Cartesian
        coords: List of curvilinear coordinate symbols
        
    Returns:
        Tuple of (transformation_map, jacobian) where:
        - transformation_map is a dictionary mapping Cartesian symbols to expressions in curvilinear coords
        - jacobian is the Jacobian matrix of the transformation
    """
    n = len(coords)
    
    # Define Cartesian coordinates as symbols
    cartesian_coords = [sp.Symbol(f'x{i}') for i in range(n)]
    
    # Create the transformation map
    transform_map = {cartesian_coords[i]: transform_functions[i] for i in range(n)}
    
    # Compute the Jacobian matrix: J_{ij} = ∂x^i/∂u^j
    jacobian = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            jacobian[i, j] = sp.diff(transform_functions[i], coords[j])
    
    return transform_map, jacobian

def curvilinear_to_cartesian(inverse_transform_functions: List[sp.Expr], 
                            coords: List[sp.Symbol]) -> Tuple[Dict[sp.Symbol, sp.Expr], sp.Matrix]:
    """
    Define a mapping from curvilinear coordinates back to Cartesian coordinates.
    
    Args:
        inverse_transform_functions: List of expressions defining the transformation from Cartesian to curvilinear
        coords: List of Cartesian coordinate symbols
        
    Returns:
        Tuple of (inverse_transformation_map, inverse_jacobian) where:
        - inverse_transformation_map is a dictionary mapping curvilinear symbols to expressions in Cartesian coords
        - inverse_jacobian is the Jacobian matrix of the inverse transformation
    """
    n = len(coords)
    
    # Define curvilinear coordinates as symbols
    curvilinear_coords = [sp.Symbol(f'u{i}') for i in range(n)]
    
    # Create the inverse transformation map
    inverse_transform_map = {curvilinear_coords[i]: inverse_transform_functions[i] for i in range(n)}
    
    # Compute the inverse Jacobian matrix: J^{-1}_{ij} = ∂u^i/∂x^j
    inverse_jacobian = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            inverse_jacobian[i, j] = sp.diff(inverse_transform_functions[i], coords[j])
    
    return inverse_transform_map, inverse_jacobian

def jacobian_matrix(transform_map: Union[List[sp.Expr], Callable], 
                   coords: Union[List[sp.Symbol], np.ndarray]) -> Union[sp.Matrix, np.ndarray]:
    """
    Compute the Jacobian matrix of a coordinate transformation.
    
    Args:
        transform_map: Either:
                      - List of expressions defining the transformation 
                      - Function that takes coordinates and returns transformed coordinates
        coords: Either:
               - List of coordinate symbols (for symbolic calculation)
               - Numpy array of coordinate values (for numeric calculation)
        
    Returns:
        Jacobian matrix J_{ij} = ∂x^i/∂u^j
    """
    # Check if we're working symbolically or numerically
    if (isinstance(transform_map, list) and all(isinstance(t, sp.Expr) for t in transform_map) and 
        all(isinstance(c, sp.Symbol) for c in coords)):
        # Symbolic calculation
        n = len(coords)
        jacobian = sp.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                jacobian[i, j] = sp.diff(transform_map[i], coords[j])
        
        return jacobian
    
    else:
        # Numeric calculation using finite differences
        if callable(transform_map):
            # If transform_map is a function, we use numerical differentiation
            coords_array = np.asarray(coords)
            n = len(coords_array)
            
            # Initialize Jacobian matrix
            jacobian = np.zeros((n, n))
            
            # Small step for finite difference
            h = 1e-6
            
            # Base point transformation
            base_point_transformed = transform_map(coords_array)
            
            # Compute Jacobian using finite differences
            for j in range(n):
                # Create a small perturbation in the j-th coordinate
                perturbed_coords = coords_array.copy()
                perturbed_coords[j] += h
                
                # Compute transformation of perturbed point
                perturbed_transformed = transform_map(perturbed_coords)
                
                # Compute partial derivatives using finite differences
                jacobian[:, j] = (perturbed_transformed - base_point_transformed) / h
            
            return jacobian
        else:
            raise ValueError("transform_map must be either a list of symbolic expressions or a callable function")

def metric_from_transformation(transform_map: List[sp.Expr], base_metric: sp.Matrix, 
                              coords: List[sp.Symbol]) -> sp.Matrix:
    """
    Compute the metric tensor in curvilinear coordinates from a coordinate transformation.
    
    Args:
        transform_map: List of expressions defining the transformation from curvilinear to Cartesian
        base_metric: Metric tensor in the base coordinate system (typically Cartesian)
        coords: List of curvilinear coordinate symbols
        
    Returns:
        Metric tensor in the curvilinear coordinate system
    """
    # Compute the Jacobian matrix
    J = jacobian_matrix(transform_map, coords)
    
    # Compute the metric using the formula: g_{ij} = (∂x^k/∂u^i) * g_{kl} * (∂x^l/∂u^j)
    # or in matrix form: g = J^T * base_metric * J
    metric = J.transpose() * base_metric * J
    
    # Simplify each component
    for i in range(metric.shape[0]):
        for j in range(metric.shape[1]):
            metric[i, j] = sp.simplify(metric[i, j])
    
    return metric

def spherical_coordinates(r: sp.Symbol, theta: sp.Symbol, phi: sp.Symbol) -> List[sp.Expr]:
    """
    Standard spherical coordinate transformation to Cartesian.
    
    Args:
        r: Radial coordinate symbol
        theta: Polar angle coordinate symbol
        phi: Azimuthal angle coordinate symbol
        
    Returns:
        List of expressions for the Cartesian coordinates [x, y, z] in terms of spherical coordinates
    """
    x = r * sp.sin(theta) * sp.cos(phi)
    y = r * sp.sin(theta) * sp.sin(phi)
    z = r * sp.cos(theta)
    
    return [x, y, z]

def cylindrical_coordinates(rho: sp.Symbol, phi: sp.Symbol, z: sp.Symbol) -> List[sp.Expr]:
    """
    Standard cylindrical coordinate transformation to Cartesian.
    
    Args:
        rho: Radial coordinate symbol in the xy-plane
        phi: Azimuthal angle coordinate symbol
        z: Height coordinate symbol
        
    Returns:
        List of expressions for the Cartesian coordinates [x, y, z] in terms of cylindrical coordinates
    """
    x = rho * sp.cos(phi)
    y = rho * sp.sin(phi)
    # z = z (unchanged)
    
    return [x, y, z]

def general_ellipsoidal_coordinates(a: sp.Symbol, b: sp.Symbol, c: sp.Symbol, 
                                   lambda1: sp.Symbol, lambda2: sp.Symbol, lambda3: sp.Symbol) -> List[sp.Expr]:
    """
    General ellipsoidal coordinate transformation to Cartesian.
    
    Args:
        a, b, c: Semi-axes of the ellipsoid
        lambda1, lambda2, lambda3: Ellipsoidal coordinate symbols
        
    Returns:
        List of expressions for the Cartesian coordinates [x, y, z] in terms of ellipsoidal coordinates
    """
    # This is a simplified version - the actual transformation is more complex
    # See literature for proper ellipsoidal coordinates
    x = a * lambda1
    y = b * lambda2
    z = c * lambda3
    
    return [x, y, z]

def transform_scalar_field(field: sp.Expr, old_coords: List[sp.Symbol], 
                          new_coords: List[sp.Symbol], transform_map: Dict[sp.Symbol, sp.Expr]) -> sp.Expr:
    """
    Transform a scalar field from one coordinate system to another.
    
    Args:
        field: Scalar field expression in old coordinates
        old_coords: List of old coordinate symbols
        new_coords: List of new coordinate symbols
        transform_map: Dictionary mapping old coordinate symbols to expressions in new coordinates
        
    Returns:
        Scalar field expression in new coordinates
    """
    # For a scalar field, we just substitute the coordinates according to the transformation
    transformed_field = field.subs(transform_map)
    
    return sp.simplify(transformed_field)

def transform_vector_field(field: List[sp.Expr], old_coords: List[sp.Symbol], 
                          new_coords: List[sp.Symbol], transform_map: Dict[sp.Symbol, sp.Expr], 
                          is_contravariant: bool = True) -> List[sp.Expr]:
    """
    Transform a vector field from one coordinate system to another.
    
    Args:
        field: Vector field components in old coordinates
        old_coords: List of old coordinate symbols
        new_coords: List of new coordinate symbols
        transform_map: Dictionary mapping old coordinate symbols to expressions in new coordinates
        is_contravariant: Whether the vector field has contravariant components
        
    Returns:
        Vector field components in new coordinates
    """
    n = len(old_coords)
    
    # Compute the Jacobian and its inverse
    # Note: we need to extract the list of expressions from the transform_map
    transform_list = [transform_map[old_coords[i]] for i in range(n)]
    
    # For contravariant vectors (typically written with upper indices), use the Jacobian
    # For covariant vectors (lower indices), use the inverse of the transpose of the Jacobian
    if is_contravariant:
        # Compute the inverse Jacobian: (∂u^i/∂x^j)
        # We compute Jacobian of the inverse transformation
        inverse_jacobian = sp.zeros(n, n)
        
        # We need expressions for new coordinates in terms of old
        # This is typically difficult to compute symbolically
        # For practical use, specific coordinate transformations should be hardcoded
        # Here we use a numerical approximation or assume the inverse is known
        
        # For demonstration, we'll use a simple case where the inverse is known
        # In practice, this should be computed for specific transformations
        for i in range(n):
            for j in range(n):
                inverse_jacobian[i, j] = sp.diff(new_coords[i], old_coords[j])
        
        # Transform the vector field using the inverse Jacobian: V'^i = (∂u^i/∂x^j) * V^j
        transformed_field = [sp.S.Zero for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                transformed_field[i] += inverse_jacobian[i, j] * field[j]
            
            transformed_field[i] = sp.simplify(transformed_field[i])
    
    else:  # Covariant vector
        # Compute the Jacobian: (∂x^i/∂u^j)
        jacobian = sp.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                jacobian[i, j] = sp.diff(transform_list[i], new_coords[j])
        
        # Transform the vector field using the transpose of the Jacobian: V'_i = (∂x^j/∂u^i) * V_j
        transformed_field = [sp.S.Zero for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                transformed_field[i] += jacobian[j, i] * field[j]
            
            transformed_field[i] = sp.simplify(transformed_field[i])
    
    return transformed_field

def transform_tensor_field(field: List[List[sp.Expr]], old_coords: List[sp.Symbol], 
                          new_coords: List[sp.Symbol], transform_map: Dict[sp.Symbol, sp.Expr], 
                          contravariant_indices: List[int], covariant_indices: List[int]) -> List[List[sp.Expr]]:
    """
    Transform a tensor field from one coordinate system to another.
    
    Args:
        field: Tensor field components in old coordinates
        old_coords: List of old coordinate symbols
        new_coords: List of new coordinate symbols
        transform_map: Dictionary mapping old coordinate symbols to expressions in new coordinates
        contravariant_indices: List of contravariant index positions (0-based)
        covariant_indices: List of covariant index positions (0-based)
        
    Returns:
        Tensor field components in new coordinates
    """
    n = len(old_coords)
    
    # Compute the Jacobian and its inverse
    transform_list = [transform_map[old_coords[i]] for i in range(n)]
    
    # Compute the Jacobian: (∂x^i/∂u^j)
    jacobian = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            jacobian[i, j] = sp.diff(transform_list[i], new_coords[j])
    
    # Compute the inverse Jacobian: (∂u^i/∂x^j)
    # For demonstration, we'll compute it directly - in practice, this would be known for specific transforms
    inverse_jacobian = jacobian.inv()
    
    # For rank 2 tensor, initialize result
    transformed_field = [[sp.S.Zero for _ in range(n)] for _ in range(n)]
    
    # Apply transformation rule for each component
    for i in range(n):
        for j in range(n):
            # Sum over old indices
            for k in range(n):
                for l in range(n):
                    # Apply appropriate transformation for each index type
                    term = field[k][l]
                    
                    # For each contravariant index, multiply by inverse Jacobian
                    if 0 in contravariant_indices:  # First index is contravariant
                        term *= inverse_jacobian[i, k]
                    else:  # First index is covariant
                        term *= jacobian[k, i]
                        
                    if 1 in contravariant_indices:  # Second index is contravariant
                        term *= inverse_jacobian[j, l]
                    else:  # Second index is covariant
                        term *= jacobian[l, j]
                    
                    transformed_field[i][j] += term
            
            transformed_field[i][j] = sp.simplify(transformed_field[i][j])
    
    return transformed_field

if __name__ == "__main__":
    print("Coordinate Transforms - Demo\n")
    
    # Example 1: Define Cartesian to Spherical transformation
    print("1. Cartesian to Spherical coordinates")
    x, y, z = 1.0, 1.0, 1.0  # Cartesian point (1,1,1)
    
    # Formula for transformation from Cartesian to Spherical
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    
    print(f"Cartesian coordinates (x,y,z): ({x}, {y}, {z})")
    print(f"Spherical coordinates (r,θ,φ): ({r}, {theta}, {phi})")
    
    # Example 2: Define transformation and compute Jacobian
    print("\n2. Computing the Jacobian matrix for Cartesian to Spherical")
    
    def cartesian_to_spherical(cartesian_coords):
        x, y, z = cartesian_coords
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi])
    
    # Compute Jacobian numerically
    cartesian_point = np.array([1.0, 1.0, 1.0])
    jacobian_numeric = jacobian_matrix(cartesian_to_spherical, cartesian_point)
    
    print("Jacobian matrix (∂(r,θ,φ)/∂(x,y,z)) computed numerically:")
    print(jacobian_numeric)
    
    # Example 3: Compute Jacobian symbolically
    print("\n3. Computing the Jacobian matrix symbolically")
    
    # Define symbolic variables
    x_sym, y_sym, z_sym = sp.symbols('x y z', real=True)
    
    # Define the symbolic transformation expressions
    r_sym = sp.sqrt(x_sym**2 + y_sym**2 + z_sym**2)
    theta_sym = sp.acos(z_sym / r_sym)
    phi_sym = sp.atan2(y_sym, x_sym)
    
    transform_expr = [r_sym, theta_sym, phi_sym]
    coords_sym = [x_sym, y_sym, z_sym]
    
    # Compute Jacobian symbolically
    jacobian_symbolic = jacobian_matrix(transform_expr, coords_sym)
    
    print("Symbolic Jacobian matrix:")
    print(jacobian_symbolic)
    
    # Example 4: Using the function to convert a point
    print("\n4. Converting coordinates using transformation function")
    cartesian_point = np.array([2.0, 0.0, 2.0])  # Point (2,0,2) in Cartesian
    spherical_point = cartesian_to_spherical(cartesian_point)
    print(f"Cartesian point: {cartesian_point}")
    print(f"Spherical point: {spherical_point}")
    
    # Example 5: Define spherical to Cartesian transformation
    print("\n5. Spherical to Cartesian coordinates")
    
    def spherical_to_cartesian(spherical_coords):
        r, theta, phi = spherical_coords
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])
    
    # Convert back to Cartesian to verify
    cartesian_result = spherical_to_cartesian(spherical_point)
    print(f"Original Cartesian: {cartesian_point}")
    print(f"Spherical: {spherical_point}")
    print(f"Back to Cartesian: {cartesian_result}")
    print(f"Difference: {np.linalg.norm(cartesian_point - cartesian_result)}")
    
    # Example 6: Transform a vector between coordinate systems
    print("\n6. Transforming a vector field between coordinate systems")
    
    # Define a vector in Cartesian coordinates at point (2,0,2)
    vector_cartesian = np.array([1.0, 1.0, 0.0])  # Vector pointing in the x-y plane
    
    # Compute the Jacobian for spherical to Cartesian at the given point
    J_s_to_c = jacobian_matrix(spherical_to_cartesian, spherical_point)
    
    # Compute the inverse Jacobian (Cartesian to spherical)
    J_c_to_s = np.linalg.inv(J_s_to_c)
    
    # Transform the vector from Cartesian to spherical coordinates
    vector_spherical = np.dot(J_c_to_s, vector_cartesian)
    
    print(f"Vector in Cartesian coordinates: {vector_cartesian}")
    print(f"Vector in Spherical coordinates: {vector_spherical}")
    
    # Transform back to verify
    vector_cartesian_again = np.dot(J_s_to_c, vector_spherical)
    print(f"Vector back in Cartesian coordinates: {vector_cartesian_again}")
    print(f"Difference: {np.linalg.norm(vector_cartesian - vector_cartesian_again)}")
    
    # Example 7: Transform a tensor between coordinate systems
    print("\n7. Transforming a 2nd-rank tensor between coordinate systems")
    
    # Define a simple 2nd rank tensor in Cartesian coordinates (identity matrix for simplicity)
    tensor_cartesian = np.eye(3)
    
    # Function to transform a tensor (simplified version for demo)
    def transform_tensor(tensor, J):
        # For simplicity, we assume tensor is contravariant
        # T'^{ij} = J^i_a J^j_b T^{ab}
        n = J.shape[0]
        tensor_transformed = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for a in range(n):
                    for b in range(n):
                        tensor_transformed[i, j] += J[i, a] * J[j, b] * tensor[a, b]
        return tensor_transformed
    
    # Transform the tensor to spherical coordinates
    tensor_spherical = transform_tensor(tensor_cartesian, J_c_to_s)
    
    print("Tensor in Cartesian coordinates:")
    print(tensor_cartesian)
    print("\nTensor in Spherical coordinates:")
    print(tensor_spherical)
    
    # Transform back to verify
    tensor_cartesian_again = transform_tensor(tensor_spherical, J_s_to_c)
    print("\nTensor back in Cartesian coordinates:")
    print(tensor_cartesian_again)
    print(f"Frobenius norm of difference: {np.linalg.norm(tensor_cartesian - tensor_cartesian_again)}")
    
    print("\nTo see more transforms and operations, modify the code in transforms.py") 