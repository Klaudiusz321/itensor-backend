"""
Symbolic implementation of differential operators in curvilinear coordinates.

This module provides functionality to compute:
1. Covariant derivatives of tensor fields
2. Gradient, divergence, curl, Laplacian, and d'Alembertian operators
3. Tensor index manipulation utilities

All calculations are performed symbolically using SymPy.
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Tuple, Union, Callable, Optional, Any

def compute_christoffel(metric: sp.Matrix, coords: List[sp.Symbol]) -> List[List[List[sp.Expr]]]:
    """
    Compute Christoffel symbols of the second kind from a metric tensor.
    
    Args:
        metric: Metric tensor as a SymPy matrix
        coords: List of coordinate symbols
        
    Returns:
        3D array of Christoffel symbols Γ^k_ij indexed as Gamma[k][i][j]
    """
    n = len(coords)
    g = metric
    g_inv = metric.inv()
    
    # Initialize Christoffel symbols array with zeros
    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    
    # Compute Christoffel symbols using the formula:
    # Γ^σ_μν = (1/2) g^σλ (∂_μ g_νλ + ∂_ν g_μλ - ∂_λ g_μν)
    for sigma in range(n):
        for mu in range(n):
            for nu in range(n):
                Gamma_sum = 0
                for lam in range(n):
                    partial_mu  = sp.diff(g[nu, lam], coords[mu])
                    partial_nu  = sp.diff(g[mu, lam], coords[nu])
                    partial_lam = sp.diff(g[mu, nu], coords[lam])
                    Gamma_sum += g_inv[sigma, lam] * (partial_mu + partial_nu - partial_lam)
                Gamma[sigma][mu][nu] = sp.simplify(sp.Rational(1, 2) * Gamma_sum)
    
    return Gamma

def covariant_derivative(
    tensor: Union[sp.Expr, List[sp.Expr], List[List[sp.Expr]]], 
    metric: sp.Matrix, 
    coords: List[sp.Symbol], 
    index_positions: Optional[List[bool]] = None,
    christoffel: Optional[List[List[List[sp.Expr]]]] = None,
    wrt_index: int = 0
) -> Union[List[sp.Expr], List[List[sp.Expr]], List[List[List[sp.Expr]]]]:
    """
    Compute the covariant derivative of a tensor field.
    
    Args:
        tensor: Tensor components (scalar, vector, or rank-2 tensor)
        metric: Metric tensor as a SymPy matrix
        coords: List of coordinate symbols
        index_positions: List of booleans indicating if each index is contravariant (True) or covariant (False)
                        If None, tensor is assumed to be fully contravariant
        christoffel: Pre-computed Christoffel symbols (if None, they will be computed)
        wrt_index: Index to differentiate with respect to (default: 0)
        
    Returns:
        Covariant derivative of the tensor as a tensor with one more index
    """
    n = len(coords)
    
    # Compute Christoffel symbols if not provided
    if christoffel is None:
        christoffel = compute_christoffel(metric, coords)
    
    # Handle scalar field (no indices)
    if isinstance(tensor, sp.Expr):
        # For a scalar, covariant derivative is just the partial derivative
        result = [sp.diff(tensor, coords[i]) for i in range(n)]
        return result
    
    # Handle vector field
    elif isinstance(tensor, list) and all(isinstance(t, sp.Expr) for t in tensor):
        # Default: assume contravariant vector if index_positions not specified
        if index_positions is None:
            index_positions = [True]
        
        result = [[sp.S.Zero for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                # Start with partial derivative term
                result[i][j] = sp.diff(tensor[i], coords[j])
                
                # Add Christoffel correction terms
                if index_positions[0]:  # Contravariant index
                    for k in range(n):
                        result[i][j] += christoffel[i][j][k] * tensor[k]
                else:  # Covariant index
                    for k in range(n):
                        result[i][j] -= christoffel[k][j][i] * tensor[k]
                        
                # Simplify the final expression
                result[i][j] = sp.simplify(result[i][j])
                
        return result
    
    # Handle rank-2 tensor
    elif isinstance(tensor, list) and all(isinstance(t, list) for t in tensor):
        # Default: assume contravariant tensor if index_positions not specified
        if index_positions is None:
            index_positions = [True, True]
            
        result = [[[sp.S.Zero for _ in range(n)] for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Start with partial derivative term
                    result[i][j][k] = sp.diff(tensor[i][j], coords[k])
                    
                    # Add Christoffel correction terms for each index
                    if index_positions[0]:  # First index contravariant
                        for l in range(n):
                            result[i][j][k] += christoffel[i][k][l] * tensor[l][j]
                    else:  # First index covariant
                        for l in range(n):
                            result[i][j][k] -= christoffel[l][k][i] * tensor[l][j]
                            
                    if index_positions[1]:  # Second index contravariant
                        for l in range(n):
                            result[i][j][k] += christoffel[j][k][l] * tensor[i][l]
                    else:  # Second index covariant
                        for l in range(n):
                            result[i][j][k] -= christoffel[l][k][j] * tensor[i][l]
                    
                    # Simplify the final expression
                    result[i][j][k] = sp.simplify(result[i][j][k])
                    
        return result
    
    else:
        raise ValueError("Unsupported tensor type or rank")

def levi_civita_tensor(metric: sp.Matrix, dimension: int) -> List:
    """
    Compute Levi-Civita tensor (fully contravariant or covariant)
    
    Args:
        metric: Metric tensor as a SymPy matrix
        dimension: Dimension of the space
        
    Returns:
        Levi-Civita tensor ε^{ijk} (contravariant) with dimension components
    """
    if dimension not in [3, 4]:
        raise ValueError("Levi-Civita tensor implemented only for dimensions 3 and 4")
    
    # Compute metric determinant
    g_det = metric.det()
    g_det_abs = sp.sqrt(sp.Abs(g_det))
    
    # Initialize tensor with zeros
    if dimension == 3:
        epsilon = [[[sp.S.Zero for _ in range(3)] for _ in range(3)] for _ in range(3)]
    else:  # dimension == 4
        epsilon = [[[[sp.S.Zero for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    # Fill in non-zero components using the permutation symbol
    # multiplied by 1/sqrt(|g|) for contravariant tensor
    
    # For 3D
    if dimension == 3:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Use sp.LeviCivita which returns the permutation symbol
                    perm_value = sp.LeviCivita(i, j, k)
                    if perm_value != 0:
                        epsilon[i][j][k] = perm_value / g_det_abs
    
    # For 4D
    else:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        perm_value = sp.LeviCivita(i, j, k, l)
                        if perm_value != 0:
                            epsilon[i][j][k][l] = perm_value / g_det_abs
    
    return epsilon

def raise_index(tensor: List, metric_inverse: sp.Matrix, index_pos: int = 0) -> List:
    """
    Raise an index of a tensor using the inverse metric.
    
    Args:
        tensor: Tensor with covariant index to be raised
        metric_inverse: Inverse metric tensor as a SymPy matrix
        index_pos: Position of the index to raise (0-based)
        
    Returns:
        Tensor with the specified index raised
    """
    n = metric_inverse.shape[0]
    
    # Handle vector (rank 1 tensor)
    if all(isinstance(t, sp.Expr) for t in tensor):
        if index_pos != 0:
            raise ValueError("Invalid index position for vector")
        
        result = [sp.S.Zero for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[i] += metric_inverse[i, j] * tensor[j]
            result[i] = sp.simplify(result[i])
        
        return result
    
    # Handle rank 2 tensor
    elif all(isinstance(t, list) for t in tensor):
        result = [[sp.S.Zero for _ in range(n)] for _ in range(n)]
        
        if index_pos == 0:  # Raise first index
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i][j] += metric_inverse[i, k] * tensor[k][j]
                    result[i][j] = sp.simplify(result[i][j])
        
        elif index_pos == 1:  # Raise second index
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i][j] += metric_inverse[j, k] * tensor[i][k]
                    result[i][j] = sp.simplify(result[i][j])
        
        else:
            raise ValueError("Invalid index position for rank 2 tensor")
        
        return result
    
    else:
        raise ValueError("Unsupported tensor type or rank")

def lower_index(tensor: List, metric: sp.Matrix, index_pos: int = 0) -> List:
    """
    Lower an index of a tensor using the metric.
    
    Args:
        tensor: Tensor with contravariant index to be lowered
        metric: Metric tensor as a SymPy matrix
        index_pos: Position of the index to lower (0-based)
        
    Returns:
        Tensor with the specified index lowered
    """
    n = metric.shape[0]
    
    # Convert any non-SymPy expressions to SymPy expressions
    if isinstance(tensor, list):
        tensor_converted = []
        for item in tensor:
            if isinstance(item, (int, float)):
                tensor_converted.append(sp.sympify(item))
            else:
                tensor_converted.append(item)
        tensor = tensor_converted
    
    # Handle vector (rank 1 tensor)
    if all(isinstance(t, sp.Expr) for t in tensor):
        if index_pos != 0:
            raise ValueError("Invalid index position for vector")
        
        result = [sp.S.Zero for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[i] += metric[i, j] * tensor[j]
            result[i] = sp.simplify(result[i])
        
        return result
    
    # Handle rank 2 tensor
    elif all(isinstance(t, list) for t in tensor):
        result = [[sp.S.Zero for _ in range(n)] for _ in range(n)]
        
        if index_pos == 0:  # Lower first index
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i][j] += metric[i, k] * tensor[k][j]
                    result[i][j] = sp.simplify(result[i][j])
        
        elif index_pos == 1:  # Lower second index
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i][j] += metric[j, k] * tensor[i][k]
                    result[i][j] = sp.simplify(result[i][j])
        
        else:
            raise ValueError("Invalid index position for rank 2 tensor")
        
        return result
    
    else:
        raise ValueError(f"Unsupported tensor type or rank: {type(tensor)}, elements: {[type(t) for t in tensor]}")

def gradient(scalar_field: sp.Expr, metric: sp.Matrix, coords: List[sp.Symbol], 
             contravariant: bool = True) -> List[sp.Expr]:
    """
    Compute the gradient of a scalar field.
    
    Args:
        scalar_field: Scalar field as a SymPy expression
        metric: Metric tensor as a SymPy matrix
        coords: List of coordinate symbols
        contravariant: If True, return contravariant components (with upper indices)
                      If False, return covariant components (with lower indices)
        
    Returns:
        Gradient of the scalar field as a vector
    """
    n = len(coords)
    
    # Compute partial derivatives (covariant components of gradient)
    grad_covariant = [sp.diff(scalar_field, coords[i]) for i in range(n)]
    
    # If contravariant components requested, convert using inverse metric
    if contravariant:
        g_inv = metric.inv()
        grad_contravariant = [sp.S.Zero for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                grad_contravariant[i] += g_inv[i, j] * grad_covariant[j]
            grad_contravariant[i] = sp.simplify(grad_contravariant[i])
        
        return grad_contravariant
    
    # Otherwise return covariant components
    return grad_covariant

def divergence(vector_field: List[sp.Expr], metric: sp.Matrix, coords: List[sp.Symbol], 
               is_contravariant: bool = True) -> sp.Expr:
    """
    Compute the divergence of a vector field.
    
    Args:
        vector_field: Vector field components
        metric: Metric tensor as a SymPy matrix
        coords: List of coordinate symbols
        is_contravariant: Whether the vector field has contravariant components
        
    Returns:
        Divergence of the vector field (scalar)
    """
    n = len(coords)
    
    # If vector is not contravariant, raise the index
    if not is_contravariant:
        g_inv = metric.inv()
        vector_contravariant = raise_index(vector_field, g_inv)
    else:
        vector_contravariant = vector_field
    
    # Compute metric determinant
    g_det = metric.det()
    g_det_sqrt = sp.sqrt(sp.Abs(g_det))
    
    # Compute divergence using the formula:
    # div V = (1/√|g|) ∂_i(√|g| V^i)
    div = sp.S.Zero
    
    for i in range(n):
        div += sp.diff(g_det_sqrt * vector_contravariant[i], coords[i])
    
    div = div / g_det_sqrt
    
    return sp.simplify(div)

def curl(vector_field: List[sp.Expr], metric: sp.Matrix, coords: List[sp.Symbol], 
         is_contravariant: bool = True) -> List[sp.Expr]:
    """
    Compute the curl of a vector field (only valid in 3D).
    
    Args:
        vector_field: Vector field components
        metric: Metric tensor as a SymPy matrix
        coords: List of coordinate symbols
        is_contravariant: Whether the vector field has contravariant components
        
    Returns:
        Curl of the vector field as a contravariant vector
    """
    n = len(coords)
    
    if n != 3:
        raise ValueError("Curl operator is only defined in 3D space")
    
    # Ensure vector_field elements are sympy expressions
    vector_field = [sp.sympify(v) if not isinstance(v, sp.Expr) else v for v in vector_field]
    
    # Convert to covariant components if needed
    if is_contravariant:
        try:
            vector_covariant = lower_index(vector_field, metric)
        except Exception as e:
            print(f"Error lowering index: {e}")
            print(f"Vector field: {vector_field}")
            print(f"Types: {[type(v) for v in vector_field]}")
            raise
    else:
        vector_covariant = vector_field
    
    # Compute covariant derivative of the vector field
    christoffel = compute_christoffel(metric, coords)
    nabla_v = covariant_derivative(vector_covariant, metric, coords, 
                                  index_positions=[False], 
                                  christoffel=christoffel)
    
    # Compute Levi-Civita tensor (contravariant)
    epsilon = levi_civita_tensor(metric, 3)
    
    # Compute curl using the formula:
    # (∇×V)^i = ε^{ijk} ∇_j V_k
    curl_vector = [sp.S.Zero for _ in range(3)]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                curl_vector[i] += epsilon[i][j][k] * nabla_v[k][j]
        curl_vector[i] = sp.simplify(curl_vector[i])
    
    return curl_vector

def laplacian(scalar_field: sp.Expr, metric: sp.Matrix, coords: List[sp.Symbol]) -> sp.Expr:
    """
    Compute the Laplacian of a scalar field.
    
    Args:
        scalar_field: Scalar field as a SymPy expression
        metric: Metric tensor as a SymPy matrix
        coords: List of coordinate symbols
        
    Returns:
        Laplacian of the scalar field (scalar)
    """
    n = len(coords)
    
    # Compute metric determinant
    g_det = metric.det()
    g_det_sqrt = sp.sqrt(sp.Abs(g_det))
    
    # Compute inverse metric
    g_inv = metric.inv()
    
    # Compute Laplacian using the formula:
    # ∇²f = (1/√|g|) ∂_i(√|g| g^{ij} ∂_j f)
    laplacian_value = sp.S.Zero
    
    for i in range(n):
        # Compute g^{ij} ∂_j f for this i
        term = sp.S.Zero
        for j in range(n):
            term += g_inv[i, j] * sp.diff(scalar_field, coords[j])
        
        # Multiply by √|g| and take derivative
        laplacian_value += sp.diff(g_det_sqrt * term, coords[i])
    
    # Divide by √|g|
    laplacian_value = laplacian_value / g_det_sqrt
    
    return sp.simplify(laplacian_value)

def dalembert(scalar_field: sp.Expr, metric: sp.Matrix, coords: List[sp.Symbol]) -> sp.Expr:
    """
    Compute the d'Alembertian (wave operator) of a scalar field.
    This is the generalization of the Laplacian to 4D spacetime.
    
    Args:
        scalar_field: Scalar field as a SymPy expression
        metric: Metric tensor as a SymPy matrix (4x4 for spacetime)
        coords: List of coordinate symbols
        
    Returns:
        d'Alembertian of the scalar field (scalar)
    """
    # The d'Alembertian is just the Laplacian generalized to 4D spacetime
    # The implementation is identical, but we expect a 4D metric
    if len(coords) != 4:
        raise ValueError("d'Alembertian operator is defined for 4D spacetime")
    
    return laplacian(scalar_field, metric, coords)

def metric_from_transformation(transform_map: List[sp.Expr], base_metric: sp.Matrix, 
                              coords: List[sp.Symbol]) -> sp.Matrix:
    """
    Compute the metric tensor in curvilinear coordinates from a coordinate transformation.
    
    Args:
        transform_map: List of expressions defining the transformation from base coords to curvilinear
        base_metric: Metric tensor in the base coordinate system
        coords: List of curvilinear coordinate symbols
        
    Returns:
        Metric tensor in the curvilinear coordinate system
    """
    n = len(coords)
    base_coords = [sp.Symbol(f'x{i}') for i in range(n)]  # Base coordinates (e.g., Cartesian)
    
    # Compute the Jacobian matrix: J_{ij} = ∂x^i/∂u^j
    jacobian = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            jacobian[i, j] = sp.diff(transform_map[i], coords[j])
    
    # Compute the metric using the formula: g_{ij} = ∑_{kl} J^T_{ik} * g_{kl} * J_{lj}
    new_metric = jacobian.transpose() * base_metric * jacobian
    
    # Simplify each component
    for i in range(n):
        for j in range(n):
            new_metric[i, j] = sp.simplify(new_metric[i, j])
    
    return new_metric

if __name__ == "__main__":
    print("Symbolic Differential Operators - Demo\n")
    
    # Example 1: Define coordinate system and variables
    print("1. Setting up spherical coordinates")
    r, theta, phi = sp.symbols('r theta phi', real=True, positive=True)
    coords = [r, theta, phi]
    
    # Define the transformation functions from spherical to Cartesian
    print("\n2. Defining coordinate transformation")
    x = r * sp.sin(theta) * sp.cos(phi)
    y = r * sp.sin(theta) * sp.sin(phi)
    z = r * sp.cos(theta)
    transform_functions = [x, y, z]
    
    print("Spherical to Cartesian transformations:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")
    
    # Example 2: Compute metric tensor for spherical coordinates
    print("\n3. Computing metric tensor")
    # Cartesian metric (identity for Euclidean space)
    cartesian_metric = sp.eye(3)
    
    # Compute the metric tensor for spherical coordinates
    spherical_metric = metric_from_transformation(transform_functions, cartesian_metric, coords)
    
    print("Metric tensor in spherical coordinates:")
    for i in range(3):
        for j in range(3):
            if spherical_metric[i, j] != 0:
                print(f"g_{i}{j} = {spherical_metric[i, j]}")
    
    # Example 3: Define a scalar field and compute its gradient
    print("\n4. Computing gradient of a scalar field")
    
    # Define a scalar field: f(r, theta, phi) = r^2 * sin(theta)
    scalar_field = r**2 * sp.sin(theta)
    print(f"Scalar field: f(r, theta, phi) = {scalar_field}")
    
    # Compute the gradient (contravariant and covariant)
    grad_contravariant = gradient(scalar_field, spherical_metric, coords, contravariant=True)
    grad_covariant = gradient(scalar_field, spherical_metric, coords, contravariant=False)
    
    print("Contravariant gradient (with upper indices):")
    for i in range(3):
        print(f"∇^{i}f = {grad_contravariant[i]}")
    
    print("\nCovariant gradient (with lower indices):")
    for i in range(3):
        print(f"∇_{i}f = {grad_covariant[i]}")
    
    # Example 4: Define a vector field and compute its divergence
    print("\n5. Computing divergence of a vector field")
    
    # Define a simple vector field: V = [r, 0, 0] in spherical coordinates
    vector_field = [r, 0, 0]
    print("Vector field (contravariant components):")
    for i in range(3):
        print(f"V^{i} = {vector_field[i]}")
    
    # Compute the divergence
    div = divergence(vector_field, spherical_metric, coords)
    
    print(f"Divergence of vector field: ∇·V = {div}")
    div_simplified = sp.simplify(div)
    print(f"Simplified divergence: ∇·V = {div_simplified}")
    
    # Example 5: Compute the Laplacian of a scalar field
    print("\n6. Computing Laplacian of a scalar field")
    
    # Compute the Laplacian of r^2
    scalar_field2 = r**2
    print(f"Scalar field: f(r, theta, phi) = {scalar_field2}")
    
    lap = laplacian(scalar_field2, spherical_metric, coords)
    
    print(f"Laplacian of scalar field: ∇²f = {lap}")
    lap_simplified = sp.simplify(lap)
    print(f"Simplified Laplacian: ∇²f = {lap_simplified}")
    
    # Example 6: Compute the curl of a vector field
    print("\n7. Computing curl of a vector field")
    
    # Define another vector field: V = [0, r, phi] in spherical coordinates
    vector_field2 = [0, r, phi]
    print("Vector field (contravariant components):")
    for i in range(3):
        print(f"V^{i} = {vector_field2[i]}")
    
    # Compute the curl
    curl_vector = curl(vector_field2, spherical_metric, coords)
    
    print("Curl of vector field (contravariant components):")
    for i in range(3):
        print(f"(∇×V)^{i} = {sp.simplify(curl_vector[i])}")
    
    # Example 7: Compute Christoffel symbols
    print("\n8. Computing Christoffel symbols")
    
    christoffel = compute_christoffel(spherical_metric, coords)
    
    print("Non-zero Christoffel symbols:")
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if christoffel[i][j][k] != 0:
                    print(f"Γ^{i}_{j}{k} = {sp.simplify(christoffel[i][j][k])}")
    
    print("\nTo see more examples and operations, modify the code in symbolic.py") 