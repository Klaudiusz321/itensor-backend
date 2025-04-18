"""
Symbolic implementation of differential operators in curvilinear coordinates.

This module provides functionality to compute:
1. Covariant derivatives of tensor fields
2. Gradient, divergence, curl, Laplacian, and d'Alembertian operators
3. Tensor index manipulation utilities

All calculations are performed symbolically using SymPy.
"""

import sympy as sp
from sympy import symbols, diff, simplify, Matrix, sqrt, Symbol
import numpy as np
from typing import Dict, List, Tuple, Union, Callable, Optional, Any
from sympy import symbols, diff, simplify, Matrix, sqrt

def calculate_christoffel_symbols(coordinates, metric):
    """Calculate Christoffel symbols from metric tensor.
    This is a fallback if pre-calculated symbols aren't provided."""
    
    if all(isinstance(c,str) for c in coordinates):
        coordinates = symbols(coordinates)
    elif all(isinstance(c, sp.Symbol) for c in coordinates):
        coord_symbols = coordinates
    else: 
        coord_symbols = [Symbol(str(c)) for c in coordinates]
    dimension = len(coordinates)
    
    
    # Convert metric to Matrix if it's a list
    if isinstance(metric, list):
        g = Matrix(metric)
    else:
        g = metric
        
    # Calculate inverse metric
    g_inv = g.inv()
    
    # Initialize Christoffel symbols
    christoffel = [[[0 for _ in range(dimension)] for _ in range(dimension)] for _ in range(dimension)]
    
    # Calculate Christoffel symbols
    for k in range(dimension):
        for i in range(dimension):
            for j in range(dimension):
                # Calculate Γᵏᵢⱼ
                sum_term = 0
                for l in range(dimension):
                    # ∂gᵢₗ/∂xʲ + ∂gⱼₗ/∂xⁱ - ∂gᵢⱼ/∂xˡ
                    term1 = diff(g[i, l], coord_symbols[j])
                    term2 = diff(g[j, l], coord_symbols[i])
                    term3 = diff(g[i, j], coord_symbols[l])
                    sum_term += g_inv[k, l] * (term1 + term2 - term3)
                
                christoffel[k][i][j] = simplify(sum_term / 2)
    
    return christoffel

def covariant_derivative(tensor_field, coordinates, metric, christoffel_symbols=None, index=None):
    """
    Calculate the covariant derivative of a tensor field.
    
    Args:
        tensor_field: The tensor field components (scalar, vector, or higher rank)
        coordinates (list): The coordinate variables as strings
        metric (list or Matrix): The metric tensor
        christoffel_symbols (list, optional): Pre-calculated Christoffel symbols
        index (int, optional): The index along which to take the derivative
        
    Returns:
        Covariant derivative of the tensor field
    """
    # Implementation will depend on tensor rank
    # For now, just implement vector covariant derivative
    
    # If tensor is a scalar (rank 0), just return gradient
    if not isinstance(tensor_field, list):
        return gradient(tensor_field, coordinates, metric, christoffel_symbols)
    
    # For vectors, implement full covariant derivative
    dimension = len(coordinates)
    coord_symbols = symbols(coordinates)
    
    # Convert vector field components to sympy expressions if they're strings
    vector = []
    for component in tensor_field:
        if isinstance(component, str):
            vector.append(sp.sympify(component))
        else:
            vector.append(component)
    
    # If not provided, calculate Christoffel symbols
    if christoffel_symbols is None:
        christoffel_symbols = calculate_christoffel_symbols(coordinates, metric)
    
    # Calculate covariant derivative of vector field
    # ∇ᵢvⱼ = ∂vⱼ/∂xⁱ - Γᵏᵢⱼvₖ
    
    covariant_derivative = [[0 for _ in range(dimension)] for _ in range(dimension)]
    
    for i in range(dimension):
        for j in range(dimension):
            # Start with partial derivative
            result = diff(vector[j], coord_symbols[i])
            
            # Subtract Christoffel symbol terms
            for k in range(dimension):
                result -= christoffel_symbols[k][i][j] * vector[k]
            
            covariant_derivative[i][j] = simplify(result)
    
    return covariant_derivative

def gradient(scalar_field, coordinates, metric, christoffel_symbols=None):
    """
    Calculate the gradient of a scalar field in curvilinear coordinates.
    …
    """
    dimension = len(coordinates)

    # ⬇️ NEW: ensure coord_symbols is always a list of Symbol, not try to
    #      call symbols() on an already‐constructed Symbol
    if coordinates and isinstance(coordinates[0], str):
        # you passed e.g. ['r','theta','phi']
        coord_symbols = symbols(coordinates)
    else:
        # you passed [r, theta, phi] which are already Symbol objects
        coord_symbols = coordinates  # no symbols() call here

    # parse the scalar field
    if isinstance(scalar_field, str):
        f = sp.sympify(scalar_field)
    else:
        f = scalar_field

    # build metric and its inverse
    g = Matrix(metric) if isinstance(metric, list) else metric
    g_inv = g.inv()

    # compute contravariant gradient ∇^i f = g^ij ∂f/∂x^j
    gradient_contravariant = []
    for i in range(dimension):
        comp = sum(
            g_inv[i, j] * diff(f, coord_symbols[j])
            for j in range(dimension)
        )
        gradient_contravariant.append(simplify(comp))

    # convert to covariant components ∇_i f = g_ij ∇^j f
    gradient_covariant = []
    for i in range(dimension):
        comp = sum(
            g[i, j] * gradient_contravariant[j]
            for j in range(dimension)
        )
        gradient_covariant.append(simplify(comp))

    return gradient_covariant

def divergence(vector_field, coordinates, metric, christoffel_symbols=None):
    """
    Calculate the divergence of a vector field in curvilinear coordinates.
    
    Args:
        vector_field (list): The contravariant components of the vector field
        coordinates (list): The coordinate variables as strings
        metric (list or Matrix): The metric tensor
        christoffel_symbols (list, optional): Pre-calculated Christoffel symbols
        
    Returns:
        Divergence as a sympy expression
    """
    if all(isinstance(c,str) for c in coordinates):
        coordinates = symbols(coordinates)
    elif all(isinstance(c, sp.Symbol) for c in coordinates):
        coord_symbols = coordinates
    else: 
        coord_symbols = [Symbol(str(c)) for c in coordinates]


    dimension = len(coordinates)
    
    
    # Convert vector field components to sympy expressions if they're strings
    vector = []
    for component in vector_field:
        if isinstance(component, str):
            vector.append(sp.sympify(component))
        else:
            vector.append(component)
    
    # Convert metric to Matrix if it's a list
    if isinstance(metric, list):
        g = Matrix(metric)
    else:
        g = metric
    
    # Get determinant of metric
    g_det = g.det()
    
    # If not provided, calculate Christoffel symbols
    if christoffel_symbols is None:
        christoffel_symbols = calculate_christoffel_symbols(coordinates, metric)
    
    # Calculate divergence using formula:
    # ∇·v = 1/√|g| ∂/∂xⁱ(√|g| vⁱ)
    div = 0
    g_det_sqrt = sqrt(abs(g_det))
    
    for i in range(dimension):
        term = diff(g_det_sqrt * vector[i], coord_symbols[i])
        div += term
    
    div = div / g_det_sqrt
    
    return simplify(div)

def curl(vector_field, coordinates, metric, christoffel_symbols=None):
    """
    Calculate the curl of a vector field in 3D curvilinear coordinates.
    """
    # 1) Normalize 'coordinates' into a list of Sympy Symbols:
    if all(isinstance(c, str) for c in coordinates):
        coord_symbols = list(symbols(coordinates))
    elif all(isinstance(c, Symbol) for c in coordinates):
        coord_symbols = list(coordinates)
    else:
        coord_symbols = [Symbol(str(c)) for c in coordinates]
    
    # Now coord_symbols is a proper list of Symbol objects.
    # Remove *any* further symbols(coordinates) calls below.

    # 2) Check dimension
    if len(coord_symbols) != 3:
        raise ValueError("Curl operation requires exactly 3 dimensions")
    dimension = 3

    # 3) Convert vector components to Sympy expressions
    vector = [
        sp.sympify(comp) if isinstance(comp, str) else comp
        for comp in vector_field
    ]

    # 4) Convert metric to Matrix
    if not isinstance(metric, Matrix):
        g = Matrix(metric)
    else:
        g = metric

    # 5) If needed, compute Christoffel symbols
    if christoffel_symbols is None:
        christoffel_symbols = calculate_christoffel_symbols(coord_symbols, g)

    # 6) Build curl
    g_det_sqrt = sqrt(abs(g.det()))
    curl_components = []
    
    for i in range(3):
        j, k = (i+1) % 3, (i+2) % 3
        # ordinary curl part
        term = diff(vector[k], coord_symbols[j]) - diff(vector[j], coord_symbols[k])
        # plus connection terms
        for l in range(3):
            term += christoffel_symbols[k][j][l]*vector[l] \
                  - christoffel_symbols[j][k][l]*vector[l]
        # adjust by metric determinant
        curl_components.append(simplify(term / g_det_sqrt))

    return curl_components

def laplacian(scalar_field, coordinates, metric, christoffel_symbols=None):
    """
    Calculate the Laplacian of a scalar field in curvilinear coordinates.
    
    Args:
        scalar_field (str or sympy expression): The scalar field function
        coordinates (list): The coordinate variables as strings
        metric (list or Matrix): The metric tensor
        christoffel_symbols (list, optional): Pre-calculated Christoffel symbols
        
    Returns:
        Laplacian as a sympy expression
    """
    if all(isinstance(c,str) for c in coordinates):
        coordinates = symbols(coordinates)
    elif all(isinstance(c, sp.Symbol) for c in coordinates):
        coord_symbols = coordinates
    else: 
        coord_symbols = [Symbol(str(c)) for c in coordinates]
    
    dimension = len(coordinates)
    
    
    # Parse scalar field if it's a string
    if isinstance(scalar_field, str):
        f = sp.sympify(scalar_field)
    else:
        f = scalar_field
    
    # Convert metric to Matrix if it's a list
    if isinstance(metric, list):
        g = Matrix(metric)
    else:
        g = metric
    
    # Calculate inverse metric
    g_inv = g.inv()
    
    # Get determinant of metric
    g_det = g.det()
    g_det_sqrt = sqrt(abs(g_det))
    
    # If not provided, calculate Christoffel symbols
    if christoffel_symbols is None:
        christoffel_symbols = calculate_christoffel_symbols(coordinates, metric)
    
    # Calculate Laplacian using the formula:
    # ∇²f = 1/√|g| ∂/∂xⁱ(√|g| gⁱʲ ∂f/∂xʲ)
    
    laplacian = 0
    
    for i in range(dimension):
        for j in range(dimension):
            # Calculate gⁱʲ ∂f/∂xʲ
            term = g_inv[i, j] * diff(f, coord_symbols[j])
            
            # Multiply by √|g|
            term = g_det_sqrt * term
            
            # Take derivative ∂/∂xⁱ of the whole term
            term = diff(term, coord_symbols[i])
            
            # Divide by √|g|
            term = term / g_det_sqrt
            
            laplacian += term
    
    return simplify(laplacian)

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
    grad_contravariant = gradient(scalar_field, coords, spherical_metric, christoffel_symbols=None)
    grad_covariant = gradient(scalar_field, coords, spherical_metric, christoffel_symbols=None)
    
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
    div = divergence(vector_field, coords, spherical_metric, christoffel_symbols=None)
    
    print(f"Divergence of vector field: ∇·V = {div}")
    div_simplified = sp.simplify(div)
    print(f"Simplified divergence: ∇·V = {div_simplified}")
    
    # Example 5: Compute the Laplacian of a scalar field
    print("\n6. Computing Laplacian of a scalar field")
    
    # Compute the Laplacian of r^2
    scalar_field2 = r**2
    print(f"Scalar field: f(r, theta, phi) = {scalar_field2}")
    
    lap = laplacian(scalar_field2, coords, spherical_metric, christoffel_symbols=None)
    
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
    curl_vector = curl(vector_field2, coords, spherical_metric, christoffel_symbols=None)
    
    print("Curl of vector field (contravariant components):")
    for i in range(3):
        print(f"(∇×V)^{i} = {sp.simplify(curl_vector[i])}")
    
    # Example 7: Compute Christoffel symbols
    print("\n8. Computing Christoffel symbols")
    
    christoffel = calculate_christoffel_symbols(coords, spherical_metric)
    
    print("Non-zero Christoffel symbols:")
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if christoffel[i][j][k] != 0:
                    print(f"Γ^{i}_{j}{k} = {sp.simplify(christoffel[i][j][k])}")
    
    print("\nTo see more examples and operations, modify the code in symbolic.py") 