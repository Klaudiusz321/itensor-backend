"""
Consistency checks for tensor calculations.

This module provides functions to verify the mathematical consistency
of tensor calculations in various coordinate systems. It includes checks
for Christoffel symbol symmetry, metric compatibility, and transformation
consistency, as well as utilities for basis conversions.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Callable, Union, Optional, Any
from sympy import Matrix, symbols, simplify, diff, sqrt, diag

# Symbolic consistency checks
def check_christoffel_symmetry(christoffel_symbols: Union[Dict[Tuple[int, int, int], Any], List[List[List[Any]]]], 
                               dimension: int, 
                               tolerance: float = 1e-10) -> bool:
    """
    Check the symmetry of Christoffel symbols: Γ^k_ij = Γ^k_ji

    Parameters
    ----------
    christoffel_symbols : Union[Dict[Tuple[int, int, int], Any], List[List[List[Any]]]
        Either a dictionary mapping (k, i, j) indices to the symbolic expression for Γ^k_ij
        or a 3D list with christoffel_symbols[k][i][j] for Γ^k_ij
    dimension : int
        Dimension of the space
    tolerance : float, optional
        Tolerance for numerical comparisons (for simplified expressions)

    Returns
    -------
    bool
        True if symmetry holds, False otherwise
    """
    # Convert list representation to dictionary if necessary
    if isinstance(christoffel_symbols, list):
        christoffel_dict = {}
        for k in range(dimension):
            for i in range(dimension):
                for j in range(dimension):
                    if christoffel_symbols[k][i][j] != 0:
                        christoffel_dict[(k, i, j)] = christoffel_symbols[k][i][j]
    else:
        christoffel_dict = christoffel_symbols

    for k in range(dimension):
        for i in range(dimension):
            for j in range(i+1, dimension):
                # For dictionary representation
                if isinstance(christoffel_dict, dict):
                    symbol_1 = christoffel_dict.get((k, i, j), 0)
                    symbol_2 = christoffel_dict.get((k, j, i), 0)
                # For list representation (fallback, should not reach here after conversion)
                else:
                    symbol_1 = christoffel_symbols[k][i][j]
                    symbol_2 = christoffel_symbols[k][j][i]
                
                difference = simplify(symbol_1 - symbol_2)
                
                # For symbolic expressions, check if the difference simplifies to zero
                if isinstance(difference, (sp.Expr, sp.Matrix)) and not difference.is_zero:
                    return False
                # For numerical values, check if the difference is within tolerance
                elif isinstance(difference, (int, float, complex, np.number)) and abs(difference) > tolerance:
                    return False
    
    return True

def check_metric_compatibility(metric: Matrix, 
                              christoffel_symbols: Union[Dict[Tuple[int, int, int], Any], List[List[List[Any]]]],
                              coordinates: List[sp.Symbol],
                              dimension: int) -> bool:
    """
    Check if the connection is metric-compatible: ∇_k g_ij = 0

    Parameters
    ----------
    metric : Matrix
        The metric tensor as a sympy Matrix
    christoffel_symbols : Union[Dict[Tuple[int, int, int], Any], List[List[List[Any]]]
        Either a dictionary mapping (k, i, j) indices to the symbolic expression for Γ^k_ij
        or a 3D list with christoffel_symbols[k][i][j] for Γ^k_ij
    coordinates : List[sp.Symbol]
        List of coordinate symbols
    dimension : int
        Dimension of the space

    Returns
    -------
    bool
        True if the connection is metric-compatible, False otherwise
    """
    # Convert list representation to dictionary if necessary
    if isinstance(christoffel_symbols, list):
        christoffel_dict = {}
        for k in range(dimension):
            for i in range(dimension):
                for j in range(dimension):
                    if christoffel_symbols[k][i][j] != 0:
                        christoffel_dict[(k, i, j)] = christoffel_symbols[k][i][j]
    else:
        christoffel_dict = christoffel_symbols

    for i in range(dimension):
        for j in range(dimension):
            for k in range(dimension):
                # Calculate ∇_k g_ij
                partial_derivative = diff(metric[i, j], coordinates[k])
                
                christoffel_sum = 0
                for l in range(dimension):
                    # For dictionary representation
                    if isinstance(christoffel_dict, dict):
                        christoffel_sum += (
                            christoffel_dict.get((l, k, i), 0) * metric[l, j] +
                            christoffel_dict.get((l, k, j), 0) * metric[i, l]
                        )
                    # For list representation (fallback, should not reach here after conversion)
                    else:
                        christoffel_sum += (
                            christoffel_symbols[l][k][i] * metric[l, j] +
                            christoffel_symbols[l][k][j] * metric[i, l]
                        )
                
                covariant_derivative = partial_derivative - christoffel_sum
                
                if not simplify(covariant_derivative).is_zero:
                    return False
    
    return True

def check_flat_metric_operators(metric: Matrix,
                               coordinates: List[sp.Symbol],
                               scalar_field: sp.Expr,
                               vector_field: List[sp.Expr],
                               dimension: int) -> bool:
    """
    Check if differential operators reduce to their standard form 
    when the metric is flat (Euclidean).

    Parameters
    ----------
    metric : Matrix
        The metric tensor as a sympy Matrix
    coordinates : List[sp.Symbol]
        List of coordinate symbols
    scalar_field : sp.Expr
        A test scalar field
    vector_field : List[sp.Expr]
        A test vector field
    dimension : int
        Dimension of the space

    Returns
    -------
    bool
        True if operators match standard flat space forms, False otherwise
    """
    from .symbolic import gradient, divergence, laplacian
    
    # Check if metric is flat (diagonal with constants)
    is_flat = True
    for i in range(dimension):
        for j in range(dimension):
            if i != j and not metric[i, j].is_zero:
                is_flat = False
                break
            if i == j and not metric[i, i].is_constant():
                is_flat = False
                break
    
    if not is_flat:
        # Not a flat metric, so this check doesn't apply
        return True
    
    # In flat space, the gradient of a scalar should be: ∂f/∂x^i
    grad_scalar = gradient(scalar_field, coordinates, metric)
    flat_grad = [diff(scalar_field, coord) for coord in coordinates]
    
    # In flat space, the divergence of a vector should be: ∑ ∂v^i/∂x^i
    div_vector = divergence(vector_field, coordinates, metric)
    flat_div = sum(diff(vector_field[i], coordinates[i]) for i in range(dimension))
    
    # In flat space, the Laplacian should be: ∑ ∂²f/∂x^i²
    lap_scalar = laplacian(scalar_field, coordinates, metric)
    flat_lap = sum(diff(scalar_field, coord, 2) for coord in coordinates)
    
    # Check if all operators match their flat space counterparts
    return (all(simplify(grad_scalar[i] - flat_grad[i]).is_zero for i in range(dimension)) and
            simplify(div_vector - flat_div).is_zero and
            simplify(lap_scalar - flat_lap).is_zero)

def check_diagonal_metric_consistency(diagonal_components: List[sp.Expr],
                                     coordinates: List[sp.Symbol],
                                     test_scalar: sp.Expr,
                                     dimension: int) -> bool:
    """
    Check if a diagonally-dependent metric yields correct tensor calculus identities.

    Parameters
    ----------
    diagonal_components : List[sp.Expr]
        The diagonal components of the metric tensor
    coordinates : List[sp.Symbol]
        List of coordinate symbols
    test_scalar : sp.Expr
        A test scalar field
    dimension : int
        Dimension of the space

    Returns
    -------
    bool
        True if tensor identities hold, False otherwise
    """
    from .symbolic import gradient, divergence, laplacian
    
    # Create diagonal metric
    metric = Matrix(dimension, dimension, lambda i, j: diagonal_components[i] if i == j else 0)
    
    # Check if ∇²f = div(grad(f))
    grad_f = gradient(test_scalar, coordinates, metric)
    div_grad_f = divergence(grad_f, coordinates, metric)
    lap_f = laplacian(test_scalar, coordinates, metric)
    
    # This identity should hold: div(grad(f)) = ∇²f
    return simplify(div_grad_f - lap_f).is_zero

def convert_to_orthonormal_basis(tensor: Union[sp.Expr, List[sp.Expr], Matrix],
                                metric: Matrix,
                                dimension: int,
                                rank: int = 1) -> Union[sp.Expr, List[sp.Expr], Matrix]:
    """
    Convert a tensor from coordinate basis to orthonormal basis.

    Parameters
    ----------
    tensor : Union[sp.Expr, List[sp.Expr], Matrix]
        Tensor in coordinate basis
    metric : Matrix
        The metric tensor
    dimension : int
        Dimension of the space
    rank : int, optional
        Rank of the tensor (1 for vectors, 2 for rank-2 tensors, etc.)

    Returns
    -------
    Union[sp.Expr, List[sp.Expr], Matrix]
        Tensor in orthonormal basis
    """
    # For scalar fields (rank 0), no transformation is needed
    if rank == 0 or isinstance(tensor, sp.Expr):
        return tensor
    
    # Get the transformation matrix (the vielbein or tetrad)
    # e^a_i is the transformation from coordinate to orthonormal basis
    # We need to solve: e^a_i e^b_j g_ij = η_ab (where η is the Minkowski/Euclidean metric)
    
    # For simplicity, assuming a diagonal metric for now
    vielbein = Matrix(dimension, dimension, lambda i, j: sqrt(abs(metric[i, i])) if i == j else 0)
    
    if rank == 1:
        # For a vector: v^a = e^a_i v^i
        if isinstance(tensor, list):
            tensor = Matrix(tensor)
        
        return vielbein * tensor
    
    elif rank == 2:
        # For a rank-2 tensor: T^ab = e^a_i e^b_j T^ij
        result = Matrix(dimension, dimension, lambda a, b: 0)
        
        for a in range(dimension):
            for b in range(dimension):
                for i in range(dimension):
                    for j in range(dimension):
                        result[a, b] += vielbein[a, i] * vielbein[b, j] * tensor[i, j]
        
        return result
    
    else:
        raise NotImplementedError("Conversion for tensors of rank > 2 not implemented yet")

def check_transformation_consistency(from_coords: List[sp.Expr],
                                    to_coords: List[sp.Expr],
                                    original_metric: Matrix,
                                    dimension: int) -> bool:
    """
    Check if a coordinate transformation is consistent.

    Parameters
    ----------
    from_coords : List[sp.Expr]
        Original coordinates expressed in terms of the new coordinates
    to_coords : List[sp.Expr]
        New coordinates expressed in terms of the original coordinates
    original_metric : Matrix
        The metric tensor in the original coordinates
    dimension : int
        Dimension of the space

    Returns
    -------
    bool
        True if the transformation is consistent, False otherwise
    """
    from .transforms import jacobian_matrix, metric_from_transformation
    
    # Compute Jacobian matrices
    J_forward = jacobian_matrix(to_coords, from_coords)
    J_backward = jacobian_matrix(from_coords, to_coords)
    
    # Check if J_forward * J_backward ≈ Identity matrix
    prod = J_forward * J_backward
    identity = Matrix.eye(dimension)
    
    # Check if the product is approximately the identity matrix
    if not all(simplify(prod[i, j] - identity[i, j]).is_zero 
               for i in range(dimension) for j in range(dimension)):
        return False
    
    # Get the transformed metric
    new_metric = metric_from_transformation(original_metric, J_backward)
    
    # Transform back to original coordinates
    transformed_back = metric_from_transformation(new_metric, J_forward)
    
    # Check if we recover the original metric
    return all(simplify(transformed_back[i, j] - original_metric[i, j]).is_zero 
               for i in range(dimension) for j in range(dimension))

def run_all_consistency_checks(metric: Matrix,
                              coordinates: List[sp.Symbol],
                              dimension: int,
                              test_scalar: Optional[sp.Expr] = None,
                              test_vector: Optional[List[sp.Expr]] = None) -> Dict[str, bool]:
    """
    Run all consistency checks for a given metric or coordinate transformation.

    Parameters
    ----------
    metric : Matrix
        The metric tensor
    coordinates : List[sp.Symbol]
        List of coordinate symbols
    dimension : int
        Dimension of the space
    test_scalar : sp.Expr, optional
        A test scalar field
    test_vector : List[sp.Expr], optional
        A test vector field

    Returns
    -------
    Dict[str, bool]
        Dictionary with results of each consistency check
    """
    from .symbolic import compute_christoffel
    
    results = {}
    
    # Compute Christoffel symbols
    christoffel = compute_christoffel(metric, coordinates)
    
    # Check Christoffel symbol symmetry
    results['christoffel_symmetry'] = check_christoffel_symmetry(christoffel, dimension)
    
    # Check metric compatibility
    results['metric_compatibility'] = check_metric_compatibility(metric, christoffel, coordinates, dimension)
    
    # If test fields are provided, run additional checks
    if test_scalar is not None:
        if test_vector is None:
            # Generate a simple test vector field (gradient of the scalar)
            test_vector = [diff(test_scalar, coord) for coord in coordinates]
        
        # Check flat space consistency
        results['flat_metric_operators'] = check_flat_metric_operators(
            metric, coordinates, test_scalar, test_vector, dimension
        )
        
        # Check diagonal metric consistency
        if all(metric[i, j].is_zero for i in range(dimension) for j in range(dimension) if i != j):
            diagonal_components = [metric[i, i] for i in range(dimension)]
            results['diagonal_metric_consistency'] = check_diagonal_metric_consistency(
                diagonal_components, coordinates, test_scalar, dimension
            )
    
    return results

# Numerical implementations for consistency checks
def check_christoffel_symmetry_numeric(christoffel_numeric: np.ndarray, 
                                      tolerance: float = 1e-10) -> bool:
    """
    Check the symmetry of Christoffel symbols numerically.

    Parameters
    ----------
    christoffel_numeric : np.ndarray
        Numerical array of shape (dimension, dimension, dimension) for Γ^k_ij
    tolerance : float, optional
        Tolerance for numerical comparisons

    Returns
    -------
    bool
        True if symmetry holds, False otherwise
    """
    dimension = christoffel_numeric.shape[0]
    
    for k in range(dimension):
        for i in range(dimension):
            for j in range(i+1, dimension):
                if abs(christoffel_numeric[k, i, j] - christoffel_numeric[k, j, i]) > tolerance:
                    return False
    
    return True

def convert_to_orthonormal_basis_numeric(tensor_numeric: np.ndarray,
                                        metric_numeric: np.ndarray,
                                        rank: int = 1) -> np.ndarray:
    """
    Convert a tensor from coordinate basis to orthonormal basis numerically.

    Parameters
    ----------
    tensor_numeric : np.ndarray
        Numerical tensor in coordinate basis
    metric_numeric : np.ndarray
        Numerical metric tensor
    rank : int, optional
        Rank of the tensor (1 for vectors, 2 for rank-2 tensors, etc.)

    Returns
    -------
    np.ndarray
        Tensor in orthonormal basis
    """
    dimension = metric_numeric.shape[0]
    
    # Compute vielbein (assuming diagonal metric for simplicity)
    vielbein = np.zeros((dimension, dimension))
    for i in range(dimension):
        vielbein[i, i] = np.sqrt(np.abs(metric_numeric[i, i]))
    
    if rank == 0:
        # Scalar, no transformation needed
        return tensor_numeric
    
    elif rank == 1:
        # Vector transformation
        result = np.zeros(dimension)
        for a in range(dimension):
            for i in range(dimension):
                result[a] += vielbein[a, i] * tensor_numeric[i]
        return result
    
    elif rank == 2:
        # Rank-2 tensor transformation
        result = np.zeros((dimension, dimension))
        for a in range(dimension):
            for b in range(dimension):
                for i in range(dimension):
                    for j in range(dimension):
                        result[a, b] += vielbein[a, i] * vielbein[b, j] * tensor_numeric[i, j]
        return result
    
    else:
        raise NotImplementedError("Conversion for tensors of rank > 2 not implemented yet") 