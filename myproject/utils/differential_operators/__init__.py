"""
Differential operators module for iTensor.

This module implements differential operators (gradient, divergence, curl, Laplacian, d'Alembertian)
in general curvilinear coordinate systems for both symbolic and numerical calculations.
"""

# Differential operators package
# This package contains modules for calculating differential operators on tensor fields

# Import main functions for convenience
from .symbolic import (
    calculate_christoffel_symbols,
    gradient,
    divergence,
    curl,
    laplacian,
    covariant_derivative
)

# Import from numeric module
try:
    from .numeric import (
        evaluate_gradient,
        evaluate_divergence,
        evaluate_curl,
        evaluate_laplacian
    )
except ImportError:
    # Numeric implementation might not be available
    pass

# Import consistency checks
try:
    from .consistency_checks import (
        check_christoffel_symmetry,
        check_metric_compatibility
    )
except ImportError:
    # Consistency checks might not be implemented yet
    pass

from .transforms import (
    cartesian_to_curvilinear,
    curvilinear_to_cartesian,
    jacobian_matrix,
    metric_from_transformation,
    spherical_coordinates,
    cylindrical_coordinates,
    transform_scalar_field,
    transform_vector_field,
    transform_tensor_field
)

from .consistency_checks import (
    check_christoffel_symmetry_numeric,
    convert_to_orthonormal_basis_numeric
)

__all__ = [
    # Symbolic differential operators
    'calculate_christoffel_symbols',
    'gradient',
    'divergence',
    'curl',
    'laplacian',
    'covariant_derivative',
    
    # Numerical differential operators
    'evaluate_gradient',
    'evaluate_divergence',
    'evaluate_curl',
    'evaluate_laplacian',
    
    # Coordinate transformations
    'cartesian_to_curvilinear',
    'curvilinear_to_cartesian',
    'jacobian_matrix',
    'metric_from_transformation',
    'spherical_coordinates',
    'cylindrical_coordinates',
    'transform_scalar_field',
    'transform_vector_field',
    'transform_tensor_field',
    
    # Consistency checks
    'check_christoffel_symmetry',
    'check_metric_compatibility',
    'check_christoffel_symmetry_numeric',
    'convert_to_orthonormal_basis_numeric'
]

if __name__ == "__main__":
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== iTensor Differential Operators Demo ===\n")
    
    print("PART 1: SYMBOLIC CALCULATIONS WITH SYMPY")
    print("----------------------------------------")
    
    # Set up spherical coordinates
    print("Setting up spherical coordinates...")
    r, theta, phi = sp.symbols('r theta phi', real=True, positive=True)
    coords = [r, theta, phi]
    
    # Define a scalar field in spherical coordinates
    print("\nDefining a scalar field: f(r, theta, phi) = r² sin²(θ) cos(φ)")
    scalar_field = r**2 * sp.sin(theta)**2 * sp.cos(phi)
    
    # Set up transformation from spherical to Cartesian
    print("\nComputing the metric tensor for spherical coordinates...")
    transform_functions = [
        r * sp.sin(theta) * sp.cos(phi),  # x
        r * sp.sin(theta) * sp.sin(phi),  # y
        r * sp.cos(theta)                 # z
    ]
    
    # Compute the metric tensor for spherical coordinates
    cartesian_metric = sp.eye(3)  # Identity matrix for Euclidean space
    spherical_metric = metric_from_transformation(transform_functions, cartesian_metric, coords)
    
    print("Non-zero components of the metric tensor:")
    for i in range(3):
        for j in range(3):
            if spherical_metric[i, j] != 0:
                print(f"g_{i}{j} = {spherical_metric[i, j]}")
    
    # Compute Christoffel symbols
    print("\nComputing Christoffel symbols...")
    christoffel = calculate_christoffel_symbols(spherical_metric, coords)
    
    print("Sample of non-zero Christoffel symbols:")
    shown = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if christoffel[i][j][k] != 0 and shown < 5:
                    print(f"Γ^{i}_{j}{k} = {sp.simplify(christoffel[i][j][k])}")
                    shown += 1
    print("(showing 5 of many non-zero components)")
    
    # Compute the gradient
    print("\nComputing the gradient of the scalar field...")
    grad_f = gradient(scalar_field, spherical_metric, coords)
    
    print("Contravariant components of the gradient:")
    for i in range(3):
        print(f"∇^{i}f = {sp.simplify(grad_f[i])}")
    
    # Compute the Laplacian
    print("\nComputing the Laplacian of the scalar field...")
    lap_f = laplacian(scalar_field, spherical_metric, coords)
    print(f"∇²f = {sp.simplify(lap_f)}")
    
    # Compute divergence of a vector field
    print("\nComputing the divergence of a vector field V = [r sin(θ), r cos(θ), 0]...")
    vector_field = [r * sp.sin(theta), r * sp.cos(theta), 0]
    div_v = divergence(vector_field, spherical_metric, coords)
    print(f"∇·V = {sp.simplify(div_v)}")
    
    print("\nPART 2: NUMERICAL CALCULATIONS WITH NUMPY")
    print("----------------------------------------")
    
    # Create a grid in spherical coordinates
    print("Creating a grid in spherical coordinates...")
    r_vals = np.linspace(0.1, 2.0, 20)
    theta_vals = np.linspace(0.1, np.pi-0.1, 20)
    phi_vals = np.linspace(0, 2*np.pi, 20)
    
    r_grid, theta_grid, phi_grid = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')
    grid = [r_grid, theta_grid, phi_grid]
    
    print(f"Grid shape: {r_grid.shape}")
    
    # Define a scalar field on the grid
    print("\nDefining a scalar field on the grid: f(r, theta, phi) = r² sin²(θ) cos(φ)")
    scalar_field_grid = r_grid**2 * np.sin(theta_grid)**2 * np.cos(phi_grid)
    
    # Create the metric tensor
    print("\nComputing the metric tensor on the grid...")
    metric = np.zeros((3, 3) + r_grid.shape)
    for i in range(r_grid.shape[0]):
        for j in range(r_grid.shape[1]):
            for k in range(r_grid.shape[2]):
                r_val, theta_val = r_grid[i,j,k], theta_grid[i,j,k]
                # Diagonal metric for spherical coordinates
                metric[0, 0, i, j, k] = 1.0  # g_rr = 1
                metric[1, 1, i, j, k] = r_val**2  # g_θθ = r²
                metric[2, 2, i, j, k] = r_val**2 * np.sin(theta_val)**2  # g_φφ = r²sin²θ
    
    # Compute the inverse metric
    print("Computing the inverse metric tensor...")
    metric_inverse = np.zeros_like(metric)
    for i in range(r_grid.shape[0]):
        for j in range(r_grid.shape[1]):
            for k in range(r_grid.shape[2]):
                metric_inverse[:, :, i, j, k] = np.linalg.inv(metric[:, :, i, j, k])
    
    # Compute the gradient
    print("\nComputing the gradient numerically...")
    gradient_field = evaluate_gradient(scalar_field_grid, metric_inverse, grid)
    
    # Compute the Laplacian
    print("Computing the Laplacian numerically...")
    laplacian_field = evaluate_laplacian(scalar_field_grid, metric, metric_inverse, grid)
    
    # Create a vector field
    print("\nDefining a vector field V = [r sin(θ), r cos(θ), 0] on the grid")
    vector_field_grid = [
        r_grid * np.sin(theta_grid),
        r_grid * np.cos(theta_grid),
        np.zeros_like(r_grid)
    ]
    
    # Compute the divergence
    print("Computing the divergence of the vector field...")
    divergence_field = evaluate_divergence(vector_field_grid, metric, grid)
    
    # Visualize some results
    print("\nVisualizing the scalar field at phi = 0...")
    plt.figure(figsize=(10, 6))
    phi_slice = 0  # Show a 2D slice at phi = 0
    
    plt.subplot(1, 2, 1)
    plt.title('Scalar field at φ = 0')
    contour = plt.contourf(r_grid[:, :, phi_slice], theta_grid[:, :, phi_slice], 
                         scalar_field_grid[:, :, phi_slice], 20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('r')
    plt.ylabel('θ')
    
    plt.subplot(1, 2, 2)
    plt.title('Laplacian at φ = 0')
    contour = plt.contourf(r_grid[:, :, phi_slice], theta_grid[:, :, phi_slice], 
                         laplacian_field[:, :, phi_slice], 20, cmap='plasma')
    plt.colorbar(contour)
    plt.xlabel('r')
    plt.ylabel('θ')
    
    plt.tight_layout()
    
    print("\nPART 3: COORDINATE TRANSFORMATIONS")
    print("----------------------------------------")
    
    # Define a point in Cartesian and convert to spherical
    print("Converting between Cartesian and spherical coordinates...")
    cartesian_point = np.array([1.0, 1.0, 1.0])
    
    def cartesian_to_spherical(cartesian_coords):
        x, y, z = cartesian_coords
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi])
    
    def spherical_to_cartesian(spherical_coords):
        r, theta, phi = spherical_coords
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])
    
    spherical_point = cartesian_to_spherical(cartesian_point)
    cartesian_again = spherical_to_cartesian(spherical_point)
    
    print(f"Cartesian point: {cartesian_point}")
    print(f"Converted to spherical: {spherical_point}")
    print(f"Back to Cartesian: {cartesian_again}")
    print(f"Difference: {np.linalg.norm(cartesian_point - cartesian_again)}")
    
    # Compute the Jacobian 
    print("\nComputing the Jacobian of the transformation...")
    jacobian = jacobian_matrix(spherical_to_cartesian, spherical_point)
    print("Jacobian (∂(x,y,z)/∂(r,θ,φ)):")
    print(jacobian)
    
    print("\nMake sure to explore the other modules:")
    print("- symbolic.py: For symbolic calculations with SymPy")
    print("- numeric.py: For numerical calculations with NumPy")
    print("- transforms.py: For coordinate transformations")
    
    print("\nPlot will be displayed. Close the plot window to exit the demo.")
    plt.show() 