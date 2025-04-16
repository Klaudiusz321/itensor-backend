#!/usr/bin/env python3
"""
Demo script for iTensor differential operators.

This script demonstrates how to use the iTensor package to perform
differential operations in spherical coordinates, including both
symbolic calculations with SymPy and numerical calculations with NumPy.
"""

import sys
import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to the Python path to import the myproject package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myproject.utils.differential_operators import (
    # Symbolic operators
    gradient, divergence, laplacian, compute_christoffel, metric_from_transformation,
    # Numeric operators
    evaluate_gradient, evaluate_divergence, evaluate_laplacian
)

def symbolic_demo():
    """Run a demonstration of symbolic differential operators."""
    print("\n=== SYMBOLIC CALCULATIONS WITH SYMPY ===")
    
    # Define spherical coordinates
    r, theta, phi = sp.symbols('r theta phi', real=True, positive=True)
    coords = [r, theta, phi]
    
    # Define the transformation from spherical to Cartesian
    x = r * sp.sin(theta) * sp.cos(phi)
    y = r * sp.sin(theta) * sp.sin(phi)
    z = r * sp.cos(theta)
    transform_functions = [x, y, z]
    
    # Compute the metric tensor for spherical coordinates
    cartesian_metric = sp.eye(3)  # Identity matrix for Euclidean space
    spherical_metric = metric_from_transformation(transform_functions, cartesian_metric, coords)
    
    print("\nMetric tensor in spherical coordinates:")
    for i in range(3):
        for j in range(3):
            if spherical_metric[i, j] != 0:
                print(f"g_{i}{j} = {spherical_metric[i, j]}")
    
    # Define a scalar field: f(r, theta, phi) = r^2 * sin(theta)^2 * cos(phi)
    scalar_field = r**2 * sp.sin(theta)**2 * sp.cos(phi)
    print(f"\nScalar field: f(r, theta, phi) = {scalar_field}")
    
    # Compute the gradient
    grad_f = gradient(scalar_field, spherical_metric, coords)
    print("\nGradient components (contravariant):")
    for i in range(3):
        print(f"∇^{i}f = {sp.simplify(grad_f[i])}")
    
    # Define a vector field: V = [r*sin(theta), r*cos(theta), 0]
    vector_field = [r * sp.sin(theta), r * sp.cos(theta), 0]
    print("\nVector field (contravariant components):")
    for i in range(3):
        print(f"V^{i} = {vector_field[i]}")
    
    # Compute the divergence
    div_v = divergence(vector_field, spherical_metric, coords)
    print(f"\nDivergence of vector field: ∇·V = {sp.simplify(div_v)}")
    
    # Compute the Laplacian of the scalar field
    lap_f = laplacian(scalar_field, spherical_metric, coords)
    print(f"\nLaplacian of scalar field: ∇²f = {sp.simplify(lap_f)}")
    
    # Compute Christoffel symbols
    christoffel = compute_christoffel(spherical_metric, coords)
    print("\nSample of non-zero Christoffel symbols:")
    shown = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if christoffel[i][j][k] != 0 and shown < 5:
                    print(f"Γ^{i}_{j}{k} = {sp.simplify(christoffel[i][j][k])}")
                    shown += 1

def numeric_demo():
    """Run a demonstration of numeric differential operators."""
    print("\n=== NUMERICAL CALCULATIONS WITH NUMPY ===")
    
    # Create a grid in spherical coordinates
    r_vals = np.linspace(0.1, 2.0, 20)
    theta_vals = np.linspace(0.1, np.pi-0.1, 20)
    phi_vals = np.linspace(0, 2*np.pi, 20)
    
    # Create meshgrid with correct indexing
    r_grid, theta_grid, phi_grid = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')
    grid = [r_vals, theta_vals, phi_vals]  # Use 1D arrays for the grid
    
    print(f"Grid dimensions: {r_grid.shape}")
    
    # Define a scalar field: f(r, theta, phi) = r^2 * sin(theta)^2 * cos(phi)
    scalar_field = r_grid**2 * np.sin(theta_grid)**2 * np.cos(phi_grid)
    
    # Create a constant diagonal metric tensor for spherical coordinates
    metric = np.zeros((3, 3))
    metric_inverse = np.zeros((3, 3))
    
    # Method 1: Position-dependent metric (full tensor)
    print("\nUsing position-dependent metric...")
    
    # Initialize position-dependent metric tensor
    metric_full = np.zeros((3, 3) + r_grid.shape)
    metric_inverse_full = np.zeros_like(metric_full)
    
    # Fill the metric tensor values
    for i in range(r_grid.shape[0]):
        for j in range(r_grid.shape[1]):
            for k in range(r_grid.shape[2]):
                r_val, theta_val = r_grid[i,j,k], theta_grid[i,j,k]
                
                # Create the metric at this point (diagonal in spherical coords)
                g_point = np.zeros((3, 3))
                g_point[0, 0] = 1.0  # g_rr = 1
                g_point[1, 1] = r_val**2  # g_θθ = r²
                g_point[2, 2] = r_val**2 * np.sin(theta_val)**2  # g_φφ = r²sin²θ
                
                metric_full[:, :, i, j, k] = g_point
                metric_inverse_full[:, :, i, j, k] = np.linalg.inv(g_point)
    
    # Compute the gradient with position-dependent metric
    gradient_field = evaluate_gradient(scalar_field, metric_inverse_full, grid)
    print(f"Gradient computed. Shape of each component: {gradient_field[0].shape}")
    
    # Create a vector field: V = [r*sin(theta), r*cos(theta), 0]
    vector_field = [
        r_grid * np.sin(theta_grid),
        r_grid * np.cos(theta_grid),
        np.zeros_like(r_grid)
    ]
    
    # Compute the divergence
    divergence_field = evaluate_divergence(vector_field, metric_full, grid)
    print(f"Divergence computed. Shape: {divergence_field.shape}")
    
    # Compute the Laplacian
    laplacian_field = evaluate_laplacian(scalar_field, metric_full, metric_inverse_full, grid)
    print(f"Laplacian computed. Shape: {laplacian_field.shape}")
    
    # Method 2: Using metric functions (alternative approach)
    print("\nUsing metric functions (alternative approach)...")
    
    # Define metric functions
    def g_rr(point):
        return 1.0
        
    def g_theta_theta(point):
        r = point[0]
        return r**2
        
    def g_phi_phi(point):
        r, theta = point[0], point[1]
        return r**2 * np.sin(theta)**2
    
    # Create a list of metric functions
    metric_funcs = [[g_rr, 0, 0], 
                    [0, g_theta_theta, 0], 
                    [0, 0, g_phi_phi]]
    
    # Fill in zero functions
    for i in range(3):
        for j in range(3):
            if metric_funcs[i][j] == 0:
                metric_funcs[i][j] = lambda point: 0.0
    
    # Visualize a 2D slice of the scalar field and its Laplacian
    visualize_fields(r_grid, theta_grid, scalar_field, laplacian_field)

def visualize_fields(r_grid, theta_grid, scalar_field, laplacian_field):
    """Visualize 2D slices of the scalar field and its Laplacian."""
    # Take a slice at phi = 0
    phi_slice = 0
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot scalar field
    im1 = axes[0].contourf(
        r_grid[:, :, phi_slice], theta_grid[:, :, phi_slice], 
        scalar_field[:, :, phi_slice], 20, cmap='viridis'
    )
    axes[0].set_title('Scalar Field f(r,θ,φ) at φ=0')
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('θ')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot Laplacian
    im2 = axes[1].contourf(
        r_grid[:, :, phi_slice], theta_grid[:, :, phi_slice], 
        laplacian_field[:, :, phi_slice], 20, cmap='plasma'
    )
    axes[1].set_title('Laplacian ∇²f at φ=0')
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('θ')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the demonstration."""
    print("iTensor Differential Operators Demo")
    print("===================================")
    
    symbolic_demo()
    numeric_demo()

if __name__ == "__main__":
    main() 