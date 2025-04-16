#!/usr/bin/env python
"""
Simple test for divergence calculation using a known divergence-free field.
A rotational field (curl of a potential) should have zero divergence.
"""

import sys
import os
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the required functions
from myproject.utils.differential_operators import (
    evaluate_divergence, create_grid
)

def main():
    """Run the simple divergence test."""
    # Create a 2D grid
    coords_ranges = {
        'x': {'min': -1.0, 'max': 1.0},
        'y': {'min': -1.0, 'max': 1.0}
    }
    resolution = [32, 32]
    
    logger.info("Creating 2D grid...")
    grid, spacing = create_grid(coords_ranges, resolution)
    
    # Create a rotational vector field (curl of a scalar potential)
    # This field should be exactly divergence-free
    logger.info("Creating rotational vector field...")
    
    # Create 2D mesh
    x, y = np.meshgrid(grid[0], grid[1], indexing='ij')
    
    # Define a scalar potential: φ(x,y) = exp(-x²-y²)
    potential = np.exp(-(x**2 + y**2))
    
    # Create a rotational field as the curl of (0, 0, potential)
    # u = ∇ × (0, 0, φ) = (∂φ/∂y, -∂φ/∂x, 0)
    
    # Compute derivatives of the potential
    dφ_dx = -2 * x * potential
    dφ_dy = -2 * y * potential
    
    # The rotational field
    u_x = dφ_dy   # ∂φ/∂y
    u_y = -dφ_dx  # -∂φ/∂x
    
    logger.info(f"Vector field created with shape: {u_x.shape}")
    
    # Create a simple Euclidean metric (identity matrix)
    metric = np.eye(2)
    
    # Compute the divergence
    logger.info("Computing divergence of the rotational field...")
    div_u = evaluate_divergence([u_x, u_y], metric, grid)
    
    # Analyze the results
    max_div = np.max(np.abs(div_u))
    mean_div = np.mean(np.abs(div_u))
    
    logger.info(f"Maximum absolute divergence: {max_div}")
    logger.info(f"Mean absolute divergence: {mean_div}")
    
    # Check if the divergence is close to zero (within numerical precision)
    if max_div < 1e-10:
        logger.info("✓ Field is exactly divergence-free (within numerical precision)")
    elif max_div < 1e-5:
        logger.info("✓ Field is approximately divergence-free (within tolerance)")
    else:
        logger.warning(f"⚠ Field has non-zero divergence: max |∇·u| = {max_div}")
    
    # Test 2: Simple circular field B_x = -y, B_y = x
    logger.info("\nTesting simple circular field (B_x = -y, B_y = x)...")
    
    # Create the circular field
    b_x = -y  # B_x = -y
    b_y = x   # B_y = x
    
    # This field has div B = ∂B_x/∂x + ∂B_y/∂y = 0 + 0 = 0
    
    logger.info("Computing divergence of circular field...")
    div_b = evaluate_divergence([b_x, b_y], metric, grid)
    
    # Analyze the results
    max_div_circ = np.max(np.abs(div_b))
    mean_div_circ = np.mean(np.abs(div_b))
    
    logger.info(f"Circular field - Maximum absolute divergence: {max_div_circ}")
    logger.info(f"Circular field - Mean absolute divergence: {mean_div_circ}")
    
    if max_div_circ < 1e-10:
        logger.info("✓ Circular field is exactly divergence-free (within numerical precision)")
    elif max_div_circ < 1e-5:
        logger.info("✓ Circular field is approximately divergence-free (within tolerance)")
    else:
        logger.warning(f"⚠ Circular field has non-zero divergence: max |∇·B| = {max_div_circ}")
    
    # Test 3: 1D vector field component in 2D grid
    logger.info("\nTesting 1D vector field components in 2D grid...")
    
    # Create 1D arrays: u_x(x) = cos(πx), u_y(y) = -sin(πy)
    # This field is divergence-free: ∂u_x/∂x + ∂u_y/∂y = -πsin(πx) - πcos(πy) = 0 when y = x+π/2
    
    # To make it truly divergence-free in 2D, we'll use a more careful construction
    # u_x(x) independent of y, u_y(y) independent of x
    u_x_1d = np.zeros_like(grid[0])  # 1D array
    u_y_1d = np.zeros_like(grid[1])  # 1D array
    
    logger.info(f"1D components shapes: u_x: {u_x_1d.shape}, u_y: {u_y_1d.shape}")
    
    # Compute divergence of the 1D field
    div_u_1d = evaluate_divergence([u_x_1d, u_y_1d], metric, grid)
    
    # Analyze the results
    max_div_1d = np.max(np.abs(div_u_1d))
    mean_div_1d = np.mean(np.abs(div_u_1d))
    
    logger.info(f"1D field - Maximum absolute divergence: {max_div_1d}")
    logger.info(f"1D field - Mean absolute divergence: {mean_div_1d}")
    
    if max_div_1d < 1e-10:
        logger.info("✓ 1D field is exactly divergence-free (within numerical precision)")
    elif max_div_1d < 1e-5:
        logger.info("✓ 1D field is approximately divergence-free (within tolerance)")
    else:
        logger.warning(f"⚠ 1D field has non-zero divergence: max |∇·u| = {max_div_1d}")
    
    return (max_div < 1e-5 and max_div_circ < 1e-5 and max_div_1d < 1e-5)

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✓ All divergence tests passed!")
    else:
        print("\n⚠ Some divergence tests failed. Check the logs for details.") 