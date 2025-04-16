#!/usr/bin/env python
"""
Test script to verify gradient calculation with various array shapes.
"""

import sys
import os
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from myproject.utils.differential_operators import (
    evaluate_gradient, create_grid, compute_partial_derivative
)

def setup_logging():
    """Set up logging to see debug messages."""
    import logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def test_1d_gradient():
    """Test gradient calculation on a 1D field."""
    logger = setup_logging()
    logger.info("Testing 1D gradient calculation")
    
    # Create a 1D grid
    coords_ranges = {'x': {'min': 0.0, 'max': 2.0}}
    dimensions = [10]
    grid, spacing = create_grid(coords_ranges, dimensions)
    
    # Create a 1D field: f(x) = x²
    x = grid[0]
    scalar_field = x**2
    
    # Create a simple metric (Euclidean)
    metric_inverse = np.eye(1)
    
    logger.info(f"Grid shape: {x.shape}, Field shape: {scalar_field.shape}")
    
    # Compute the gradient
    logger.info("Computing gradient of 1D field")
    gradient = evaluate_gradient(scalar_field, metric_inverse, grid)
    
    # The gradient should be df/dx = 2x
    expected = 2 * x
    
    logger.info(f"Gradient shape: {gradient[0].shape}")
    logger.info(f"First few values - Computed: {gradient[0][:3]}, Expected: {expected[:3]}")
    
    # Check for close match
    if np.allclose(gradient[0], expected, rtol=1e-2):
        logger.info("✓ 1D gradient test PASSED")
    else:
        logger.error("✗ 1D gradient test FAILED")
        logger.error(f"Max difference: {np.max(np.abs(gradient[0] - expected))}")

def test_2d_gradient():
    """Test gradient calculation on a 2D field."""
    logger = setup_logging()
    logger.info("Testing 2D gradient calculation")
    
    # Create a 2D grid
    coords_ranges = {
        'x': {'min': 0.0, 'max': 2.0},
        'y': {'min': 0.0, 'max': 2.0}
    }
    dimensions = [10, 10]
    grid, spacing = create_grid(coords_ranges, dimensions)
    
    # Create a meshgrid for evaluation
    x_grid, y_grid = np.meshgrid(grid[0], grid[1], indexing='ij')
    
    # Create a 2D field: f(x,y) = x² + y²
    scalar_field = x_grid**2 + y_grid**2
    
    # Create a simple metric (Euclidean)
    metric_inverse = np.eye(2)
    
    logger.info(f"Grid shape: {x_grid.shape}, Field shape: {scalar_field.shape}")
    
    # Compute the gradient
    logger.info("Computing gradient of 2D field")
    gradient = evaluate_gradient(scalar_field, metric_inverse, grid)
    
    # The gradient should be ∇f = (2x, 2y)
    expected_dx = 2 * x_grid
    expected_dy = 2 * y_grid
    
    logger.info(f"Gradient shapes: {gradient[0].shape}, {gradient[1].shape}")
    logger.info(f"Sample at (0,0) - Computed: ({gradient[0][0,0]}, {gradient[1][0,0]}), " 
               f"Expected: ({expected_dx[0,0]}, {expected_dy[0,0]})")
    
    # Check for close match
    if (np.allclose(gradient[0], expected_dx, rtol=1e-2) and 
        np.allclose(gradient[1], expected_dy, rtol=1e-2)):
        logger.info("✓ 2D gradient test PASSED")
    else:
        logger.error("✗ 2D gradient test FAILED")
        logger.error(f"Max difference (x): {np.max(np.abs(gradient[0] - expected_dx))}")
        logger.error(f"Max difference (y): {np.max(np.abs(gradient[1] - expected_dy))}")

def test_mixed_dimensions():
    """Test gradient calculation with mismatched dimensions."""
    logger = setup_logging()
    logger.info("Testing gradient with mismatched dimensions")
    
    # Create a 2D grid
    coords_ranges = {
        'x': {'min': 0.0, 'max': 2.0},
        'y': {'min': 0.0, 'max': 2.0}
    }
    dimensions = [10, 10]
    grid, spacing = create_grid(coords_ranges, dimensions)
    
    # Create a field that depends only on x (effectively 1D but in a 2D space)
    # This mimics cases where a scalar field might be simple in some dimensions
    x = grid[0]
    x_grid, _ = np.meshgrid(x, grid[1], indexing='ij')
    scalar_field_2d = x_grid**2  # Field is f(x,y) = x²
    
    # Also create a truly 1D field
    scalar_field_1d = x**2  # Just x²
    
    # Create a simple metric (Euclidean)
    metric_inverse = np.eye(2)
    
    logger.info(f"2D Field shape: {scalar_field_2d.shape}, 1D Field shape: {scalar_field_1d.shape}")
    
    # First, compute the gradient of the properly shaped 2D field
    logger.info("Computing gradient of x-dependent 2D field")
    gradient_2d = evaluate_gradient(scalar_field_2d, metric_inverse, grid)
    
    # Now try with the 1D field (should trigger reshaping)
    logger.info("Computing gradient of 1D field with 2D grid (should trigger reshaping)")
    try:
        gradient_1d = evaluate_gradient(scalar_field_1d, metric_inverse, grid)
        
        logger.info(f"2D Field gradient shapes: {gradient_2d[0].shape}, {gradient_2d[1].shape}")
        logger.info(f"1D Field gradient shapes: {gradient_1d[0].shape}, {gradient_1d[1].shape}")
        
        # Get some sample values
        logger.info(f"Sample at x[0]: 2D gradient x-component: {gradient_2d[0][0,0]}")
        
        # The expected gradient for x² in the x-direction is 2x
        expected_dx = 2 * x[0]
        
        # For the 1D case, the first value might be accessible differently depending on reshaping
        if gradient_1d[0].ndim == 1:
            first_val = gradient_1d[0][0]
        else:
            first_val = gradient_1d[0][0,0]
            
        logger.info(f"Sample at x[0]: 1D gradient x-component: {first_val}")
        logger.info(f"Expected: {expected_dx}")
        
        if (np.isclose(gradient_2d[0][0,0], expected_dx, rtol=1e-2) and 
            np.isclose(first_val, expected_dx, rtol=1e-2)):
            logger.info("✓ Mixed dimensions test PASSED")
        else:
            logger.error("✗ Mixed dimensions test FAILED")
            logger.error(f"Expected {expected_dx}, got {gradient_2d[0][0,0]} and {first_val}")
    
    except Exception as e:
        import traceback
        logger.error(f"Error computing gradient with mismatched dimensions: {e}")
        logger.error(traceback.format_exc())
        logger.error("✗ Mixed dimensions test FAILED")

if __name__ == "__main__":
    print("Testing gradient calculations with different array dimensions")
    test_1d_gradient()
    print("\n" + "-"*50 + "\n")
    test_2d_gradient()
    print("\n" + "-"*50 + "\n")
    test_mixed_dimensions() 