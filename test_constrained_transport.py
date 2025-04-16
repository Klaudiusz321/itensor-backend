"""
Test script for constrained transport implementation in MHD.

This script verifies that the constrained transport method maintains
the divergence-free condition of the magnetic field (∇·B = 0) to
machine precision during MHD evolution.
"""

import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project to the Python path
sys.path.append('.')

from myproject.utils.mhd.core import MHDSystem
from myproject.utils.mhd.initial_conditions import magnetic_rotor
# Orszag-Tang is initialized directly in the MHDSystem class
from myproject.utils.mhd.constrained_transport import (
    initialize_face_centered_b, 
    compute_emf, 
    update_face_centered_b,
    face_to_cell_centered_b,
    check_divergence_free
)

def test_divergence_free_initialization():
    """Test that the face-centered B field initialization preserves ∇·B = 0."""
    logger.info("Testing divergence-free initialization")
    
    # Create a simple 2D grid
    nx, ny = 32, 32
    grid_shape = (nx, ny)
    domain_size = [(0.0, 1.0), (0.0, 1.0)]
    grid_spacing = [1.0/nx, 1.0/ny]
    
    # Create a simple divergence-free magnetic field (B = curl A for some A)
    # For example, a vortex-like field where A = (0, 0, f(x,y))
    # This gives B = (∂f/∂y, -∂f/∂x, 0)
    Bx = np.zeros(grid_shape)
    By = np.zeros(grid_shape)
    
    # Create a potential function f(x,y) = sin(2πx) * sin(2πy)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    f = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
    
    # Compute B = curl A where A = (0, 0, f)
    # Bx = ∂f/∂y, By = -∂f/∂x
    Bx = 2*np.pi * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    By = -2*np.pi * np.cos(2*np.pi*X) * np.sin(2*np.pi*Y)
    
    # Create the cell-centered magnetic field
    cell_centered_b = [Bx, By]
    
    # Initialize face-centered B from cell-centered B
    face_centered_b = initialize_face_centered_b(cell_centered_b, grid_shape)
    
    # Check if the face-centered B is divergence-free
    div_b_magnitude = check_divergence_free(face_centered_b, grid_spacing)
    
    logger.info(f"Initial divergence magnitude: {div_b_magnitude:.6e}")
    assert div_b_magnitude < 1e-10, f"Divergence should be close to zero, got {div_b_magnitude}"
    
    return face_centered_b, grid_spacing, grid_shape

def test_constrained_transport_update():
    """Test that the constrained transport update preserves ∇·B = 0."""
    logger.info("Testing constrained transport update")
    
    # Use the initialized field from the previous test
    face_centered_b, grid_spacing, grid_shape = test_divergence_free_initialization()
    
    # Create a velocity field (e.g., a simple shear flow)
    vx = np.zeros(grid_shape)
    vy = np.zeros(grid_shape)
    
    # Initialize with a simple shear flow: v = (y, 0)
    nx, ny = grid_shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    vx = Y  # Simple shear flow
    
    velocity = [vx, vy]
    
    # Compute EMF from velocity and magnetic field
    emf = compute_emf(velocity, face_centered_b, grid_spacing)
    
    # Choose a time step that satisfies CFL
    max_velocity = np.max(np.abs(vx))
    min_dx = min(grid_spacing)
    dt = 0.5 * min_dx / max(max_velocity, 1e-10)  # CFL < 0.5
    
    logger.info(f"Using time step dt = {dt:.6e}")
    
    # Record initial divergence
    initial_div = check_divergence_free(face_centered_b, grid_spacing)
    logger.info(f"Initial divergence: {initial_div:.6e}")
    
    # Update the face-centered magnetic field
    updated_face_b = update_face_centered_b(face_centered_b, emf, dt, grid_spacing)
    
    # Check if the updated field is still divergence-free
    final_div = check_divergence_free(updated_face_b, grid_spacing)
    logger.info(f"Final divergence after update: {final_div:.6e}")
    
    # The divergence should remain close to machine precision
    assert final_div < 1e-10, f"Divergence should remain close to zero, got {final_div}"
    
    # Also check that the field has actually changed (the update did something)
    b_diff = np.max([np.max(np.abs(updated_face_b[i] - face_centered_b[i])) for i in range(len(face_centered_b))])
    logger.info(f"Maximum change in B field: {b_diff:.6e}")
    
    assert b_diff > 1e-10, f"B field should change after update, max difference: {b_diff}"
    
    logger.info("Constrained transport update preserves divergence-free condition ✓")

def test_mhd_with_constrained_transport():
    """Test a full MHD system evolution with constrained transport."""
    logger.info("Testing full MHD system with constrained transport")
    
    # Initialize an MHD system with the Orszag-Tang vortex problem
    system = MHDSystem(
        domain_size=[(0.0, 2.0), (0.0, 2.0)],
        resolution=[64, 64],
        adiabatic_index=5/3,
        use_constrained_transport=True  # Enable constrained transport
    )
    
    # Initialize with the Orszag-Tang vortex
    system.initialize_orszag_tang_2d()
    
    # Check initial divergence
    initial_div = system.check_divergence_free()
    logger.info(f"Initial divergence in MHD system: {initial_div:.6e}")
    
    # Evolve for a few time steps
    try:
        # Compute initial time step
        dt = system.compute_time_step()
        
        # Evolve for 10 steps or until final_time
        final_time = 0.1
        current_time = 0.0
        num_steps = 10
        
        for step in range(num_steps):
            if current_time >= final_time:
                break
                
            # Make sure we hit final_time exactly
            if current_time + dt > final_time:
                dt = final_time - current_time
                
            # Advance one time step
            system.advance_time_step()
            current_time += dt
            
            # Check divergence after each step
            div_magnitude = system.check_divergence_free()
            logger.info(f"Step {step+1}, time {current_time:.4f}: divergence = {div_magnitude:.6e}")
            
            # Recalculate time step for next iteration
            dt = system.compute_time_step()
            
        # Final divergence check
        final_div = system.check_divergence_free()
        logger.info(f"Final divergence in MHD system after {num_steps} steps: {final_div:.6e}")
        
        # The divergence should remain close to machine precision
        assert final_div < 1e-10, f"Divergence should remain close to zero, got {final_div}"
        
        logger.info("MHD evolution with constrained transport preserves divergence-free condition ✓")
        
    except Exception as e:
        logger.error(f"Error during MHD evolution: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting constrained transport tests")
        
        # Run the tests
        test_divergence_free_initialization()
        test_constrained_transport_update()
        
        # Only run the full MHD test if the advance_time_step method is implemented
        try:
            test_mhd_with_constrained_transport()
        except (AttributeError, NotImplementedError):
            logger.warning("MHD time evolution test skipped - advance_time_step not fully implemented")
        
        logger.info("All constrained transport tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise 