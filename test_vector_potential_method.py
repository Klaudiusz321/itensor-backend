"""
Test for divergence-free magnetic field evolution using vector potential method.

This script demonstrates how to maintain the divergence-free condition (∇·B = 0)
in MHD simulations by evolving the magnetic vector potential A instead of B directly,
since B = ∇×A is always divergence-free by construction.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project to the Python path
sys.path.append('.')

try:
    from myproject.utils.differential_operators.numeric import create_grid, evaluate_divergence, evaluate_curl
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def vector_potential_magnetic_field(grid, t=0):
    """
    Create a time-dependent magnetic field from a vector potential.
    
    For a 2D problem in the x-y plane:
    A = (0, 0, Az(x,y,t)) where Az = sin(2πx)sin(2πy)cos(ωt)
    This gives B = ∇×A = (∂Az/∂y, -∂Az/∂x, 0)
    
    Args:
        grid: List of coordinate arrays [x, y]
        t: Time
        
    Returns:
        Tuple of (A, B) where:
        - A is the vector potential (only z-component in 2D)
        - B is the magnetic field components [Bx, By]
    """
    # Create meshgrid
    X, Y = np.meshgrid(grid[0], grid[1], indexing='ij')
    
    # Set frequency for time-dependent field
    omega = 2.0 * np.pi
    
    # Calculate vector potential Az
    Az = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y) * np.cos(omega*t)
    
    # Calculate B = ∇×A
    Bx = 2*np.pi * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y) * np.cos(omega*t)
    By = -2*np.pi * np.cos(2*np.pi*X) * np.sin(2*np.pi*Y) * np.cos(omega*t)
    
    return Az, [Bx, By]

def evolve_vector_potential(Az, grid, v, dt):
    """
    Evolve the vector potential according to ∂A/∂t = v×B - ∇φ
    where φ is a gauge potential (we use φ=0 gauge for simplicity).
    
    In 2D with A = (0, 0, Az), this reduces to:
    ∂Az/∂t = vx*By - vy*Bx
    
    Args:
        Az: z-component of the vector potential
        grid: List of coordinate arrays [x, y]
        v: Velocity field components [vx, vy]
        dt: Time step
        
    Returns:
        Updated vector potential
    """
    # Calculate B = ∇×A
    B = calculate_B_from_A(Az, grid)
    
    # Create 2D meshgrid
    X, Y = np.meshgrid(grid[0], grid[1], indexing='ij')
    
    # EMF = v×B (in 2D, this is just the z-component)
    vx, vy = v
    Bx, By = B
    
    # EMF = vx*By - vy*Bx (z-component of v×B)
    emf = vx * By - vy * Bx
    
    # Update Az
    Az_new = Az + dt * emf
    
    return Az_new

def calculate_B_from_A(Az, grid):
    """
    Calculate magnetic field B = ∇×A from vector potential.
    
    For a 2D problem with A = (0, 0, Az):
    Bx = ∂Az/∂y
    By = -∂Az/∂x
    
    Args:
        Az: z-component of the vector potential
        grid: List of coordinate arrays [x, y]
        
    Returns:
        Magnetic field components [Bx, By]
    """
    dx = grid[0][1] - grid[0][0]
    dy = grid[1][1] - grid[1][0]
    
    # Use central differences for derivatives
    Bx = np.zeros_like(Az)
    By = np.zeros_like(Az)
    
    # Interior points
    Bx[1:-1, 1:-1] = (Az[1:-1, 2:] - Az[1:-1, :-2]) / (2*dy)
    By[1:-1, 1:-1] = -(Az[2:, 1:-1] - Az[:-2, 1:-1]) / (2*dx)
    
    # Boundary points (use forward/backward differences)
    # Left boundary
    By[0, 1:-1] = -(Az[1, 1:-1] - Az[0, 1:-1]) / dx
    # Right boundary
    By[-1, 1:-1] = -(Az[-1, 1:-1] - Az[-2, 1:-1]) / dx
    # Bottom boundary
    Bx[1:-1, 0] = (Az[1:-1, 1] - Az[1:-1, 0]) / dy
    # Top boundary
    Bx[1:-1, -1] = (Az[1:-1, -1] - Az[1:-1, -2]) / dy
    
    # Corners (use one-sided differences)
    # Bottom left
    Bx[0, 0] = (Az[0, 1] - Az[0, 0]) / dy
    By[0, 0] = -(Az[1, 0] - Az[0, 0]) / dx
    # Bottom right
    Bx[-1, 0] = (Az[-1, 1] - Az[-1, 0]) / dy
    By[-1, 0] = -(Az[-1, 0] - Az[-2, 0]) / dx
    # Top left
    Bx[0, -1] = (Az[0, -1] - Az[0, -2]) / dy
    By[0, -1] = -(Az[1, -1] - Az[0, -1]) / dx
    # Top right
    Bx[-1, -1] = (Az[-1, -1] - Az[-1, -2]) / dy
    By[-1, -1] = -(Az[-1, -1] - Az[-2, -1]) / dx
    
    return [Bx, By]

def plot_field_and_divergence(B, div_B, grid, title):
    """
    Plot the magnetic field and its divergence.
    
    Args:
        B: Magnetic field components [Bx, By]
        div_B: Divergence of the magnetic field
        grid: List of coordinate arrays [x, y]
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(grid[0], grid[1], indexing='ij')
    
    # Plot vector field
    Bx, By = B
    skip = 2  # Skip some points for clarity
    ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               Bx[::skip, ::skip], By[::skip, ::skip],
               scale=20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Magnetic Field B')
    ax1.set_aspect('equal')
    
    # Plot divergence
    max_div = np.max(np.abs(div_B))
    vmax = max(max_div, 1e-10)  # Avoid zero scale
    
    im = ax2.imshow(div_B.T, origin='lower', 
                   extent=[grid[0][0], grid[0][-1], grid[1][0], grid[1][-1]],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Divergence (∇·B), max = {max_div:.2e}')
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    logger.info(f"Saved plot: {title.replace(' ', '_').lower()}.png")

def test_vector_potential_method():
    """
    Test the vector potential method for maintaining ∇·B = 0.
    """
    logger.info("Testing vector potential method")
    
    # Create a 2D grid
    nx, ny = 64, 64
    x_range = (0.0, 1.0)
    y_range = (0.0, 1.0)
    
    # Set up grid
    coords_ranges = {'x': {'min': x_range[0], 'max': x_range[1]},
                     'y': {'min': y_range[0], 'max': y_range[1]}}
    dimensions = [nx, ny]
    
    grid, spacing = create_grid(coords_ranges, dimensions)
    dx, dy = spacing['x'], spacing['y']
    
    # Time parameters
    t = 0.0
    dt = 0.1 * min(dx, dy)  # CFL condition
    final_time = 0.5
    
    # Initialize vector potential and magnetic field
    Az, B = vector_potential_magnetic_field(grid, t)
    
    # Define a simple velocity field (rotational)
    X, Y = np.meshgrid(grid[0], grid[1], indexing='ij')
    center_x, center_y = 0.5, 0.5
    vx = -(Y - center_y)  # clockwise rotation
    vy = (X - center_x)
    v = [vx, vy]
    
    # Compute initial divergence
    metric_identity = np.eye(2)
    div_B = evaluate_divergence(B, metric_identity, grid)
    
    max_div = np.max(np.abs(div_B))
    logger.info(f"Initial maximum |∇·B| = {max_div:.6e}")
    
    # Plot initial configuration
    plot_field_and_divergence(B, div_B, grid, f"Initial Field (t={t:.2f})")
    
    # Evolve through time steps
    num_steps = int(final_time / dt)
    
    for step in range(num_steps):
        # Update time
        t += dt
        
        # Evolve vector potential
        Az = evolve_vector_potential(Az, grid, v, dt)
        
        # Calculate B field from updated A
        B = calculate_B_from_A(Az, grid)
        
        # Calculate divergence
        div_B = evaluate_divergence(B, metric_identity, grid)
        max_div = np.max(np.abs(div_B))
        
        logger.info(f"Step {step+1}, t = {t:.2f}, max |∇·B| = {max_div:.6e}")
    
    # Plot final configuration
    plot_field_and_divergence(B, div_B, grid, f"Final Field (t={t:.2f})")
    
    # Compare with direct calculation at final time
    Az_exact, B_exact = vector_potential_magnetic_field(grid, t)
    
    # Calculate error
    Bx_error = np.max(np.abs(B[0] - B_exact[0]))
    By_error = np.max(np.abs(B[1] - B_exact[1]))
    
    logger.info(f"Maximum error in Bx: {Bx_error:.6e}")
    logger.info(f"Maximum error in By: {By_error:.6e}")
    
    # Final divergence check
    if max_div < 1e-10:
        logger.info("✓ Vector potential method successfully maintained ∇·B = 0")
    else:
        logger.warning(f"⚠ Divergence not zero: max |∇·B| = {max_div:.6e}")
    
    return B, div_B, grid, t

if __name__ == "__main__":
    try:
        logger.info("Starting vector potential method test")
        B, div_B, grid, t = test_vector_potential_method()
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 