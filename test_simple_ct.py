"""
Simplified test for constrained transport (CT) method in MHD.

This script demonstrates the basic principle of the constrained transport method
for maintaining the divergence-free condition of the magnetic field (∇·B = 0).
"""

import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project to the Python path
sys.path.append('.')

try:
    from myproject.utils.differential_operators.numeric import create_grid, evaluate_divergence
    from myproject.utils.mhd.core import MHDSystem
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def create_face_centered_b(grid, method='circular'):
    """
    Create a face-centered magnetic field directly (without Numba).
    
    Args:
        grid: List of coordinate arrays [x, y]
        method: 'circular' for B = (-y, x) or 'potential' for B = curl(A)
        
    Returns:
        List of face-centered magnetic field components [Bx, By]
    """
    nx, ny = len(grid[0]), len(grid[1])
    
    # Create face-centered grids
    # For Bx: (nx+1) x ny grid, staggered in x
    # For By: nx x (ny+1) grid, staggered in y
    
    # Face grid coordinates
    x_faces = np.linspace(grid[0][0] - 0.5*(grid[0][1]-grid[0][0]), 
                         grid[0][-1] + 0.5*(grid[0][1]-grid[0][0]), 
                         nx+1)
    y_faces = np.linspace(grid[1][0] - 0.5*(grid[1][1]-grid[1][0]), 
                         grid[1][-1] + 0.5*(grid[1][1]-grid[1][0]), 
                         ny+1)
    
    # Create mesh grids for face centers
    X_bx, Y_bx = np.meshgrid(x_faces, grid[1], indexing='ij')
    X_by, Y_by = np.meshgrid(grid[0], y_faces, indexing='ij')
    
    # Initialize face-centered B fields
    face_bx = np.zeros((nx+1, ny))
    face_by = np.zeros((nx, ny+1))
    
    if method == 'circular':
        # Simple circular field: B = (-y, x)
        face_bx = -Y_bx
        face_by = X_by
    elif method == 'potential':
        # Magnetic field from a vector potential A = (0, 0, sin(2πx)sin(2πy))
        # This gives B = (∂A_z/∂y, -∂A_z/∂x, 0)
        for i in range(nx+1):
            for j in range(ny):
                # Bx at (i-1/2, j)
                face_bx[i, j] = 2 * np.pi * np.sin(2*np.pi*X_bx[i, j]) * np.cos(2*np.pi*Y_bx[i, j])
        
        for i in range(nx):
            for j in range(ny+1):
                # By at (i, j-1/2)
                face_by[i, j] = -2 * np.pi * np.cos(2*np.pi*X_by[i, j]) * np.sin(2*np.pi*Y_by[i, j])
    
    return [face_bx, face_by]

def compute_face_centered_divergence(face_b, dx, dy):
    """
    Compute the divergence of a face-centered magnetic field.
    
    In a staggered grid, divergence is naturally defined at cell centers:
    div(B) = [Bx(i+1/2,j) - Bx(i-1/2,j)]/dx + [By(i,j+1/2) - By(i,j-1/2)]/dy
    
    Args:
        face_b: List of face-centered magnetic field components [Bx, By]
        dx, dy: Grid spacing in x and y directions
        
    Returns:
        Array of divergence values at cell centers
    """
    face_bx, face_by = face_b
    nx, ny = face_by.shape[0], face_bx.shape[1]
    
    div_b = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            # Divergence = (Bx(i+1/2,j) - Bx(i-1/2,j))/dx + (By(i,j+1/2) - By(i,j-1/2))/dy
            div_b[i, j] = (face_bx[i+1, j] - face_bx[i, j]) / dx + \
                         (face_by[i, j+1] - face_by[i, j]) / dy
    
    return div_b

def compute_emf_2d(vx, vy, face_bx, face_by, grid):
    """
    Compute the z-component of the electromotive force E = v × B for 2D.
    
    Args:
        vx, vy: Velocity components at cell centers
        face_bx, face_by: Face-centered magnetic field components
        grid: List of coordinate arrays [x, y]
        
    Returns:
        EMF z-component at cell corners (Ex and Ey are zero in 2D)
    """
    nx, ny = len(grid[0]), len(grid[1])
    
    # EMF is at cell corners in 2D
    emf_z = np.zeros((nx-1, ny-1))
    
    for i in range(nx-1):
        for j in range(ny-1):
            # Average velocity to corner (i+1/2, j+1/2)
            # The velocity is defined at cell centers
            vx_corner = 0.25 * (vx[i, j] + vx[i+1, j] + vx[i, j+1] + vx[i+1, j+1])
            vy_corner = 0.25 * (vy[i, j] + vy[i+1, j] + vy[i, j+1] + vy[i+1, j+1])
            
            # Average B to corner (i+1/2, j+1/2)
            # Bx is defined at (i+1/2, j) faces
            # By is defined at (i, j+1/2) faces
            # To get Bx at corner (i+1/2, j+1/2), average Bx(i+1/2, j) and Bx(i+1/2, j+1)
            # To get By at corner (i+1/2, j+1/2), average By(i, j+1/2) and By(i+1, j+1/2)
            bx_corner = 0.5 * (face_bx[i+1, j] + face_bx[i+1, j+1])
            by_corner = 0.5 * (face_by[i, j+1] + face_by[i+1, j+1])
            
            # Ez = vx*By - vy*Bx at corner
            emf_z[i, j] = vx_corner * by_corner - vy_corner * bx_corner
    
    return emf_z

def update_face_centered_b_2d(face_bx, face_by, emf_z, dt, dx, dy):
    """
    Update face-centered magnetic field using constrained transport in 2D.
    
    Args:
        face_bx, face_by: Face-centered magnetic field components
        emf_z: EMF z-component at cell corners
        dt: Time step
        dx, dy: Grid spacing
        
    Returns:
        Updated face-centered magnetic field components
    """
    nx, ny = face_by.shape[0], face_bx.shape[1]
    
    # Create new arrays for updated fields
    new_face_bx = face_bx.copy()
    new_face_by = face_by.copy()
    
    # Update Bx faces using Faraday's law: ∂Bx/∂t = -∂Ez/∂y
    # For each x-face at (i+1/2, j), we compute ∂Ez/∂y using adjacent EMF values
    for i in range(1, nx):  # Interior faces only (i=1...nx-1)
        for j in range(1, ny-1):  # Interior faces only (j=1...ny-2)
            # At face (i+1/2, j), compute ∂Ez/∂y using EMF at corners
            dez_dy = (emf_z[i-1, j] - emf_z[i-1, j-1]) / dy
            new_face_bx[i, j] = face_bx[i, j] - dt * dez_dy
    
    # Update By faces using Faraday's law: ∂By/∂t = ∂Ez/∂x
    # For each y-face at (i, j+1/2), we compute ∂Ez/∂x using adjacent EMF values
    for i in range(1, nx-1):  # Interior faces only (i=1...nx-2)
        for j in range(1, ny):  # Interior faces only (j=1...ny-1)
            # At face (i, j+1/2), compute ∂Ez/∂x using EMF at corners
            dez_dx = (emf_z[i, j-1] - emf_z[i-1, j-1]) / dx
            new_face_by[i, j] = face_by[i, j] + dt * dez_dx
    
    return new_face_bx, new_face_by

def face_to_cell_centered(face_bx, face_by):
    """
    Convert face-centered magnetic field to cell-centered values.
    
    Args:
        face_bx, face_by: Face-centered magnetic field components
        
    Returns:
        Cell-centered magnetic field components
    """
    nx, ny = face_by.shape[0], face_bx.shape[1]
    
    cell_bx = np.zeros((nx, ny))
    cell_by = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            # Average Bx from adjacent faces
            cell_bx[i, j] = 0.5 * (face_bx[i, j] + face_bx[i+1, j])
            # Average By from adjacent faces
            cell_by[i, j] = 0.5 * (face_by[i, j] + face_by[i, j+1])
    
    return cell_bx, cell_by

def plot_field_and_divergence(cell_bx, cell_by, div_b, grid, title):
    """Plot the magnetic field and its divergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(grid[0], grid[1], indexing='ij')
    
    # Plot vector field
    skip = 2  # Skip some points for clarity
    ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               cell_bx[::skip, ::skip], cell_by[::skip, ::skip], 
               scale=20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Magnetic Field')
    ax1.set_aspect('equal')
    
    # Use a symmetric and appropriate color scale for divergence
    max_div = np.max(np.abs(div_b))
    div_scale = max(max_div, 1e-10)  # Avoid zero scale
    
    # Log the scale we're using
    logger.info(f"Divergence plot scale: ±{div_scale:.6e}")
    
    # Plot divergence with symmetric colormap
    im = ax2.imshow(div_b.T, origin='lower', 
                   extent=[grid[0][0], grid[0][-1], grid[1][0], grid[1][-1]],
                   cmap='RdBu_r', vmin=-div_scale, vmax=div_scale)
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

def test_constrained_transport():
    """Test the constrained transport implementation."""
    logger.info("Testing constrained transport")
    
    # Create a 2D grid
    nx, ny = 32, 32
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)
    
    # Create grid points with correct parameter order (coords_ranges, dimensions)
    coords_ranges = {'x': {'min': x_range[0], 'max': x_range[1]},
                     'y': {'min': y_range[0], 'max': y_range[1]}}
    dimensions = [nx, ny]
    
    grid, spacing = create_grid(coords_ranges, dimensions)
    dx, dy = spacing['x'], spacing['y']
    
    # Create face-centered magnetic field
    face_b = create_face_centered_b(grid, method='potential')
    face_bx, face_by = face_b
    
    # Convert to cell-centered for visualization and divergence check
    cell_bx, cell_by = face_to_cell_centered(face_bx, face_by)
    
    # Compute divergence using our staggered grid method
    div_b = compute_face_centered_divergence(face_b, dx, dy)
    
    # Also compute divergence using the utility function for comparison
    cell_b = [cell_bx, cell_by]
    metric_identity = np.eye(2)
    div_b_numeric = evaluate_divergence(cell_b, metric_identity, grid)
    
    # Report divergence
    max_div = np.max(np.abs(div_b))
    mean_div = np.mean(np.abs(div_b))
    logger.info(f"Max |∇·B| (staggered): {max_div:.6e}")
    logger.info(f"Mean |∇·B| (staggered): {mean_div:.6e}")
    
    max_div_numeric = np.max(np.abs(div_b_numeric))
    mean_div_numeric = np.mean(np.abs(div_b_numeric))
    logger.info(f"Max |∇·B| (numeric): {max_div_numeric:.6e}")
    logger.info(f"Mean |∇·B| (numeric): {mean_div_numeric:.6e}")
    
    # Plot initial field and divergence
    plot_field_and_divergence(cell_bx, cell_by, div_b, grid, "Initial Magnetic Field")
    
    # Create a simple velocity field (e.g., rotating flow)
    # Create 2D meshgrid for the velocity field
    X, Y = np.meshgrid(grid[0], grid[1], indexing='ij')
    vx = -Y  # -y component for rotation
    vy = X   # x component for rotation
    
    # Compute EMF
    emf_z = compute_emf_2d(vx, vy, face_bx, face_by, grid)
    
    # Choose a time step
    v_max = np.max([np.max(np.abs(vx)), np.max(np.abs(vy))])
    dt = 0.1 * min(dx, dy) / (v_max + 1e-10)  # CFL condition
    logger.info(f"Using time step dt = {dt}")
    
    # Update the magnetic field using constrained transport
    new_face_bx, new_face_by = update_face_centered_b_2d(face_bx, face_by, emf_z, dt, dx, dy)
    
    # Convert to cell-centered for visualization and divergence check
    new_cell_bx, new_cell_by = face_to_cell_centered(new_face_bx, new_face_by)
    
    # Compute divergence of the updated field
    new_face_b = [new_face_bx, new_face_by]
    new_div_b = compute_face_centered_divergence(new_face_b, dx, dy)
    
    # Report divergence of updated field
    new_max_div = np.max(np.abs(new_div_b))
    new_mean_div = np.mean(np.abs(new_div_b))
    logger.info(f"Max |∇·B| after update: {new_max_div:.6e}")
    logger.info(f"Mean |∇·B| after update: {new_mean_div:.6e}")
    
    # Plot updated field and divergence
    plot_field_and_divergence(new_cell_bx, new_cell_by, new_div_b, grid, "Updated Magnetic Field")
    
    # Check if field changed but divergence remained small
    b_diff = np.max([
        np.max(np.abs(new_cell_bx - cell_bx)),
        np.max(np.abs(new_cell_by - cell_by))
    ])
    logger.info(f"Maximum change in B: {b_diff:.6e}")
    
    if new_max_div < 1e-10 and b_diff > 1e-10:
        logger.info("✓ Constrained transport successfully maintained ∇·B = 0 while evolving the field")
    else:
        if new_max_div >= 1e-10:
            logger.warning(f"⚠ Updated field has non-zero divergence: {new_max_div:.6e}")
        if b_diff <= 1e-10:
            logger.warning("⚠ Magnetic field did not change significantly during update")

if __name__ == "__main__":
    try:
        logger.info("Starting simple constrained transport test")
        test_constrained_transport()
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 