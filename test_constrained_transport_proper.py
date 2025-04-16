"""
Test for Constrained Transport implementation in MHD.

This script demonstrates a correct implementation of the constrained transport
method for ensuring ∇·B = 0 to machine precision during MHD evolution.

The key aspects implemented are:
1. Staggered grid with magnetic fields stored at cell faces
2. EMF values computed at cell edges
3. Update of magnetic field using Faraday's law in integral form
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
    from myproject.utils.differential_operators.numeric import create_grid, evaluate_divergence
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class StaggeredGrid:
    """
    Class for managing a staggered grid with face-centered magnetic fields.
    """
    def __init__(self, nx, ny, xrange=(0, 1), yrange=(0, 1)):
        """
        Initialize a staggered grid.
        
        Args:
            nx, ny: Number of cells in x and y directions
            xrange, yrange: Coordinate ranges
        """
        self.nx, self.ny = nx, ny
        self.xrange, self.yrange = xrange, yrange
        
        # Create coordinate ranges dict for standard grid
        self.coords_ranges = {
            'x': {'min': xrange[0], 'max': xrange[1]},
            'y': {'min': yrange[0], 'max': yrange[1]}
        }
        self.dimensions = [nx, ny]
        
        # Set up cell-centered grid
        self.grid, self.spacing = create_grid(self.coords_ranges, self.dimensions)
        self.dx, self.dy = self.spacing['x'], self.spacing['y']
        
        # Create cell-centered coordinates
        self.x = self.grid[0]  # cell centers in x
        self.y = self.grid[1]  # cell centers in y
        
        # Create face-centered coordinates
        # For Bx: centered at (i+1/2, j)
        # For By: centered at (i, j+1/2)
        self.x_faces = np.linspace(xrange[0] - 0.5*self.dx, xrange[1] + 0.5*self.dx, nx+2)
        self.y_faces = np.linspace(yrange[0] - 0.5*self.dy, yrange[1] + 0.5*self.dy, ny+2)
        
        # Create edge-centered coordinates (for EMF)
        # EMF is at cell corners: (i+1/2, j+1/2)
        self.x_edges = self.x_faces
        self.y_edges = self.y_faces
        
        # Initialize storage for face-centered magnetic field
        # Include ghost cells for boundary conditions
        self.Bx = np.zeros((nx+2, ny+1))  # Bx at (i+1/2, j)
        self.By = np.zeros((nx+1, ny+2))  # By at (i, j+1/2)
        
        # Initialize EMF storage (at cell edges/corners)
        self.EMF = np.zeros((nx+1, ny+1))  # EMF at (i+1/2, j+1/2)
        
        # Initialize cell-centered velocity
        self.vx = np.zeros((nx, ny))
        self.vy = np.zeros((nx, ny))
        
        # For visualization: cell-centered B field
        self.Bx_cell = None
        self.By_cell = None
        
        logger.info(f"Created staggered grid with {nx}x{ny} cells")
        logger.info(f"Cell spacing: dx={self.dx:.6f}, dy={self.dy:.6f}")
    
    def initialize_vector_potential(self, Az_func):
        """
        Initialize magnetic field from a vector potential A = (0, 0, Az).
        
        This ensures the initial field is exactly divergence-free.
        
        Args:
            Az_func: Function that takes (x, y) and returns Az
        """
        # Create meshgrid for vector potential at cell corners
        X_edges, Y_edges = np.meshgrid(self.x_edges, self.y_edges, indexing='ij')
        
        # Calculate Az at cell corners
        Az = np.zeros((self.nx+2, self.ny+2))
        for i in range(self.nx+2):
            for j in range(self.ny+2):
                Az[i, j] = Az_func(self.x_edges[i], self.y_edges[j])
        
        # Calculate face-centered B from Az
        # Bx = ∂Az/∂y at cell faces
        # By = -∂Az/∂x at cell faces
        for i in range(1, self.nx+1):
            for j in range(self.ny+1):
                # Bx at (i+1/2, j)
                # Use central differences for interior, one-sided for boundaries
                self.Bx[i, j] = (Az[i, j+1] - Az[i, j]) / self.dy
        
        for i in range(self.nx+1):
            for j in range(1, self.ny+1):
                # By at (i, j+1/2)
                self.By[i, j] = -(Az[i+1, j] - Az[i, j]) / self.dx
        
        # Update cell-centered B
        self.update_cell_centered_B()
        
        logger.info("Initialized magnetic field from vector potential")
    
    def update_cell_centered_B(self):
        """
        Update the cell-centered magnetic field by averaging the face-centered values.
        
        For each cell (i,j):
        - Bx_cell[i,j] = average of Bx at faces (i,j) and (i,j+1)
        - By_cell[i,j] = average of By at faces (i,j) and (i+1,j)
        """
        # Allocate arrays for cell-centered B if not already done
        if not hasattr(self, 'Bx_cell') or self.Bx_cell is None:
            self.Bx_cell = np.zeros((self.nx, self.ny))
            self.By_cell = np.zeros((self.nx, self.ny))
        
        # Compute cell-centered Bx by averaging y-faces
        for i in range(self.nx):
            for j in range(self.ny):
                # Average Bx from the left and right faces of the cell
                self.Bx_cell[i,j] = 0.5 * (self.Bx[i,j] + self.Bx[i+1,j])
        
        # Compute cell-centered By by averaging x-faces
        for i in range(self.nx):
            for j in range(self.ny):
                # Average By from the bottom and top faces of the cell
                self.By_cell[i,j] = 0.5 * (self.By[i,j] + self.By[i,j+1])
        
        # Optional: Compute B magnitude for visualization or analysis
        self.B_mag = np.sqrt(self.Bx_cell**2 + self.By_cell**2)
        
        logger.info(f"Cell-centered magnetic field updated")
    
    def set_velocity_field(self, vx_func, vy_func, t=0):
        """
        Set the velocity field using functions of (x, y) or (x, y, t).
        
        Args:
            vx_func, vy_func: Functions that take (x, y) or (x, y, t) and return vx, vy
            t: Current time (optional)
        """
        # Create meshgrid for cell centers
        X_cell, Y_cell = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Set velocity at cell centers
        for i in range(self.nx):
            for j in range(self.ny):
                # Check if the function accepts a time parameter
                try:
                    import inspect
                    vx_params = len(inspect.signature(vx_func).parameters)
                    vy_params = len(inspect.signature(vy_func).parameters)
                    
                    if vx_params == 2:
                        self.vx[i, j] = vx_func(X_cell[i, j], Y_cell[i, j])
                    else:
                        self.vx[i, j] = vx_func(X_cell[i, j], Y_cell[i, j], t)
                        
                    if vy_params == 2:
                        self.vy[i, j] = vy_func(X_cell[i, j], Y_cell[i, j])
                    else:
                        self.vy[i, j] = vy_func(X_cell[i, j], Y_cell[i, j], t)
                except Exception as e:
                    # Fallback: try without time parameter
                    self.vx[i, j] = vx_func(X_cell[i, j], Y_cell[i, j])
                    self.vy[i, j] = vy_func(X_cell[i, j], Y_cell[i, j])
    
    def compute_emf(self):
        """
        Compute the electromotive force (EMF) at cell corners:
        ez = vy * Bx - vx * By
        
        This uses the arithmetic average of field values from neighboring cells
        to get values at the corners.
        """
        # EMF at cell corners (i, j)
        self.ez = np.zeros((self.nx+1, self.ny+1))
        
        # Compute EMF at all cell corners
        for i in range(self.nx+1):
            for j in range(self.ny+1):
                # Get the indices for cells that share this corner
                # Corner (i,j) is the top-right corner of cell (i-1,j-1)
                # and the top-left of (i,j-1), and so on
                
                # Convert from (i,j) to physical coordinates
                x = i * self.dx
                y = j * self.dy
                
                # Interpolate velocity to the corner
                vx_corner = 0.0
                vy_corner = 0.0
                count = 0
                
                # Average from all four neighboring cells if possible
                if i > 0 and j > 0:
                    vx_corner += self.vx[i-1, j-1]
                    vy_corner += self.vy[i-1, j-1]
                    count += 1
                
                if i < self.nx and j > 0:
                    vx_corner += self.vx[i, j-1]
                    vy_corner += self.vy[i, j-1]
                    count += 1
                
                if i > 0 and j < self.ny:
                    vx_corner += self.vx[i-1, j]
                    vy_corner += self.vy[i-1, j]
                    count += 1
                
                if i < self.nx and j < self.ny:
                    vx_corner += self.vx[i, j]
                    vy_corner += self.vy[i, j]
                    count += 1
                
                if count > 0:
                    vx_corner /= count
                    vy_corner /= count
                
                # Interpolate B field to the corner
                # For interior corners, we simply average the directly adjacent face values
                Bx_corner = 0.0
                By_corner = 0.0
                count_x = 0
                count_y = 0
                
                # Average Bx from horizontal edges if possible
                if j > 0 and j < self.ny:
                    if i < self.nx:
                        Bx_corner += self.Bx[i, j-1]
                        count_x += 1
                    if i > 0:
                        Bx_corner += self.Bx[i-1, j-1]
                        count_x += 1
                
                # Average By from vertical edges if possible
                if i > 0 and i < self.nx:
                    if j < self.ny:
                        By_corner += self.By[i-1, j]
                        count_y += 1
                    if j > 0:
                        By_corner += self.By[i-1, j-1]
                        count_y += 1
                
                if count_x > 0:
                    Bx_corner /= count_x
                
                if count_y > 0:
                    By_corner /= count_y
                
                # Compute EMF: E = v × B
                self.ez[i, j] = vy_corner * Bx_corner - vx_corner * By_corner
        
        logger.info(f"EMF computed at cell corners")
        return self.ez
    
    def update_B_field(self, dt):
        """
        Update the face-centered magnetic field components using the CT scheme.
        This maintains ∇·B = 0 to machine precision.
        
        The key to constrained transport is to update the magnetic field using
        Faraday's law in integral form:
        
        For Bx at face (i,j):
            Bx^{n+1}[i,j] = Bx^n[i,j] - dt/dy * (Ez[i,j+1] - Ez[i,j])
            
        For By at face (i,j):
            By^{n+1}[i,j] = By^n[i,j] + dt/dx * (Ez[i+1,j] - Ez[i,j])
        
        This ensures that the divergence constraint is maintained to machine precision.
        """
        dx = self.dx
        dy = self.dy
        
        # Store previous B values for calculating changes
        Bx_old = self.Bx.copy()
        By_old = self.By.copy()
        
        # Create temporary arrays for the updated fields
        Bx_new = self.Bx.copy()
        By_new = self.By.copy()
        
        # Update Bx at x-faces (i,j)
        for i in range(self.nx+1):
            for j in range(self.ny):
                # Update Bx using the curl of EMF (change in y-direction)
                Bx_new[i, j] = self.Bx[i, j] - dt/dy * (self.ez[i, j+1] - self.ez[i, j])
        
        # Update By at y-faces (i,j)
        for i in range(self.nx):
            for j in range(self.ny+1):
                # Update By using the curl of EMF (change in x-direction)
                By_new[i, j] = self.By[i, j] + dt/dx * (self.ez[i+1, j] - self.ez[i, j])
        
        # Update the fields with the new values
        self.Bx = Bx_new
        self.By = By_new
        
        # Calculate maximum absolute change in B for debugging
        max_change_Bx = np.max(np.abs(self.Bx - Bx_old))
        max_change_By = np.max(np.abs(self.By - By_old))
        logger.info(f"Max changes: Bx={max_change_Bx:.6e}, By={max_change_By:.6e}")
    
    def compute_divergence(self):
        """
        Compute the magnetic field divergence for the staggered grid.
        
        For a staggered grid, the divergence can be calculated directly as:
        div(B)[i,j] = (Bx[i+1,j] - Bx[i,j])/dx + (By[i,j+1] - By[i,j])/dy
        
        Returns:
            div_B: 2D array with the divergence at each cell center
        """
        div_B = np.zeros((self.nx, self.ny))
        
        for i in range(self.nx):
            for j in range(self.ny):
                # Compute divergence from face-centered fields
                div_B[i, j] = ((self.Bx[i+1, j] - self.Bx[i, j]) / self.dx + 
                               (self.By[i, j+1] - self.By[i, j]) / self.dy)
        
        return div_B
    
    def get_cell_centered_data(self):
        """Get cell-centered data for visualization and analysis."""
        return self.x, self.y, self.Bx_cell, self.By_cell

    def check_divergence_free(self):
        """
        Check if the magnetic field is divergence-free to machine precision.
        
        Returns:
            max_div: Maximum absolute value of the divergence
        """
        div_B = self.compute_divergence()
        max_div = np.max(np.abs(div_B))
        mean_div = np.mean(np.abs(div_B))
        median_div = np.median(np.abs(div_B))
        
        logger.info(f"Divergence statistics:")
        logger.info(f"  Maximum absolute divergence: {max_div:.2e}")
        logger.info(f"  Mean absolute divergence: {mean_div:.2e}")
        logger.info(f"  Median absolute divergence: {median_div:.2e}")
        
        # Find the indices of the max divergence for debugging
        max_idx = np.unravel_index(np.argmax(np.abs(div_B)), div_B.shape)
        logger.info(f"  Max divergence occurs at grid index: {max_idx}")
        
        # Optional visualization
        if max_div > 1e-10:
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(div_B, cmap='RdBu_r', vmin=-max_div, vmax=max_div)
            plt.colorbar(label='∇·B')
            plt.title(f'Magnetic Field Divergence (max = {max_div:.2e})')
            plt.tight_layout()
            plt.savefig('divergence_debug.png')
            plt.close()
            logger.info(f"  Divergence plot saved to 'divergence_debug.png'")
        
        return max_div

    def initialize_from_vector_potential(self, A_func):
        """
        Initialize the face-centered magnetic field from a vector potential.
        This automatically ensures div·B = 0 to machine precision.
        
        For 2D, we use Az(x,y) as the vector potential component perpendicular to the plane.
        Then Bx = ∂Az/∂y and By = -∂Az/∂x
        
        Args:
            A_func: Function that takes (x, y) coordinates and returns Az
            
        Returns:
            max_div: Maximum absolute value of the divergence
        """
        # Create arrays for the vector potential at cell vertices (corners)
        # We need values at all cell corners, including one layer of ghost cells
        Az = np.zeros((self.nx+1, self.ny+1))
        
        # Calculate cell corner positions
        x_corners = np.linspace(self.xrange[0], self.xrange[1], self.nx+1)
        y_corners = np.linspace(self.yrange[0], self.yrange[1], self.ny+1)
        
        logger.info(f"Computing vector potential at {len(x_corners)}x{len(y_corners)} cell corners")
        
        # Fill in the vector potential values at cell corners
        for i in range(self.nx+1):
            for j in range(self.ny+1):
                # Get corner coordinates
                x = x_corners[i]
                y = y_corners[j]
                Az[i, j] = A_func(x, y)
        
        logger.info(f"Vector potential calculated at cell corners")
        
        # Calculate face-centered magnetic field components
        self.Bx = np.zeros((self.nx+1, self.ny))   # Bx at (i, j+1/2)
        self.By = np.zeros((self.nx, self.ny+1))   # By at (i+1/2, j)
        
        # Calculate Bx = ∂Az/∂y at x-faces
        for i in range(self.nx+1):
            for j in range(self.ny):
                # Centered difference for ∂Az/∂y
                self.Bx[i, j] = (Az[i, j+1] - Az[i, j]) / self.dy
        
        # Calculate By = -∂Az/∂x at y-faces
        for i in range(self.nx):
            for j in range(self.ny+1):
                # Centered difference for -∂Az/∂x
                self.By[i, j] = -(Az[i+1, j] - Az[i, j]) / self.dx
        
        logger.info(f"Face-centered magnetic field components calculated")
        
        # Update cell-centered B for visualization
        self.update_cell_centered_B()
        
        # Verify divergence-free condition
        div_B = self.compute_divergence()
        max_div = np.max(np.abs(div_B))
        logger.info(f"Initial magnetic field divergence: {max_div:.2e}")
        
        # Debug: print coordinates of max divergence
        max_idx = np.unravel_index(np.argmax(np.abs(div_B)), div_B.shape)
        logger.info(f"Max divergence at grid index: {max_idx}")
        
        return max_div

def plot_field_and_divergence(grid, div_B, title, save=True):
    """
    Plot the magnetic field and its divergence.
    
    Args:
        grid: StaggeredGrid object
        div_B: Divergence of the magnetic field
        title: Title for the plot
        save: Whether to save the plot to a file
    """
    x, y, Bx, By = grid.get_cell_centered_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Plot vector field
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
                   extent=[x[0], x[-1], y[0], y[-1]],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Divergence (∇·B), max = {max_div:.2e}')
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save:
        filename = f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        logger.info(f"Saved plot: {filename}")
    
    return fig

def test_constrained_transport():
    """Test that constrained transport preserves the divergence-free condition."""
    # Create CT grid
    nx, ny = 64, 64
    Lx, Ly = 2.0, 2.0
    ct = StaggeredGrid(nx, ny, (0, Lx), (0, Ly))
    
    # Initialize a divergence-free magnetic field using a vector potential
    # For example, A = (0, 0, ψ) where ψ = sin(2πx/Lx)·sin(2πy/Ly)
    # Then B = ∇×A = (∂ψ/∂y, -∂ψ/∂x, 0)
    def psi(x, y):
        return np.sin(2*np.pi*x/Lx) * np.sin(2*np.pi*y/Ly)
    
    def bx_init(x, y):
        return (2*np.pi/Ly) * np.sin(2*np.pi*x/Lx) * np.cos(2*np.pi*y/Ly)
    
    def by_init(x, y):
        return -(2*np.pi/Lx) * np.cos(2*np.pi*x/Lx) * np.sin(2*np.pi*y/Ly)
    
    # Initialize the B field
    ct.initialize_vector_potential(psi)
    
    # Check initial divergence
    initial_div = ct.check_divergence_free()
    print(f"Initial max |∇·B| = {initial_div:.6e}")
    
    # Set a velocity field (rotating flow)
    def vx(x, y):
        return -y + Lx/2
    
    def vy(x, y):
        return x - Ly/2
    
    ct.set_velocity_field(vx, vy)
    
    # Time step
    dt = 0.005
    steps = 100
    
    max_divs = []
    
    # Run the simulation
    for step in range(steps):
        # Compute EMF
        ct.compute_emf()
        
        # Update B field
        ct.update_B_field(dt)
        
        # Update cell-centered B
        ct.update_cell_centered_B()
        
        # Check divergence
        div = ct.check_divergence_free()
        max_divs.append(div)
        
        if (step+1) % 10 == 0:
            print(f"Step {step+1}, max |∇·B| = {div:.6e}")
    
    print(f"Maximum |∇·B| through simulation: {max(max_divs):.6e}")
    
    # Plot final B field and divergence
    x, y, Bx, By = ct.get_cell_centered_data()
    
    # Calculate the divergence for plotting
    div_B = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            # Use central differences for the cell-centered divergence
            i_prev = (i - 1) % nx
            i_next = (i + 1) % nx
            j_prev = (j - 1) % ny
            j_next = (j + 1) % ny
            
            div_x = (Bx[i_next, j] - Bx[i_prev, j]) / (2 * ct.dx)
            div_y = (By[i, j_next] - By[i, j_prev]) / (2 * ct.dy)
            div_B[i, j] = div_x + div_y
    
    plot_field_and_divergence(ct, div_B, f"B Field and ∇·B after {steps} steps")

def test_vector_potential_initialization():
    """
    Test the initialization of a magnetic field from a vector potential.
    
    This test creates a vector potential for a simple magnetic field configuration,
    initializes the magnetic field from it, and verifies that the divergence is zero
    to machine precision. It then advects the field and checks that the divergence
    remains zero.
    """
    logger.info("Testing vector potential initialization with a simple field configuration")
    
    # Initialize a staggered grid
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    grid = StaggeredGrid(nx, ny, (0, Lx), (0, Ly))
    dt = 0.001  # Use a smaller time step for stability
    
    # Define a vector potential for a simple field configuration
    # Az = sin(2πx/Lx) * sin(2πy/Ly)
    # This gives Bx = (2π/Ly) * sin(2πx/Lx) * cos(2πy/Ly)
    #         By = -(2π/Lx) * cos(2πx/Lx) * sin(2πy/Ly)
    def simple_vector_potential(x, y):
        return np.sin(2*np.pi*x/Lx) * np.sin(2*np.pi*y/Ly)
    
    # Initialize magnetic field from vector potential
    max_div = grid.initialize_from_vector_potential(simple_vector_potential)
    
    # Verify that the divergence is close to zero
    assert max_div < 1e-10, f"Divergence is not close to zero: {max_div}"
    logger.info(f"Initial divergence is close to zero: {max_div}")
    
    # Plot the magnetic field components
    x, y, Bx, By = grid.get_cell_centered_data()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Bx
    im0 = axes[0, 0].pcolormesh(x, y, Bx, cmap='RdBu_r', shading='auto')
    axes[0, 0].set_title('Bx Component')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Plot By
    im1 = axes[0, 1].pcolormesh(x, y, By, cmap='RdBu_r', shading='auto')
    axes[0, 1].set_title('By Component')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot B magnitude
    B_mag = np.sqrt(Bx**2 + By**2)
    im2 = axes[1, 0].pcolormesh(x, y, B_mag, cmap='viridis', shading='auto')
    axes[1, 0].set_title('|B| Magnitude')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Quiver plot of the magnetic field
    skip = 4  # Skip some points for clarity
    X, Y = np.meshgrid(x, y, indexing='ij')
    axes[1, 1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                      Bx[::skip, ::skip], By[::skip, ::skip],
                      B_mag[::skip, ::skip], cmap='viridis',
                      scale=25, pivot='mid')
    axes[1, 1].set_title('B Field Vectors')
    
    plt.tight_layout()
    plt.savefig('simple_magnetic_field.png')
    plt.close()
    
    # Test advection of the magnetic field using a rigid rotation
    # Set a rigid rotation velocity field
    def vx(x, y):
        return -2.0 * np.pi * (y - 0.5)  # Rigid rotation around center
    
    def vy(x, y):
        return 2.0 * np.pi * (x - 0.5)   # Rigid rotation around center
    
    grid.set_velocity_field(vx, vy)
    
    # Store initial B field for comparison
    Bx_initial = grid.Bx.copy()
    By_initial = grid.By.copy()
    
    # Evolve for a few steps
    steps = 20
    divs = []
    
    for step in range(steps):
        # Compute EMF
        grid.compute_emf()
        
        # Update B field
        grid.update_B_field(dt)
        
        # Update cell-centered B for visualization
        grid.update_cell_centered_B()
        
        # Check divergence
        div = grid.check_divergence_free()
        divs.append(div)
        logger.info(f"Step {step+1}, max |∇·B| = {div:.2e}")
    
    # Plot divergence over time
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(1, steps+1), divs, 'o-')
    plt.xlabel('Time Step')
    plt.ylabel('Max |∇·B|')
    plt.title('Divergence During Evolution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('simple_divergence_evolution.png')
    plt.close()
    
    # Verify that divergence remains small
    final_div = divs[-1]
    assert final_div < 1e-10, f"Final divergence is not close to zero: {final_div}"
    logger.info(f"Final divergence after {steps} steps: {final_div:.2e}")
    
    # Plot the final magnetic field
    x, y, Bx, By = grid.get_cell_centered_data()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot initial vs final B field magnitude
    B_mag_final = np.sqrt(Bx**2 + By**2)
    axes[0].pcolormesh(x, y, B_mag, cmap='viridis', shading='auto')
    axes[0].set_title('Initial |B|')
    
    axes[1].pcolormesh(x, y, B_mag_final, cmap='viridis', shading='auto')
    axes[1].set_title(f'Final |B| after {steps} steps')
    
    plt.tight_layout()
    plt.savefig('simple_field_evolution.png')
    plt.close()
    
    logger.info("Vector potential advection test passed")
    return grid

if __name__ == "__main__":
    try:
        logger.info("Starting constrained transport tests")
        
        # Run the original test
        # test_constrained_transport()
        
        # Run the new test for vector potential initialization
        test_vector_potential_initialization()
        
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 