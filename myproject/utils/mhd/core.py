"""
Core MHD (Magnetohydrodynamics) implementation for the tensor calculator.

This module provides the fundamental components for MHD simulations based on
the finite-volume method with high-resolution shock-capturing techniques.
It implements ideal and resistive MHD in conservative form to ensure
conservation of mass, momentum, energy, and magnetic flux.
"""

import sys
import os

# Add the parent directory of 'myproject' to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import sympy as sp
from numba import njit, prange
from myproject.utils.differential_operators import (
    evaluate_gradient, evaluate_divergence, evaluate_curl,
    create_grid, metric_from_transformation
)
from myproject.utils.numerical.tensor_utils import flatten_3d_array

class MHDSystem:
    """
    Class representing an MHD system with equations, state, and evolution methods.
    
    This class follows the architecture described in the documentation, implementing
    MHD as a specific EquationSystem that can be evolved using various solvers.
    """
    
    def __init__(self, coordinate_system, domain_size, resolution,
                 gamma=5/3, resistivity=0.0, use_constrained_transport=True):
        """
        Initialize an MHD system with the given parameters.
        
        Args:
            coordinate_system: The coordinate system to use
            domain_size: Size of the computational domain (list of min/max values)
            resolution: Grid resolution (list of points in each dimension)
            gamma: Adiabatic index (default: 5/3)
            resistivity: Magnetic resistivity (default: 0 for ideal MHD)
            use_constrained_transport: Whether to use constrained transport for 
                                      maintaining div(B) = 0 (default: True)
        """
        self.coordinate_system = coordinate_system
        self.domain_size = domain_size
        self.resolution = resolution
        self.gamma = gamma
        self.resistivity = resistivity
        self.use_constrained_transport = use_constrained_transport
        
        # Grid setup
        self.dimension = len(resolution)
        self.setup_grid()
        
        # Physical fields - these will store the current state
        self.density = None
        self.velocity = [None] * self.dimension
        self.pressure = None
        self.magnetic_field = [None] * self.dimension
        
        # Derived quantities
        self.conserved_vars = None
        self.max_wavespeed = None
        
        # Time evolution parameters
        self.time = 0.0
        self.dt = None
        self.cfl_number = 0.4  # Default CFL safety factor
        
    def setup_grid(self):
        """Set up the computational grid based on the coordinate system."""
        # Convert domain_size to the format expected by create_grid
        coords_ranges = {}
        for i in range(self.dimension):
            coord_name = self.coordinate_system.get('coordinates', [f'x{i}'])[i]
            coords_ranges[coord_name] = {
                'min': self.domain_size[i][0],
                'max': self.domain_size[i][1]
            }
        
        # Create the grid using the properly formatted parameters
        self.grid, self.spacing = create_grid(coords_ranges, self.resolution)
        
        # Create metric tensors for the chosen coordinate system
        if isinstance(self.coordinate_system, dict):
            # Get coordinate symbols
            coord_names = self.coordinate_system.get('coordinates', [f'x{i}' for i in range(self.dimension)])
            coord_symbols = [sp.Symbol(name) for name in coord_names]
            
            if 'transformation' in self.coordinate_system and self.coordinate_system['transformation'] is not None:
                # Custom coordinate system with transformation functions defined
                transform_map = self.coordinate_system['transformation']
            else:
                # Default to identity transformation for Cartesian coordinates
                # Create a list of sympy expressions for the identity transformation
                transform_map = coord_symbols.copy()  # Identity transformation: x' = x, y' = y, etc.
                
            # Compute the metric
            self.metric = metric_from_transformation(
                transform_map,
                sp.eye(self.dimension),  # Identity matrix for Cartesian
                coord_symbols
            )
        else:
            # Use a predefined coordinate system
            # (This is a placeholder - would need to be implemented)
            self.metric = sp.eye(self.dimension)
        
        # Convert to numeric metric for computations
        # (This is a placeholder - would need to be implemented)
        self.numeric_metric = np.eye(self.dimension)
    
    def set_initial_conditions(self, density_func, velocity_func, 
                              pressure_func, magnetic_field_func):
        """
        Set the initial conditions for the MHD simulation.
        
        Args:
            density_func: Function that returns density values at grid points
            velocity_func: List of functions that return velocity components
            pressure_func: Function that returns pressure values at grid points
            magnetic_field_func: List of functions that return magnetic field components
        """
        # Initialize physical fields
        self.density = density_func(*self.grid)
        
        for i in range(self.dimension):
            self.velocity[i] = velocity_func[i](*self.grid)
            self.magnetic_field[i] = magnetic_field_func[i](*self.grid)
            
        self.pressure = pressure_func(*self.grid)
        
        # Initialize constrained magnetic field if using constrained transport
        if self.use_constrained_transport:
            self.initialize_constrained_magnetic_field()
            
        # Compute conserved variables from primitives
        self.compute_conserved_variables()
    
    def initialize_constrained_magnetic_field(self, vector_potential_func=None):
        """
        Initialize magnetic field to maintain div(B) = 0 constraint.
        
        Uses a staggered grid approach for constrained transport.
        
        Args:
            vector_potential_func: Optional function that returns vector potential components.
                                 If provided, magnetic field is initialized directly from
                                 the vector potential, ensuring div(B) = 0 to machine precision.
                                 For 2D: A single function for A_z component is sufficient.
                                 For 3D: A list of 3 functions for [A_x, A_y, A_z] components.
        
        Returns:
            Maximum absolute value of divergence
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Extract grid spacing
        grid_spacing = [self.spacing[coord] for coord in self.coordinate_system.get(
            'coordinates', [f'x{i}' for i in range(self.dimension)])]
        
        if vector_potential_func is not None:
            # Initialize directly from vector potential
            logger.info("Initializing magnetic field from vector potential")
            from .constrained_transport import initialize_from_vector_potential
            
            # Initialize face-centered B from vector potential
            self.face_centered_b = initialize_from_vector_potential(
                vector_potential_func, self.grid, grid_spacing)
            
            # Convert face-centered B to cell-centered for conventional use
            from .constrained_transport import face_to_cell_centered_b
            cell_centered_b = face_to_cell_centered_b(self.face_centered_b)
            
            # Update the cell-centered magnetic field
            for i in range(self.dimension):
                self.magnetic_field[i] = cell_centered_b[i]
        else:
            # Initialize face-centered fields from cell-centered B
            logger.info("Initializing face-centered magnetic field from cell-centered values")
            from .constrained_transport import initialize_face_centered_b
            
            self.face_centered_b = initialize_face_centered_b(
                self.magnetic_field, [len(self.grid[i]) for i in range(self.dimension)])
        
        # Check divergence
        from .constrained_transport import check_divergence_free
        max_div = check_divergence_free(self.face_centered_b, grid_spacing)
        logger.info(f"Maximum divergence after initialization: {max_div:.6e}")
        
        return max_div
        
    def compute_conserved_variables(self):
        """
        Compute the conserved variables of the MHD system.
        
        The conserved variables are:
        - density: ρ
        - momentum: ρv
        - energy: E = ρε + 0.5ρv² + 0.5B²
        - magnetic field: B
        
        Returns:
            tuple: Tuple containing the conserved variables (density, momentum, energy, magnetic field)
        """
        logger = logging.getLogger(__name__)
        
        # Check if any fields are 1D but should be 2D based on system dimension
        if self.dimension == 2 and len(self.density.shape) == 1:
            logger.info("Reshaping 1D arrays to 2D for 2D system")
            nx = len(self.grid[0])
            ny = len(self.grid[1])
            
            # Create 2D arrays
            density_2d = np.zeros((nx, ny))
            pressure_2d = np.zeros((nx, ny))
            velocity_2d = [np.zeros((nx, ny)) for _ in range(self.dimension)]
            
            # Fill in values (broadcasting 1D to 2D)
            for i in range(nx):
                for j in range(ny):
                    density_2d[i, j] = self.density[i] if i < len(self.density) else 0
                    pressure_2d[i, j] = self.pressure[i] if i < len(self.pressure) else 0
                    for d in range(self.dimension):
                        velocity_2d[d][i, j] = self.velocity[d][i] if i < len(self.velocity[d]) else 0
            
            # Update the original arrays
            self.density = density_2d
            self.pressure = pressure_2d
            self.velocity = velocity_2d
        
        # Check if magnetic field components have the same shape as density
        B_interpolated = list(self.magnetic_field)
        need_interpolation = False
        
        if self.magnetic_field is not None:
            for i, B_component in enumerate(self.magnetic_field):
                if B_component.shape != self.density.shape:
                    need_interpolation = True
                    break
        
        if need_interpolation:
            logger.info("Magnetic field components have different shapes, interpolating to cell centers")
            logger.info(f"Density shape: {self.density.shape}")
            
            for i, B_component in enumerate(self.magnetic_field):
                logger.info(f"B[{i}] shape: {B_component.shape}")
                
                # For 2D fields, interpolate from face-centered to cell-centered
                if len(self.density.shape) == 2 and len(B_component.shape) == 2:
                    # Create array with same shape as density
                    B_interp = np.zeros_like(self.density)
                    
                    # Determine valid range for interpolation (avoiding edges)
                    ni, nj = B_component.shape
                    
                    # Simple averaging for internal points
                    for i_idx in range(min(ni, self.density.shape[0]-1)):
                        for j_idx in range(min(nj, self.density.shape[1]-1)):
                            B_interp[i_idx, j_idx] = B_component[i_idx, j_idx]
                    
                    B_interpolated[i] = B_interp
            
            # Update the magnetic field with interpolated values
            self.magnetic_field = B_interpolated
        
        # Allocate arrays for conserved variables
        density = self.density
        momentum = [self.density * self.velocity[i] for i in range(self.dimension)]
        energy = np.zeros_like(self.density)
        
        # Compute energy
        # First, add internal energy
        energy = self.pressure / (self.gamma - 1)
        
        # Add kinetic energy
        for i in range(self.dimension):
            energy += 0.5 * self.density * self.velocity[i]**2
        
        # Add magnetic energy if magnetic field is present
        if self.magnetic_field is not None:
            B_squared = np.zeros_like(self.density)
            for i in range(self.dimension):
                B_squared += self.magnetic_field[i]**2
            energy += 0.5 * B_squared
        
        return density, momentum, energy, self.magnetic_field
    
    def compute_primitive_variables(self):
        """
        Compute primitive variables from conserved variables.
        """
        cons = self.conserved_vars
        rho = cons['density']
        
        # Velocity: momentum / density
        for i in range(self.dimension):
            self.velocity[i] = cons['momentum'][i] / rho
            
        # Update magnetic field (if not using constrained transport)
        if not self.use_constrained_transport:
            for i in range(self.dimension):
                self.magnetic_field[i] = cons['magnetic_field'][i]
        
        # Compute magnetic energy: B^2/2
        B_squared = np.zeros_like(rho)
        for i in range(self.dimension):
            B_squared += self.magnetic_field[i]**2
        magnetic_energy = 0.5 * B_squared
        
        # Compute kinetic energy: rho * v^2 / 2
        kinetic_energy = np.zeros_like(rho)
        for i in range(self.dimension):
            kinetic_energy += 0.5 * rho * self.velocity[i]**2
            
        # Internal energy: total energy - kinetic - magnetic
        internal_energy = cons['energy'] - kinetic_energy - magnetic_energy
        
        # Pressure: internal energy * (gamma - 1)
        self.pressure = internal_energy * (self.gamma - 1)
    
    @njit
    def compute_mhd_flux(self, rho, v, p, B, gamma, direction):
        """
        Compute the MHD flux in the specified direction.
        
        This method computes the numerical flux for the conservation law form
        of the MHD equations: ∂U/∂t + ∇·F(U) = 0
        
        Args:
            rho: Density
            v: List of velocity components
            p: Pressure
            B: List of magnetic field components
            gamma: Adiabatic index
            direction: Direction in which to compute the flux (0, 1, or 2)
            
        Returns:
            Dictionary containing the flux components
        """
        # Implementation would compute the MHD flux tensor
        # This is a placeholder
        pass
    
    def compute_wavespeeds(self):
        """
        Compute the maximum wave speeds for determining the CFL time step.
        
        Returns:
            Maximum wave speed in the domain
        """
        rho = self.density
        v = self.velocity
        p = self.pressure
        B = self.magnetic_field
        
        # Sound speed: c_s = sqrt(gamma * p / rho)
        sound_speed = np.sqrt(self.gamma * p / rho)
        
        # Alfven speed: c_A = |B| / sqrt(rho)
        B_magnitude = np.zeros_like(rho)
        for i in range(self.dimension):
            B_magnitude += B[i]**2
        B_magnitude = np.sqrt(B_magnitude)
        alfven_speed = B_magnitude / np.sqrt(rho)
        
        # Fast magnetosonic speed: c_f^2 = 0.5 * ((c_s^2 + c_A^2) + 
        #                               sqrt((c_s^2 + c_A^2)^2 - 4 * c_s^2 * c_A^2 * cos^2(theta)))
        # For simplicity, we'll use the upper bound: c_f <= c_s + c_A
        fast_speed = sound_speed + alfven_speed
        
        # Max wave speed is velocity magnitude plus fast speed
        max_speed = np.zeros_like(rho)
        for i in range(self.dimension):
            max_speed = np.maximum(max_speed, np.abs(v[i]) + fast_speed)
            
        self.max_wavespeed = np.max(max_speed)
        return self.max_wavespeed
    
    def compute_time_step(self):
        """
        Compute the time step based on the CFL condition.
        
        Returns:
            Time step size (dt)
        """
        if self.max_wavespeed is None:
            self.compute_wavespeeds()
            
        # Determine the minimum grid spacing
        min_dx = float('inf')
        for i in range(self.dimension):
            dx = (self.domain_size[i][1] - self.domain_size[i][0]) / self.resolution[i]
            min_dx = min(min_dx, dx)
            
        # CFL condition: dt <= CFL * dx / max_wavespeed
        self.dt = self.cfl_number * min_dx / self.max_wavespeed
        return self.dt
    
    def evolve(self, final_time, output_callback=None, output_interval=None):
        """
        Evolve the MHD system to the specified final time.
        
        Args:
            final_time: Time to evolve the system to
            output_callback: Function to call for output at specified intervals
            output_interval: Time interval between outputs
            
        Returns:
            The final state of the system
        """
        if self.dt is None:
            self.compute_time_step()
            
        if output_interval is None:
            output_interval = (final_time - self.time) / 10  # Default to 10 outputs
            
        next_output_time = self.time + output_interval
        
        # Time integration loop
        while self.time < final_time:
            # Adjust dt if needed to hit the final time exactly
            if self.time + self.dt > final_time:
                self.dt = final_time - self.time
                
            # Integrate one time step
            self.advance_time_step()
            
            # Check if it's time for output
            if output_callback and self.time >= next_output_time:
                output_callback(self)
                next_output_time += output_interval
                
        return {
            'density': self.density,
            'velocity': self.velocity,
            'pressure': self.pressure,
            'magnetic_field': self.magnetic_field,
            'time': self.time
        }
    
    def advance_time_step(self):
        """
        Advance the solution by one time step using a time integrator.
        
        This method implements a standard method of lines approach with 
        an explicit time integrator (e.g., RK2 or RK3).
        """
        # Example: Forward Euler time integration
        # 1. Compute the right-hand side (spatial derivatives)
        rhs = self.compute_rhs()
        
        # 2. Update conserved variables: U^(n+1) = U^n + dt * RHS
        for var_name, var_data in self.conserved_vars.items():
            if var_name == 'momentum' or var_name == 'magnetic_field':
                # Handle vector quantities
                for i in range(self.dimension):
                    var_data[i] += self.dt * rhs[var_name][i]
            else:
                # Handle scalar quantities
                var_data += self.dt * rhs[var_name]
                
        # 3. Apply constrained transport to maintain div(B) = 0
        if self.use_constrained_transport:
            self.apply_constrained_transport()
            
        # 4. Compute primitive variables from updated conserved variables
        self.compute_primitive_variables()
        
        # 5. Update time
        self.time += self.dt
        
        # 6. Recompute time step for next iteration
        self.compute_time_step()
        
    def compute_rhs(self):
        """
        Compute the right-hand side of the MHD equations.
        
        This computes the spatial derivatives for the method of lines approach:
        dU/dt = RHS(U) = -∇·F(U)
        
        Returns:
            Dictionary containing the RHS terms for each conserved variable
        """
        # Implementation would compute fluxes and their divergences
        # This is a placeholder
        return {
            'density': np.zeros_like(self.density),
            'momentum': [np.zeros_like(self.density) for _ in range(self.dimension)],
            'energy': np.zeros_like(self.density),
            'magnetic_field': [np.zeros_like(self.density) for _ in range(self.dimension)]
        }
    
    def apply_constrained_transport(self):
        """
        Apply constrained transport to maintain div(B) = 0.
        """
        # Implementation would ensure the magnetic field satisfies div(B) = 0
        # This is a placeholder
        pass
    
    def check_divergence_free(self):
        """
        Check if the magnetic field is divergence-free.
        
        Returns:
            Maximum absolute value of div(B) across the domain
        """
        div_B = evaluate_divergence(self.magnetic_field, self.numeric_metric, self.grid)
        return np.max(np.abs(div_B))
    
    def initialize_from_vector_potential(self, density_func, velocity_func, 
                                        pressure_func, vector_potential_func):
        """
        Initialize the MHD system with a magnetic field specified through a vector potential.
        
        This ensures that the magnetic field is divergence-free to machine precision.
        
        Args:
            density_func: Function that returns density values at grid points
            velocity_func: List of functions that return velocity components
            pressure_func: Function that returns pressure values at grid points
            vector_potential_func: Function or list of functions that return vector potential components.
                                For 2D: A single function for A_z component is sufficient.
                                For 3D: A list of 3 functions for [A_x, A_y, A_z] components.
        
        Returns:
            Maximum absolute value of divergence
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("Initializing MHD system from vector potential")
        
        # Initialize physical fields (except magnetic field)
        self.density = density_func(*self.grid)
        
        for i in range(self.dimension):
            self.velocity[i] = velocity_func[i](*self.grid)
            
        self.pressure = pressure_func(*self.grid)
        
        # Initialize temporary magnetic field with zeros
        # (These will be replaced by the values computed from the vector potential)
        for i in range(self.dimension):
            self.magnetic_field[i] = np.zeros_like(self.density)
        
        # Initialize constrained magnetic field from vector potential
        max_div = self.initialize_constrained_magnetic_field(vector_potential_func)
        
        # Compute conserved variables from primitives
        self.compute_conserved_variables()
        
        logger.info(f"MHD system initialization complete, max |div(B)| = {max_div:.6e}")
        
        return max_div


# Helper functions for common MHD initial conditions
def orszag_tang_vortex_2d(domain_size, resolution, gamma=5/3):
    """
    Create an MHD system with the Orszag-Tang vortex initial condition.
    
    This is a standard test case for MHD codes that produces complex flow
    structures and tests the code's ability to handle MHD turbulence.
    
    Args:
        domain_size: Size of the computational domain
        resolution: Grid resolution
        gamma: Adiabatic index
        
    Returns:
        Initialized MHD system
    """
    # Define coordinate system (Cartesian)
    coordinate_system = {
        'name': 'cartesian',
        'coordinates': ['x', 'y'],
        # No transformation needed for Cartesian coordinates - will use identity transformation
        'transformation': None  
    }
    
    # Create MHD system
    mhd = MHDSystem(coordinate_system, domain_size, resolution, gamma)
    
    # Grid points
    x, y = mhd.grid
    
    # Orszag-Tang vortex initial conditions
    density = np.ones_like(x)
    
    velocity_x = -np.sin(2 * np.pi * y)
    velocity_y = np.sin(2 * np.pi * x)
    velocity = [velocity_x, velocity_y]
    
    pressure = 1.0 / mhd.gamma * np.ones_like(x)
    
    magnetic_x = -np.sin(2 * np.pi * y)
    magnetic_y = np.sin(4 * np.pi * x)
    magnetic = [magnetic_x, magnetic_y]
    
    # Set initial conditions
    mhd.set_initial_conditions(
        lambda x, y: density,
        [lambda x, y: velocity_x, lambda x, y: velocity_y],
        lambda x, y: pressure,
        [lambda x, y: magnetic_x, lambda x, y: magnetic_y]
    )
    
    return mhd

def magnetic_rotor_2d(domain_size, resolution, gamma=5/3):
    """
    Create an MHD system with the magnetic rotor initial condition.
    
    This test involves a dense disk rotating in a static background,
    threaded by an initially uniform magnetic field.
    
    Args:
        domain_size: Size of the computational domain
        resolution: Grid resolution
        gamma: Adiabatic index
        
    Returns:
        Initialized MHD system
    """
    # Define coordinate system (Cartesian)
    coordinate_system = {
        'name': 'cartesian',
        'coordinates': ['x', 'y'],
        # No transformation needed for Cartesian coordinates - will use identity transformation
        'transformation': None  
    }
    
    # Create MHD system
    mhd = MHDSystem(coordinate_system, domain_size, resolution, gamma)
    
    # Grid points
    x, y = mhd.grid
    
    # Parameters
    x0, y0 = 0.5, 0.5  # Center of domain
    r0 = 0.1  # Radius of rotor
    r1 = 0.115  # Transition region outer radius
    v0 = 2.0  # Rotation speed
    rho0 = 10.0  # Density inside rotor
    rho1 = 1.0  # Density outside
    p0 = 1.0  # Pressure
    B0 = 5.0 / np.sqrt(4 * np.pi)  # Initial magnetic field strength
    
    # Compute distance from center
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    
    # Define density with smooth transition
    density = np.ones_like(r) * rho1
    mask_inner = r <= r0
    mask_transition = (r > r0) & (r < r1)
    density[mask_inner] = rho0
    # Smooth transition in transition region
    f = (r1 - r[mask_transition]) / (r1 - r0)
    density[mask_transition] = rho1 + (rho0 - rho1) * f
    
    # Define velocity with smooth transition
    velocity_x = np.zeros_like(r)
    velocity_y = np.zeros_like(r)
    
    velocity_x[mask_inner] = -v0 * (y[mask_inner] - y0) / r0
    velocity_y[mask_inner] = v0 * (x[mask_inner] - x0) / r0
    
    velocity_x[mask_transition] = -v0 * (y[mask_transition] - y0) / r0 * f
    velocity_y[mask_transition] = v0 * (x[mask_transition] - x0) / r0 * f
    
    velocity = [velocity_x, velocity_y]
    
    # Define pressure (uniform)
    pressure = p0 * np.ones_like(r)
    
    # Define magnetic field (initially uniform in x-direction)
    magnetic_x = B0 * np.ones_like(r)
    magnetic_y = np.zeros_like(r)
    magnetic = [magnetic_x, magnetic_y]
    
    # Set initial conditions
    mhd.set_initial_conditions(
        lambda x, y: density,
        [lambda x, y: velocity_x, lambda x, y: velocity_y],
        lambda x, y: pressure,
        [lambda x, y: magnetic_x, lambda x, y: magnetic_y]
    )
    
    return mhd

# Placeholder for Riemann solvers (to be implemented)
def hll_riemann_solver(U_L, U_R, flux_function, max_wave_speed):
    """
    HLL (Harten-Lax-van Leer) Riemann solver for MHD.
    
    Args:
        U_L: Left state
        U_R: Right state
        flux_function: Function to compute the flux
        max_wave_speed: Maximum wave speed
        
    Returns:
        HLL flux
    """
    # Implementation would compute the HLL flux
    # This is a placeholder
    pass 