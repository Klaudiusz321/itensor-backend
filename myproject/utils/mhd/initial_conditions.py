"""
Initial Conditions module for MHD simulations.

This module provides standard test problems and initial conditions
for MHD simulations, such as the Orszag-Tang vortex, MHD blast wave,
magnetic rotor, and shock tubes.
"""

import numpy as np

def orszag_tang_vortex(grid, gamma=5/3):
    """
    Initialize the Orszag-Tang vortex, a standard 2D MHD test case.
    
    This problem leads to complex shock interactions and vortical structures
    that test the robustness of an MHD code.
    
    Args:
        grid: Grid coordinates (x, y)
        gamma: Adiabatic index
        
    Returns:
        Dictionary with density, velocity, pressure, and magnetic field
    """
    x, y = grid
    
    # Initialize fields on the grid
    density = np.ones_like(x)
    
    # Velocity field: -sin(2πy), sin(2πx)
    velocity_x = -np.sin(2 * np.pi * y)
    velocity_y = np.sin(2 * np.pi * x)
    velocity = [velocity_x, velocity_y]
    
    # Pressure field: uniform 
    pressure = 1.0 / gamma * np.ones_like(x)
    
    # Magnetic field: -sin(2πy), sin(4πx)
    magnetic_x = -np.sin(2 * np.pi * y)
    magnetic_y = np.sin(4 * np.pi * x)
    magnetic = [magnetic_x, magnetic_y]
    
    return {
        'density': density,
        'velocity': velocity,
        'pressure': pressure,
        'magnetic_field': magnetic
    }

def magnetic_rotor(grid, gamma=5/3):
    """
    Initialize the magnetic rotor problem, a 2D MHD test case.
    
    This test involves a rotating dense disk in a uniform magnetic field,
    which generates torsional Alfvén waves.
    
    Args:
        grid: Grid coordinates (x, y)
        gamma: Adiabatic index
        
    Returns:
        Dictionary with density, velocity, pressure, and magnetic field
    """
    x, y = grid
    
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
    
    return {
        'density': density,
        'velocity': velocity,
        'pressure': pressure,
        'magnetic_field': magnetic
    }

def mhd_blast_wave(grid, gamma=5/3):
    """
    Initialize the MHD blast wave problem, a 2D test case.
    
    This test involves a high pressure region expanding into a magnetized medium,
    testing the code's ability to handle strong shock interactions with the field.
    
    Args:
        grid: Grid coordinates (x, y)
        gamma: Adiabatic index
        
    Returns:
        Dictionary with density, velocity, pressure, and magnetic field
    """
    x, y = grid
    
    # Parameters
    x0, y0 = 0.5, 0.5  # Center of domain
    r0 = 0.1  # Radius of high-pressure region
    rho0 = 1.0  # Uniform density
    p_in = 100.0  # Pressure inside blast
    p_out = 1.0  # Pressure outside
    B0 = 10.0  # Magnetic field strength
    
    # Compute distance from center
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    
    # Uniform density
    density = rho0 * np.ones_like(r)
    
    # Zero initial velocity
    velocity_x = np.zeros_like(r)
    velocity_y = np.zeros_like(r)
    velocity = [velocity_x, velocity_y]
    
    # Pressure with high-pressure region
    pressure = np.ones_like(r) * p_out
    mask_inner = r <= r0
    pressure[mask_inner] = p_in
    
    # Uniform magnetic field in x-direction
    magnetic_x = B0 * np.ones_like(r)
    magnetic_y = np.zeros_like(r)
    magnetic = [magnetic_x, magnetic_y]
    
    return {
        'density': density,
        'velocity': velocity,
        'pressure': pressure,
        'magnetic_field': magnetic
    }

def mhd_shock_tube(grid, direction=0, gamma=5/3):
    """
    Initialize a 1D MHD shock tube problem (Brio-Wu or similar).
    
    This is a standard 1D test problem for MHD codes, verifying the code's
    ability to handle shock waves, rarefactions, and discontinuities.
    
    Args:
        grid: Grid coordinates (can be 1D, 2D, or 3D, but problem is 1D)
        direction: Direction of the 1D problem (0=x, 1=y, 2=z)
        gamma: Adiabatic index
        
    Returns:
        Dictionary with density, velocity, pressure, and magnetic field
    """
    # Extract the directional coordinate
    coords = grid[direction]
    
    # Determine dimension from grid
    dimension = len(grid)
    
    # Create arrays with the same shape as the grid
    shape = coords.shape
    
    # Initialize fields
    density = np.ones_like(coords)
    velocity = [np.zeros_like(coords) for _ in range(dimension)]
    pressure = np.ones_like(coords)
    magnetic = [np.zeros_like(coords) for _ in range(dimension)]
    
    # Brio-Wu shock tube setup
    midpoint = 0.5  # Domain is assumed to be [0, 1]
    
    # Left state (x < 0.5)
    left_mask = coords < midpoint
    density[left_mask] = 1.0
    pressure[left_mask] = 1.0
    magnetic[0][left_mask] = 0.75  # Bx
    magnetic[1][left_mask] = 1.0   # By
    
    # Right state (x >= 0.5)
    right_mask = coords >= midpoint
    density[right_mask] = 0.125
    pressure[right_mask] = 0.1
    magnetic[0][right_mask] = 0.75  # Bx (continuous across interface)
    magnetic[1][right_mask] = -1.0  # By
    
    return {
        'density': density,
        'velocity': velocity,
        'pressure': pressure,
        'magnetic_field': magnetic
    }

def kelvin_helmholtz_mhd(grid, gamma=5/3):
    """
    Initialize a Kelvin-Helmholtz instability with magnetic field.
    
    This test involves a shear layer with a small perturbation,
    which grows due to the KH instability but is affected by the magnetic field.
    
    Args:
        grid: Grid coordinates (x, y)
        gamma: Adiabatic index
        
    Returns:
        Dictionary with density, velocity, pressure, and magnetic field
    """
    x, y = grid
    
    # Parameters
    rho1 = 1.0    # Density in lower half
    rho2 = 2.0    # Density in upper half
    v1 = 0.5      # Velocity in lower half
    v2 = -0.5     # Velocity in upper half
    p0 = 2.5      # Uniform pressure
    B0 = 0.1      # Magnetic field strength
    sigma = 0.05  # Width of transition layer
    A = 0.01      # Amplitude of perturbation
    k = 4.0       # Wavenumber of perturbation
    
    # Initialize with a smooth density and velocity transition
    y_mid = 0.5
    density = rho1 + (rho2 - rho1) * (1 + np.tanh((y - y_mid) / sigma)) / 2
    
    # Velocity with a perturbation to trigger the instability
    velocity_x = v1 + (v2 - v1) * (1 + np.tanh((y - y_mid) / sigma)) / 2
    velocity_x += A * np.sin(2 * np.pi * k * x) * np.exp(-((y - y_mid) / sigma)**2)
    
    velocity_y = np.zeros_like(x)
    velocity = [velocity_x, velocity_y]
    
    # Uniform pressure
    pressure = p0 * np.ones_like(x)
    
    # Magnetic field along the flow (stabilizes the instability)
    magnetic_x = B0 * np.ones_like(x)
    magnetic_y = np.zeros_like(x)
    magnetic = [magnetic_x, magnetic_y]
    
    return {
        'density': density,
        'velocity': velocity,
        'pressure': pressure,
        'magnetic_field': magnetic
    } 