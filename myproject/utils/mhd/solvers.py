"""
MHD Solver module implementing various Riemann solvers for MHD simulations.

This module provides numerical Riemann solvers for the MHD equations, which
are used to compute fluxes at cell interfaces in finite volume methods.
"""

import numpy as np
from numba import njit
import logging
import time

# Setup logging
logger = logging.getLogger(__name__)

@njit
def _compute_mhd_flux_impl(rho, v, p, B, gamma, direction):
    """
    Numba-optimized implementation of the MHD flux computation.
    This version always works with arrays.
    """
    # Get velocity component in the flux direction
    v_dir = v[direction]
    
    # Compute B² using numpy operations instead of accumulation
    B_squared = np.zeros_like(rho)
    for i in range(len(B)):
        B_squared = B_squared + B[i]**2
    
    # Compute kinetic energy using numpy operations
    kinetic_energy = np.zeros_like(rho)
    for i in range(len(v)):
        kinetic_energy = kinetic_energy + 0.5 * rho * v[i]**2
    
    # Compute magnetic pressure
    magnetic_pressure = 0.5 * B_squared
    
    # Compute total pressure
    total_pressure = p + magnetic_pressure
    
    # Compute total energy
    total_energy = p / (gamma - 1) + kinetic_energy + magnetic_pressure
    
    # Mass flux: rho * v_dir
    mass_flux = rho * v_dir
    
    # Momentum flux: 
    # F_rho_v[i] = rho * v_dir * v[i] + delta[i,direction] * total_pressure - B[direction] * B[i]
    # Pre-allocate the momentum flux array
    momentum_flux = np.empty((len(v),) + rho.shape, dtype=np.float64)
    
    for i in range(len(v)):
        # Normal pressure term only applies in the direction of the flux
        if i == direction:
            pressure_term = total_pressure
        else:
            pressure_term = np.zeros_like(rho)
        
        # Magnetic tension term: -B[direction] * B[i]
        magnetic_term = -B[direction] * B[i]
        momentum_flux[i] = rho * v_dir * v[i] + pressure_term + magnetic_term
    
    # Energy flux: v_dir * (total_energy + total_pressure) - B[direction] * (v· B)
    # Calculate v·B using numpy operations
    v_dot_B = np.zeros_like(rho)
    for i in range(len(v)):
        v_dot_B = v_dot_B + v[i] * B[i]
    
    energy_flux = v_dir * (total_energy + total_pressure) - B[direction] * v_dot_B
    
    # Magnetic field flux: 
    # F_B[i] = v_dir * B[i] - B_dir * v[i] for i ≠ direction
    # F_B[direction] = 0 (induction equation preserves div(B) = 0)
    # Pre-allocate the magnetic flux array
    magnetic_flux = np.empty((len(B),) + rho.shape, dtype=np.float64)
    
    for i in range(len(B)):
        if i == direction:
            magnetic_flux[i] = np.zeros_like(rho)  # Preserve div(B) = 0
        else:
            magnetic_flux[i] = v_dir * B[i] - v[i] * B[direction]
    
    return mass_flux, momentum_flux, energy_flux, magnetic_flux

def compute_mhd_flux(rho, v, p, B, gamma, direction):
    """
    Compute the ideal MHD flux in the specified direction.
    
    This computes the continuous (not the numerical) flux for the MHD
    equations in conservative form.
    
    Args:
        rho: Density
        v: List of velocity components [vx, vy, vz]
        p: Pressure
        B: List of magnetic field components [Bx, By, Bz]
        gamma: Adiabatic index
        direction: Direction in which to compute the flux (0, 1, or 2)
        
    Returns:
        Dictionary of flux components
    """
    # Handle scalar inputs by converting to arrays
    is_scalar = isinstance(rho, (int, float))
    if is_scalar:
        rho_arr = np.array([rho])
        p_arr = np.array([p])
        v_arr = [np.array([v_i]) for v_i in v]
        B_arr = [np.array([B_i]) for B_i in B]
        
        # Call the implementation
        mass_flux, momentum_flux, energy_flux, magnetic_flux = _compute_mhd_flux_impl(
            rho_arr, v_arr, p_arr, B_arr, gamma, direction
        )
        
        # Convert back to scalar
        return {
            'density': mass_flux.item(),
            'momentum': [m_flux.item() for m_flux in momentum_flux],
            'energy': energy_flux.item(),
            'magnetic_field': [b_flux.item() for b_flux in magnetic_flux]
        }
    else:
        # Array case - call implementation directly
        mass_flux, momentum_flux, energy_flux, magnetic_flux = _compute_mhd_flux_impl(
            rho, v, p, B, gamma, direction
        )
        
        # Return dictionary with array results
    return {
        'density': mass_flux,
        'momentum': momentum_flux,
        'energy': energy_flux,
        'magnetic_field': magnetic_flux
    }

@njit
def _compute_max_wave_speeds_impl(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Numba-optimized implementation of maximum wave speed computation.
    This version always works with arrays.
    """
    # Sound speeds
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Alfven speeds - using numpy operations instead of accumulation
    B_L_squared = np.zeros_like(rho_L)
    for i in range(len(B_L)):
        B_L_squared = B_L_squared + B_L[i]**2
        
    B_R_squared = np.zeros_like(rho_R)
    for i in range(len(B_R)):
        B_R_squared = B_R_squared + B_R[i]**2
    
    c_A_L = np.sqrt(B_L_squared / rho_L)
    c_A_R = np.sqrt(B_R_squared / rho_R)
    
    # Fast magnetosonic speeds (upper bound: c_f ≤ c_s + c_A)
    c_f_L = c_L + c_A_L
    c_f_R = c_R + c_A_R
    
    # Minimum and maximum wave speeds
    s_L = np.minimum(v_L[direction] - c_f_L, v_R[direction] - c_f_R)
    s_R = np.maximum(v_L[direction] + c_f_L, v_R[direction] + c_f_R)
    
    return s_L, s_R

def compute_max_wave_speeds(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Compute the maximum wave speeds for the MHD Riemann problem.
    
    For MHD, this computes the fast magnetosonic wave speeds in the
    left and right states, which bound the maximum signal velocity.
    
    Args:
        rho_L, rho_R: Density in left and right states
        v_L, v_R: Velocity components in left and right states
        p_L, p_R: Pressure in left and right states
        B_L, B_R: Magnetic field components in left and right states
        gamma: Adiabatic index
        direction: Direction of the 1D Riemann problem
        
    Returns:
        Tuple (s_L, s_R) with minimum and maximum wave speeds
    """
    # Handle scalar inputs by converting to arrays
    is_scalar = isinstance(rho_L, (int, float))
    if is_scalar:
        rho_L_arr = np.array([rho_L])
        rho_R_arr = np.array([rho_R])
        p_L_arr = np.array([p_L])
        p_R_arr = np.array([p_R])
        v_L_arr = [np.array([v_i]) for v_i in v_L]
        v_R_arr = [np.array([v_i]) for v_i in v_R]
        B_L_arr = [np.array([B_i]) for B_i in B_L]
        B_R_arr = [np.array([B_i]) for B_i in B_R]
        
        # Call the implementation
        s_L, s_R = _compute_max_wave_speeds_impl(
            rho_L_arr, v_L_arr, p_L_arr, B_L_arr, 
            rho_R_arr, v_R_arr, p_R_arr, B_R_arr, 
            gamma, direction
        )
        
        # Convert back to scalar
        return s_L.item(), s_R.item()
    else:
        # Array case - call implementation directly
        return _compute_max_wave_speeds_impl(
            rho_L, v_L, p_L, B_L, 
            rho_R, v_R, p_R, B_R, 
            gamma, direction
        )

@njit
def _hll_flux_impl(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Numba-optimized implementation of the HLL flux calculation.
    This version always works with arrays.
    """
    # Compute physical fluxes in left and right states
    mass_flux_L, momentum_flux_L, energy_flux_L, magnetic_flux_L = _compute_mhd_flux_impl(
        rho_L, v_L, p_L, B_L, gamma, direction
    )
    
    mass_flux_R, momentum_flux_R, energy_flux_R, magnetic_flux_R = _compute_mhd_flux_impl(
        rho_R, v_R, p_R, B_R, gamma, direction
    )
    
    # Compute wave speeds
    s_L, s_R = _compute_max_wave_speeds_impl(
        rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction
    )
    
    # Conserved variables in left state
    # Pre-allocate and compute momentum
    momentum_L = np.empty((len(v_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(v_L)):
        momentum_L[i] = rho_L * v_L[i]
    
    # Energy for left state
    # Using numpy operations instead of accumulation
    v_L_squared_sum = np.zeros_like(rho_L)
    for i in range(len(v_L)):
        v_L_squared_sum = v_L_squared_sum + v_L[i]**2
        
    # Calculate B_L_squared using numpy operations
    B_L_squared_sum = np.zeros_like(rho_L)
    for i in range(len(B_L)):
        B_L_squared_sum = B_L_squared_sum + B_L[i]**2
        
    energy_L = p_L / (gamma - 1) + 0.5 * rho_L * v_L_squared_sum + 0.5 * B_L_squared_sum
    
    # Convert B_L to numpy array
    B_L_array = np.empty((len(B_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(B_L)):
        B_L_array[i] = B_L[i]
    
    # Conserved variables in right state
    # Pre-allocate and compute momentum
    momentum_R = np.empty((len(v_R),) + rho_R.shape, dtype=np.float64)
    for i in range(len(v_R)):
        momentum_R[i] = rho_R * v_R[i]
    
    # Energy for right state
    # Using numpy operations instead of accumulation
    v_R_squared_sum = np.zeros_like(rho_R)
    for i in range(len(v_R)):
        v_R_squared_sum = v_R_squared_sum + v_R[i]**2
        
    # Calculate B_R_squared using numpy operations
    B_R_squared_sum = np.zeros_like(rho_R)
    for i in range(len(B_R)):
        B_R_squared_sum = B_R_squared_sum + B_R[i]**2
        
    energy_R = p_R / (gamma - 1) + 0.5 * rho_R * v_R_squared_sum + 0.5 * B_R_squared_sum
    
    # Convert B_R to numpy array
    B_R_array = np.empty((len(B_R),) + rho_R.shape, dtype=np.float64)
    for i in range(len(B_R)):
        B_R_array[i] = B_R[i]
    
    # Since s_L and s_R are potentially arrays, we need to handle the flux calculation element-wise
    # Create a mask for each case
    case_left = s_L >= 0
    case_right = s_R <= 0
    case_middle = ~(case_left | case_right)
    
    # Initialize flux arrays
    hll_mass_flux = np.zeros_like(rho_L)
    hll_energy_flux = np.zeros_like(rho_L)
    momentum_flux = np.empty((len(v_L),) + rho_L.shape, dtype=np.float64)
    magnetic_flux = np.empty((len(B_L),) + rho_L.shape, dtype=np.float64)
    
    # Where s_L >= 0, use left flux
    if np.any(case_left):
        hll_mass_flux = np.where(case_left, mass_flux_L, hll_mass_flux)
        hll_energy_flux = np.where(case_left, energy_flux_L, hll_energy_flux)
    
    # Where s_R <= 0, use right flux
    if np.any(case_right):
        hll_mass_flux = np.where(case_right, mass_flux_R, hll_mass_flux)
        hll_energy_flux = np.where(case_right, energy_flux_R, hll_energy_flux)
    
    # For the middle case, use HLL formula
    if np.any(case_middle):
        factor = 1.0 / (s_R - s_L)
        middle_flux = factor * (s_R * mass_flux_L - s_L * mass_flux_R 
                           + s_L * s_R * (rho_R - rho_L))
        hll_mass_flux = np.where(case_middle, middle_flux, hll_mass_flux)
        
        middle_flux = factor * (s_R * energy_flux_L - s_L * energy_flux_R 
                           + s_L * s_R * (energy_R - energy_L))
        hll_energy_flux = np.where(case_middle, middle_flux, hll_energy_flux)
    
    # Process momentum fluxes
    for i in range(len(v_L)):
        momentum_flux[i] = np.zeros_like(rho_L)
        
        # Where s_L >= 0, use left flux
        if np.any(case_left):
            momentum_flux[i] = np.where(case_left, momentum_flux_L[i], momentum_flux[i])
        
        # Where s_R <= 0, use right flux
        if np.any(case_right):
            momentum_flux[i] = np.where(case_right, momentum_flux_R[i], momentum_flux[i])
        
        # For the middle case, use HLL formula
        if np.any(case_middle):
            factor = 1.0 / (s_R - s_L)
            middle_flux = factor * (s_R * momentum_flux_L[i] - s_L * momentum_flux_R[i] 
                             + s_L * s_R * (momentum_R[i] - momentum_L[i]))
            momentum_flux[i] = np.where(case_middle, middle_flux, momentum_flux[i])
    
    # Process magnetic field fluxes
    for i in range(len(B_L)):
        magnetic_flux[i] = np.zeros_like(rho_L)
        
        # Where s_L >= 0, use left flux
        if np.any(case_left):
            magnetic_flux[i] = np.where(case_left, magnetic_flux_L[i], magnetic_flux[i])
        
        # Where s_R <= 0, use right flux
        if np.any(case_right):
            magnetic_flux[i] = np.where(case_right, magnetic_flux_R[i], magnetic_flux[i])
        
        # For the middle case, use HLL formula
        if np.any(case_middle):
            factor = 1.0 / (s_R - s_L)
            middle_flux = factor * (s_R * magnetic_flux_L[i] - s_L * magnetic_flux_R[i] 
                             + s_L * s_R * (B_R_array[i] - B_L_array[i]))
            magnetic_flux[i] = np.where(case_middle, middle_flux, magnetic_flux[i])
    
    return hll_mass_flux, momentum_flux, hll_energy_flux, magnetic_flux

def hll_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Compute the HLL (Harten-Lax-van Leer) flux for MHD.
    
    HLL improves upon the Rusanov flux by using different wave speeds
    for the left and right-going waves, rather than a single maximum speed.
    
    Args:
        rho_L, rho_R: Density in left and right states
        v_L, v_R: Velocity components in left and right states
        p_L, p_R: Pressure in left and right states
        B_L, B_R: Magnetic field components in left and right states
        gamma: Adiabatic index
        direction: Direction of the 1D Riemann problem
        
    Returns:
        Dictionary with the numerical fluxes
    """
    # Handle scalar inputs by converting to arrays
    is_scalar = isinstance(rho_L, (int, float))
    if is_scalar:
        rho_L_arr = np.array([rho_L])
        rho_R_arr = np.array([rho_R])
        p_L_arr = np.array([p_L])
        p_R_arr = np.array([p_R])
        v_L_arr = [np.array([v_i]) for v_i in v_L]
        v_R_arr = [np.array([v_i]) for v_i in v_R]
        B_L_arr = [np.array([B_i]) for B_i in B_L]
        B_R_arr = [np.array([B_i]) for B_i in B_R]
        
        # Call the implementation
        mass_flux, momentum_flux, energy_flux, magnetic_flux = _hll_flux_impl(
            rho_L_arr, v_L_arr, p_L_arr, B_L_arr, 
            rho_R_arr, v_R_arr, p_R_arr, B_R_arr, 
            gamma, direction
        )
        
        # Convert back to scalar
        return {
            'density': mass_flux.item(),
            'momentum': [m_flux.item() for m_flux in momentum_flux],
            'energy': energy_flux.item(),
            'magnetic_field': [b_flux.item() for b_flux in magnetic_flux]
        }
    else:
        # Array case - call implementation directly
        mass_flux, momentum_flux, energy_flux, magnetic_flux = _hll_flux_impl(
            rho_L, v_L, p_L, B_L, 
            rho_R, v_R, p_R, B_R, 
            gamma, direction
        )
        
        # Return dictionary with array results
        return {
            'density': mass_flux,
            'momentum': momentum_flux,
            'energy': energy_flux,
            'magnetic_field': magnetic_flux
        }

@njit
def rusanov_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Compute the Rusanov (local Lax-Friedrichs) flux for MHD.
    
    The Rusanov flux is a simple, robust approximate Riemann solver.
    It uses the largest wave speed to add numerical dissipation.
    
    Args:
        rho_L, rho_R: Density in left and right states
        v_L, v_R: Velocity components in left and right states
        p_L, p_R: Pressure in left and right states
        B_L, B_R: Magnetic field components in left and right states
        gamma: Adiabatic index
        direction: Direction of the 1D Riemann problem
        
    Returns:
        Dictionary with the numerical fluxes
    """
    # Compute physical fluxes in left and right states
    flux_L = compute_mhd_flux(rho_L, v_L, p_L, B_L, gamma, direction)
    flux_R = compute_mhd_flux(rho_R, v_R, p_R, B_R, gamma, direction)
    
    # Compute maximum wave speed
    s_L, s_R = compute_max_wave_speeds(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction)
    max_speed = np.maximum(np.abs(s_L), np.abs(s_R))
    
    # Conserved variables in left state
    cons_L = {}
    cons_L['density'] = rho_L
    
    # Pre-allocate and compute momentum
    momentum_L = np.empty((len(v_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(v_L)):
        momentum_L[i] = rho_L * v_L[i]
    cons_L['momentum'] = momentum_L
    
    # Energy for left state
    # Using numpy operations instead of accumulation
    v_L_squared_sum = np.zeros_like(rho_L)
    for i in range(len(v_L)):
        v_L_squared_sum = v_L_squared_sum + v_L[i]**2
        
    # Calculate B_L_squared using numpy operations
    B_L_squared_sum = np.zeros_like(rho_L)
    for i in range(len(B_L)):
        B_L_squared_sum = B_L_squared_sum + B_L[i]**2
        
    cons_L['energy'] = p_L / (gamma - 1) + 0.5 * rho_L * v_L_squared_sum + 0.5 * B_L_squared_sum
    
    # Convert B_L to numpy array
    B_L_array = np.empty((len(B_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(B_L)):
        B_L_array[i] = B_L[i]
    cons_L['magnetic_field'] = B_L_array
    
    # Conserved variables in right state
    cons_R = {}
    cons_R['density'] = rho_R
    
    # Pre-allocate and compute momentum
    momentum_R = np.empty((len(v_R),) + rho_R.shape, dtype=np.float64)
    for i in range(len(v_R)):
        momentum_R[i] = rho_R * v_R[i]
    cons_R['momentum'] = momentum_R
    
    # Energy for right state
    # Using numpy operations instead of accumulation
    v_R_squared_sum = np.zeros_like(rho_R)
    for i in range(len(v_R)):
        v_R_squared_sum = v_R_squared_sum + v_R[i]**2
        
    # Calculate B_R_squared using numpy operations
    B_R_squared_sum = np.zeros_like(rho_R)
    for i in range(len(B_R)):
        B_R_squared_sum = B_R_squared_sum + B_R[i]**2
        
    cons_R['energy'] = p_R / (gamma - 1) + 0.5 * rho_R * v_R_squared_sum + 0.5 * B_R_squared_sum
    
    # Convert B_R to numpy array
    B_R_array = np.empty((len(B_R),) + rho_R.shape, dtype=np.float64)
    for i in range(len(B_R)):
        B_R_array[i] = B_R[i]
    cons_R['magnetic_field'] = B_R_array
    
    # Rusanov flux: F = 0.5 * (F_L + F_R - |max_speed| * (U_R - U_L))
    rusanov_flux = {}
    
    # Density flux
    rusanov_flux['density'] = 0.5 * (flux_L['density'] + flux_R['density'] - max_speed * (cons_R['density'] - cons_L['density']))
    
    # Momentum flux - pre-allocate the array
    momentum_flux = np.empty((len(v_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(v_L)):
        momentum_flux[i] = 0.5 * (flux_L['momentum'][i] + flux_R['momentum'][i] - max_speed * (cons_R['momentum'][i] - cons_L['momentum'][i]))
    rusanov_flux['momentum'] = momentum_flux
    
    # Energy flux
    rusanov_flux['energy'] = 0.5 * (flux_L['energy'] + flux_R['energy'] - max_speed * (cons_R['energy'] - cons_L['energy']))
    
    # Magnetic field flux - pre-allocate the array
    magnetic_flux = np.empty((len(B_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(B_L)):
        magnetic_flux[i] = 0.5 * (flux_L['magnetic_field'][i] + flux_R['magnetic_field'][i] - max_speed * (cons_R['magnetic_field'][i] - cons_L['magnetic_field'][i]))
    rusanov_flux['magnetic_field'] = magnetic_flux
    
    return rusanov_flux

@njit
def hlld_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Compute the HLLD (Harten-Lax-van Leer-Discontinuities) flux for MHD.
    
    HLLD is an extension of HLL that resolves intermediate states,
    improving the resolution of Alfvén and contact waves.
    
    This is a simplified HLLD solver - a full implementation would be more complex.
    
    Args:
        rho_L, rho_R: Density in left and right states
        v_L, v_R: Velocity components in left and right states
        p_L, p_R: Pressure in left and right states
        B_L, B_R: Magnetic field components in left and right states
        gamma: Adiabatic index
        direction: Direction of the 1D Riemann problem
        
    Returns:
        Dictionary with the numerical fluxes
    """
    # For this demo, we'll defer to the HLL solver
    # A complete HLLD solver would resolve the contact and Alfvén waves
    # by constructing intermediate states
    return hll_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction)

def evolve_mhd(mhd_system, solver='hll', steps=1):
    """
    Evolve an MHD system using the specified solver.
    
    This function advances the MHD system by the specified number of time steps
    using the chosen Riemann solver for flux computation.
    
    Args:
        mhd_system: MHD system object to evolve
        solver: Name of the Riemann solver to use ('rusanov', 'hll', or 'hlld')
        steps: Number of time steps to evolve
        
    Returns:
        Updated MHD system
    """
    # Try to use Numba-compiled versions first
    try:
        # Select the appropriate flux function
        if solver.lower() == 'rusanov':
            flux_function = rusanov_flux
        elif solver.lower() == 'hll':
            flux_function = hll_flux
        elif solver.lower() == 'hlld':
            flux_function = hlld_flux
        else:
            raise ValueError(f"Unknown solver: {solver}")
            
        # Try a test call to see if Numba version works
        shape = (4, 4)
        rho_test = np.ones(shape)
        v_test = [np.ones(shape) * 0.1, np.ones(shape) * 0.2]
        p_test = np.ones(shape)
        B_test = [np.ones(shape) * 0.5, np.ones(shape) * 0.2]
        flux_function(rho_test, v_test, p_test, B_test, rho_test, v_test, p_test, B_test, 5/3, 0)
        
        # If test call succeeds, use Numba version
        print(f"Using Numba-compiled {solver} flux function")
        using_numba = True
    except Exception as e:
        # If Numba version fails, fall back to Python version
        print(f"Numba compilation failed: {str(e)}. Falling back to Python version.")
        if solver.lower() == 'rusanov':
            flux_function = py_rusanov_flux
        elif solver.lower() == 'hll':
            flux_function = py_hll_flux
        elif solver.lower() == 'hlld':
            flux_function = py_hlld_flux
        else:
            raise ValueError(f"Unknown solver: {solver}")
            
        # Patch MHD system's compute_all_fluxes method to use our Python functions
        using_numba = False
        
        # Store the original function if it exists
        if hasattr(mhd_system, '_original_compute_numerical_flux'):
            pass  # Already saved
        else:
            mhd_system._original_compute_numerical_flux = mhd_system.compute_numerical_flux
            
            # Define a patched method that uses our Python functions
            def patched_compute_numerical_flux(self, U, direction, time_step=None):
                """
                Patched version of compute_numerical_flux that uses Python-only functions.
                """
                from myproject.utils.mhd.solvers import py_hll_flux
                
                # Extract state
                rho = self.density
                v = self.velocity
                p = self.pressure
                B = self.magnetic_field
                
                # Prepare left and right states (simplified)
                nx, ny = rho.shape
                
                # Use periodic boundary conditions (simplified)
                rho_L = rho
                v_L = v
                p_L = p
                B_L = B
                
                # Shifted indices for right state
                shift = [0] * self.dimension
                shift[direction] = 1
                
                # Right state (with periodic boundary assumption)
                rho_R = np.roll(rho, shift, axis=direction)
                v_R = [np.roll(v_comp, shift, axis=direction) for v_comp in v]
                p_R = np.roll(p, shift, axis=direction)
                B_R = [np.roll(B_comp, shift, axis=direction) for B_comp in B]
                
                # Call the flux function
                flux = py_hll_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, self.gamma, direction)
                
                return flux
                
            # Replace the method with our patched version
            mhd_system.compute_numerical_flux = patched_compute_numerical_flux.__get__(mhd_system, type(mhd_system))
    
    # Evolve the system
    for _ in range(steps):
        # You would typically register the solver with the MHD system
        # and then call its advance_time_step method
        try:
            mhd_system.advance_time_step()
        except Exception as e:
            if hasattr(mhd_system, 'logger'):
                mhd_system.logger.error(f"Error during time step evolution: {str(e)}")
            else:
                print(f"Error during time step evolution: {str(e)}")
                
            # If using Numba version failed, try falling back to Python version
            if using_numba:
                print("Falling back to Python implementation for this time step.")
                if not hasattr(mhd_system, '_original_compute_numerical_flux'):
                    # Save original method
                    mhd_system._original_compute_numerical_flux = mhd_system.compute_numerical_flux
                    
                    # Define a patched method that uses our Python functions
                    def patched_compute_numerical_flux(self, U, direction, time_step=None):
                        """
                        Patched version of compute_numerical_flux that uses Python-only functions.
                        """
                        from myproject.utils.mhd.solvers import py_hll_flux
                        
                        # Extract state
                        rho = self.density
                        v = self.velocity
                        p = self.pressure
                        B = self.magnetic_field
                        
                        # Prepare left and right states (simplified)
                        nx, ny = rho.shape
                        
                        # Use periodic boundary conditions (simplified)
                        rho_L = rho
                        v_L = v
                        p_L = p
                        B_L = B
                        
                        # Shifted indices for right state
                        shift = [0] * self.dimension
                        shift[direction] = 1
                        
                        # Right state (with periodic boundary assumption)
                        rho_R = np.roll(rho, shift, axis=direction)
                        v_R = [np.roll(v_comp, shift, axis=direction) for v_comp in v]
                        p_R = np.roll(p, shift, axis=direction)
                        B_R = [np.roll(B_comp, shift, axis=direction) for B_comp in B]
                        
                        # Call the flux function
                        flux = py_hll_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, self.gamma, direction)
                        
                        return flux
                        
                    # Replace the method with our patched version
                    mhd_system.compute_numerical_flux = patched_compute_numerical_flux.__get__(mhd_system, type(mhd_system))
                
                # Try again with Python version
                try:
                    mhd_system.advance_time_step()
                except Exception as e2:
                    if hasattr(mhd_system, 'logger'):
                        mhd_system.logger.error(f"Error during time step evolution (Python fallback): {str(e2)}")
                    else:
                        print(f"Error during time step evolution (Python fallback): {str(e2)}")
                    raise
            else:
                raise
    
    # Restore original method if we replaced it
    if hasattr(mhd_system, '_original_compute_numerical_flux'):
        mhd_system.compute_numerical_flux = mhd_system._original_compute_numerical_flux
        delattr(mhd_system, '_original_compute_numerical_flux')
    
    return mhd_system 

# Add Python-only versions that don't rely on Numba

def py_compute_mhd_flux(rho, v, p, B, gamma, direction):
    """
    Python-only version of compute_mhd_flux for fallback when Numba fails.
    """
    # Handle scalar inputs - convert to numpy arrays
    is_scalar = isinstance(rho, (int, float))
    if is_scalar:
        rho = np.array([rho])
        p = np.array([p])
        v = [np.array([v_i]) for v_i in v]
        B = [np.array([B_i]) for B_i in B]
    
    # Get velocity component in the flux direction
    v_dir = v[direction]
    
    # Compute B² using numpy operations instead of accumulation
    B_squared = np.zeros_like(rho)
    for i in range(len(B)):
        B_squared = B_squared + B[i]**2
    
    # Compute kinetic energy using numpy operations
    kinetic_energy = np.zeros_like(rho)
    for i in range(len(v)):
        kinetic_energy = kinetic_energy + 0.5 * rho * v[i]**2
    
    # Compute magnetic pressure
    magnetic_pressure = 0.5 * B_squared
    
    # Compute total pressure
    total_pressure = p + magnetic_pressure
    
    # Compute total energy
    total_energy = p / (gamma - 1) + kinetic_energy + magnetic_pressure
    
    # Mass flux: rho * v_dir
    mass_flux = rho * v_dir
    
    # Momentum flux: 
    # F_rho_v[i] = rho * v_dir * v[i] + delta[i,direction] * total_pressure - B[direction] * B[i]
    # Pre-allocate the momentum flux array
    momentum_flux = np.empty((len(v),) + rho.shape, dtype=np.float64)
    
    for i in range(len(v)):
        # Normal pressure term only applies in the direction of the flux
        if i == direction:
            pressure_term = total_pressure
        else:
            pressure_term = np.zeros_like(rho)
        
        # Magnetic tension term: -B[direction] * B[i]
        magnetic_term = -B[direction] * B[i]
        momentum_flux[i] = rho * v_dir * v[i] + pressure_term + magnetic_term
    
    # Energy flux: v_dir * (total_energy + total_pressure) - B[direction] * (v· B)
    # Calculate v·B using numpy operations
    v_dot_B = np.zeros_like(rho)
    for i in range(len(v)):
        v_dot_B = v_dot_B + v[i] * B[i]
    
    energy_flux = v_dir * (total_energy + total_pressure) - B[direction] * v_dot_B
    
    # Magnetic field flux: 
    # F_B[i] = v_dir * B[i] - B_dir * v[i] for i ≠ direction
    # F_B[direction] = 0 (induction equation preserves div(B) = 0)
    # Pre-allocate the magnetic flux array
    magnetic_flux = np.empty((len(B),) + rho.shape, dtype=np.float64)
    
    for i in range(len(B)):
        if i == direction:
            magnetic_flux[i] = np.zeros_like(rho)  # Preserve div(B) = 0
        else:
            magnetic_flux[i] = v_dir * B[i] - v[i] * B[direction]
    
    # Convert back to scalar if input was scalar
    if is_scalar:
        mass_flux = mass_flux.item()
        energy_flux = energy_flux.item()
        momentum_flux = [m_flux.item() for m_flux in momentum_flux]
        magnetic_flux = [b_flux.item() for b_flux in magnetic_flux]
    
    return {
        'density': mass_flux,
        'momentum': momentum_flux,
        'energy': energy_flux,
        'magnetic_field': magnetic_flux
    }

def py_compute_max_wave_speeds(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Python-only version of compute_max_wave_speeds for fallback when Numba fails.
    """
    # Handle scalar inputs - convert to numpy arrays
    is_scalar = isinstance(rho_L, (int, float))
    if is_scalar:
        rho_L = np.array([rho_L])
        rho_R = np.array([rho_R])
        p_L = np.array([p_L])
        p_R = np.array([p_R])
        v_L = [np.array([v_i]) for v_i in v_L]
        v_R = [np.array([v_i]) for v_i in v_R]
        B_L = [np.array([B_i]) for B_i in B_L]
        B_R = [np.array([B_i]) for B_i in B_R]
    
    # Sound speeds
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Alfven speeds - using numpy operations instead of accumulation
    B_L_squared = np.zeros_like(rho_L)
    for i in range(len(B_L)):
        B_L_squared = B_L_squared + B_L[i]**2
        
    B_R_squared = np.zeros_like(rho_R)
    for i in range(len(B_R)):
        B_R_squared = B_R_squared + B_R[i]**2
    
    c_A_L = np.sqrt(B_L_squared / rho_L)
    c_A_R = np.sqrt(B_R_squared / rho_R)
    
    # Fast magnetosonic speeds (upper bound: c_f ≤ c_s + c_A)
    c_f_L = c_L + c_A_L
    c_f_R = c_R + c_A_R
    
    # Minimum and maximum wave speeds
    s_L = np.minimum(v_L[direction] - c_f_L, v_R[direction] - c_f_R)
    s_R = np.maximum(v_L[direction] + c_f_L, v_R[direction] + c_f_R)
    
    # Convert back to scalar if input was scalar
    if is_scalar:
        s_L = s_L.item()
        s_R = s_R.item()
    
    return s_L, s_R

def py_hll_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Python-only version of hll_flux for fallback when Numba fails.
    """
    # Handle scalar inputs - convert to numpy arrays
    is_scalar = isinstance(rho_L, (int, float))
    if is_scalar:
        rho_L = np.array([rho_L])
        rho_R = np.array([rho_R])
        p_L = np.array([p_L])
        p_R = np.array([p_R])
        v_L = [np.array([v_i]) for v_i in v_L]
        v_R = [np.array([v_i]) for v_i in v_R]
        B_L = [np.array([B_i]) for B_i in B_L]
        B_R = [np.array([B_i]) for B_i in B_R]
    
    # Compute physical fluxes in left and right states
    flux_L = py_compute_mhd_flux(rho_L, v_L, p_L, B_L, gamma, direction)
    flux_R = py_compute_mhd_flux(rho_R, v_R, p_R, B_R, gamma, direction)
    
    # Compute wave speeds
    s_L, s_R = py_compute_max_wave_speeds(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction)
    
    # Conserved variables in left state
    cons_L = {}
    cons_L['density'] = rho_L
    
    # Pre-allocate and compute momentum
    momentum_L = np.empty((len(v_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(v_L)):
        momentum_L[i] = rho_L * v_L[i]
    cons_L['momentum'] = momentum_L
    
    # Energy for left state
    # Using numpy operations instead of accumulation
    v_L_squared_sum = np.zeros_like(rho_L)
    for i in range(len(v_L)):
        v_L_squared_sum = v_L_squared_sum + v_L[i]**2
        
    # Calculate B_L_squared using numpy operations
    B_L_squared_sum = np.zeros_like(rho_L)
    for i in range(len(B_L)):
        B_L_squared_sum = B_L_squared_sum + B_L[i]**2
        
    cons_L['energy'] = p_L / (gamma - 1) + 0.5 * rho_L * v_L_squared_sum + 0.5 * B_L_squared_sum
    
    # Convert B_L to numpy array
    B_L_array = np.empty((len(B_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(B_L)):
        B_L_array[i] = B_L[i]
    cons_L['magnetic_field'] = B_L_array
    
    # Conserved variables in right state
    cons_R = {}
    cons_R['density'] = rho_R
    
    # Pre-allocate and compute momentum
    momentum_R = np.empty((len(v_R),) + rho_R.shape, dtype=np.float64)
    for i in range(len(v_R)):
        momentum_R[i] = rho_R * v_R[i]
    cons_R['momentum'] = momentum_R
    
    # Energy for right state
    # Using numpy operations instead of accumulation
    v_R_squared_sum = np.zeros_like(rho_R)
    for i in range(len(v_R)):
        v_R_squared_sum = v_R_squared_sum + v_R[i]**2
        
    # Calculate B_R_squared using numpy operations
    B_R_squared_sum = np.zeros_like(rho_R)
    for i in range(len(B_R)):
        B_R_squared_sum = B_R_squared_sum + B_R[i]**2
        
    cons_R['energy'] = p_R / (gamma - 1) + 0.5 * rho_R * v_R_squared_sum + 0.5 * B_R_squared_sum
    
    # Convert B_R to numpy array
    B_R_array = np.empty((len(B_R),) + rho_R.shape, dtype=np.float64)
    for i in range(len(B_R)):
        B_R_array[i] = B_R[i]
    cons_R['magnetic_field'] = B_R_array
    
    # Since s_L and s_R are potentially arrays, we need to handle the flux calculation element-wise
    # Create a mask for each case
    case_left = s_L >= 0
    case_right = s_R <= 0
    case_middle = ~(case_left | case_right)
    
    # Initialize flux dictionary
    hll_flux = {}
    
    # Handle density flux
    hll_flux['density'] = np.zeros_like(rho_L)
    
    # Where s_L >= 0, use left flux
    if np.any(case_left):
        hll_flux['density'] = np.where(case_left, flux_L['density'], hll_flux['density'])
    
    # Where s_R <= 0, use right flux
    if np.any(case_right):
        hll_flux['density'] = np.where(case_right, flux_R['density'], hll_flux['density'])
    
    # For the middle case, use HLL formula
    if np.any(case_middle):
        factor = np.where(case_middle, 1.0 / (s_R - s_L), 0.0)
        middle_flux = factor * (s_R * flux_L['density'] - s_L * flux_R['density'] 
                           + s_L * s_R * (cons_R['density'] - cons_L['density']))
        hll_flux['density'] = np.where(case_middle, middle_flux, hll_flux['density'])
    
    # Initialize momentum and magnetic field fluxes
    momentum_flux = np.empty((len(v_L),) + rho_L.shape, dtype=np.float64)
    magnetic_flux = np.empty((len(B_L),) + rho_L.shape, dtype=np.float64)
    
    # Process momentum fluxes
    for i in range(len(v_L)):
        momentum_flux[i] = np.zeros_like(rho_L)
        
        # Where s_L >= 0, use left flux
        if np.any(case_left):
            momentum_flux[i] = np.where(case_left, flux_L['momentum'][i], momentum_flux[i])
        
        # Where s_R <= 0, use right flux
        if np.any(case_right):
            momentum_flux[i] = np.where(case_right, flux_R['momentum'][i], momentum_flux[i])
        
        # For the middle case, use HLL formula
        if np.any(case_middle):
            factor = 1.0 / (s_R - s_L)
            middle_flux = factor * (s_R * flux_L['momentum'][i] - s_L * flux_R['momentum'][i] 
                             + s_L * s_R * (cons_R['momentum'][i] - cons_L['momentum'][i]))
            momentum_flux[i] = np.where(case_middle, middle_flux, momentum_flux[i])
    
    hll_flux['momentum'] = momentum_flux
    
    # Process energy flux
    hll_flux['energy'] = np.zeros_like(rho_L)
    
    # Where s_L >= 0, use left flux
    if np.any(case_left):
        hll_flux['energy'] = np.where(case_left, flux_L['energy'], hll_flux['energy'])
    
    # Where s_R <= 0, use right flux
    if np.any(case_right):
        hll_flux['energy'] = np.where(case_right, flux_R['energy'], hll_flux['energy'])
    
    # For the middle case, use HLL formula
    if np.any(case_middle):
        factor = 1.0 / (s_R - s_L)
        middle_flux = factor * (s_R * flux_L['energy'] - s_L * flux_R['energy'] 
                           + s_L * s_R * (cons_R['energy'] - cons_L['energy']))
        hll_flux['energy'] = np.where(case_middle, middle_flux, hll_flux['energy'])
    
    # Process magnetic field fluxes
    for i in range(len(B_L)):
        magnetic_flux[i] = np.zeros_like(rho_L)
        
        # Where s_L >= 0, use left flux
        if np.any(case_left):
            magnetic_flux[i] = np.where(case_left, flux_L['magnetic_field'][i], magnetic_flux[i])
        
        # Where s_R <= 0, use right flux
        if np.any(case_right):
            magnetic_flux[i] = np.where(case_right, flux_R['magnetic_field'][i], magnetic_flux[i])
        
        # For the middle case, use HLL formula
        if np.any(case_middle):
            factor = 1.0 / (s_R - s_L)
            middle_flux = factor * (s_R * flux_L['magnetic_field'][i] - s_L * flux_R['magnetic_field'][i] 
                             + s_L * s_R * (cons_R['magnetic_field'][i] - cons_L['magnetic_field'][i]))
            magnetic_flux[i] = np.where(case_middle, middle_flux, magnetic_flux[i])
    
    hll_flux['magnetic_field'] = magnetic_flux
    
    # Convert back to scalar if input was scalar
    if is_scalar:
        hll_flux['density'] = hll_flux['density'].item()
        hll_flux['energy'] = hll_flux['energy'].item()
        hll_flux['momentum'] = [m_flux.item() for m_flux in hll_flux['momentum']]
        hll_flux['magnetic_field'] = [b_flux.item() for b_flux in hll_flux['magnetic_field']]
    
    return hll_flux

def py_rusanov_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Python-only version of rusanov_flux for fallback when Numba fails.
    """
    # This function follows the same logic as hll_flux but uses simpler wave speed calculation
    # Compute physical fluxes in left and right states
    flux_L = py_compute_mhd_flux(rho_L, v_L, p_L, B_L, gamma, direction)
    flux_R = py_compute_mhd_flux(rho_R, v_R, p_R, B_R, gamma, direction)
    
    # Compute wave speeds
    s_L, s_R = py_compute_max_wave_speeds(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction)
    max_speed = np.maximum(np.abs(s_L), np.abs(s_R))
    
    # Conserved variables in left state
    cons_L = {}
    cons_L['density'] = rho_L
    
    # Pre-allocate and compute momentum
    momentum_L = np.empty((len(v_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(v_L)):
        momentum_L[i] = rho_L * v_L[i]
    cons_L['momentum'] = momentum_L
    
    # Energy for left state
    # Using numpy operations instead of accumulation
    v_L_squared_sum = np.zeros_like(rho_L)
    for i in range(len(v_L)):
        v_L_squared_sum = v_L_squared_sum + v_L[i]**2
        
    # Calculate B_L_squared using numpy operations
    B_L_squared_sum = np.zeros_like(rho_L)
    for i in range(len(B_L)):
        B_L_squared_sum = B_L_squared_sum + B_L[i]**2
        
    cons_L['energy'] = p_L / (gamma - 1) + 0.5 * rho_L * v_L_squared_sum + 0.5 * B_L_squared_sum
    
    # Convert B_L to numpy array
    B_L_array = np.empty((len(B_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(B_L)):
        B_L_array[i] = B_L[i]
    cons_L['magnetic_field'] = B_L_array
    
    # Conserved variables in right state
    cons_R = {}
    cons_R['density'] = rho_R
    
    # Pre-allocate and compute momentum
    momentum_R = np.empty((len(v_R),) + rho_R.shape, dtype=np.float64)
    for i in range(len(v_R)):
        momentum_R[i] = rho_R * v_R[i]
    cons_R['momentum'] = momentum_R
    
    # Energy for right state
    # Using numpy operations instead of accumulation
    v_R_squared_sum = np.zeros_like(rho_R)
    for i in range(len(v_R)):
        v_R_squared_sum = v_R_squared_sum + v_R[i]**2
        
    # Calculate B_R_squared using numpy operations
    B_R_squared_sum = np.zeros_like(rho_R)
    for i in range(len(B_R)):
        B_R_squared_sum = B_R_squared_sum + B_R[i]**2
        
    cons_R['energy'] = p_R / (gamma - 1) + 0.5 * rho_R * v_R_squared_sum + 0.5 * B_R_squared_sum
    
    # Convert B_R to numpy array
    B_R_array = np.empty((len(B_R),) + rho_R.shape, dtype=np.float64)
    for i in range(len(B_R)):
        B_R_array[i] = B_R[i]
    cons_R['magnetic_field'] = B_R_array
    
    # Rusanov flux: F = 0.5 * (F_L + F_R - |max_speed| * (U_R - U_L))
    rusanov_flux = {}
    
    # Density flux
    rusanov_flux['density'] = 0.5 * (flux_L['density'] + flux_R['density'] - max_speed * (cons_R['density'] - cons_L['density']))
    
    # Momentum flux - pre-allocate the array
    momentum_flux = np.empty((len(v_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(v_L)):
        momentum_flux[i] = 0.5 * (flux_L['momentum'][i] + flux_R['momentum'][i] - max_speed * (cons_R['momentum'][i] - cons_L['momentum'][i]))
    rusanov_flux['momentum'] = momentum_flux
    
    # Energy flux
    rusanov_flux['energy'] = 0.5 * (flux_L['energy'] + flux_R['energy'] - max_speed * (cons_R['energy'] - cons_L['energy']))
    
    # Magnetic field flux - pre-allocate the array
    magnetic_flux = np.empty((len(B_L),) + rho_L.shape, dtype=np.float64)
    for i in range(len(B_L)):
        magnetic_flux[i] = 0.5 * (flux_L['magnetic_field'][i] + flux_R['magnetic_field'][i] - max_speed * (cons_R['magnetic_field'][i] - cons_L['magnetic_field'][i]))
    rusanov_flux['magnetic_field'] = magnetic_flux
    
    return rusanov_flux

def py_hlld_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction):
    """
    Python-only version of hlld_flux for fallback when Numba fails.
    """
    # For this demo version, we just defer to the HLL solver
    return py_hll_flux(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction) 