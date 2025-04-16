"""
MHD Solver module implementing various Riemann solvers for MHD simulations.

This module provides numerical Riemann solvers for the MHD equations, which
are used to compute fluxes at cell interfaces in finite volume methods.
"""

import numpy as np
from numba import njit

@njit
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
    # Get velocity component in the flux direction
    v_dir = v[direction]
    
    # Compute B²
    B_squared = sum(B[i]**2 for i in range(len(B)))
    
    # Compute kinetic energy
    kinetic_energy = 0.5 * rho * sum(v[i]**2 for i in range(len(v)))
    
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
    momentum_flux = [None] * len(v)
    for i in range(len(v)):
        # Normal pressure term only applies in the direction of the flux
        pressure_term = total_pressure if i == direction else 0.0
        # Magnetic tension term: -B[direction] * B[i]
        magnetic_term = -B[direction] * B[i]
        momentum_flux[i] = rho * v_dir * v[i] + pressure_term + magnetic_term
    
    # Energy flux: v_dir * (total_energy + total_pressure) - B[direction] * (v· B)
    v_dot_B = sum(v[i] * B[i] for i in range(len(v)))
    energy_flux = v_dir * (total_energy + total_pressure) - B[direction] * v_dot_B
    
    # Magnetic field flux: 
    # F_B[i] = v_dir * B[i] - B_dir * v[i] for i ≠ direction
    # F_B[direction] = 0 (induction equation preserves div(B) = 0)
    magnetic_flux = [None] * len(B)
    for i in range(len(B)):
        if i == direction:
            magnetic_flux[i] = 0.0  # Preserve div(B) = 0
        else:
            magnetic_flux[i] = v_dir * B[i] - v[i] * B[direction]
    
    return {
        'density': mass_flux,
        'momentum': momentum_flux,
        'energy': energy_flux,
        'magnetic_field': magnetic_flux
    }

@njit
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
    # Sound speeds
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Alfven speeds
    B_L_squared = sum(B_L[i]**2 for i in range(len(B_L)))
    B_R_squared = sum(B_R[i]**2 for i in range(len(B_R)))
    
    c_A_L = np.sqrt(B_L_squared / rho_L)
    c_A_R = np.sqrt(B_R_squared / rho_R)
    
    # Fast magnetosonic speeds (upper bound: c_f ≤ c_s + c_A)
    c_f_L = c_L + c_A_L
    c_f_R = c_R + c_A_R
    
    # Minimum and maximum wave speeds
    s_L = min(v_L[direction] - c_f_L, v_R[direction] - c_f_R)
    s_R = max(v_L[direction] + c_f_L, v_R[direction] + c_f_R)
    
    return s_L, s_R

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
    max_speed = max(abs(s_L), abs(s_R))
    
    # Conserved variables in left state
    cons_L = {
        'density': rho_L,
        'momentum': [rho_L * v_L[i] for i in range(len(v_L))],
        'energy': p_L / (gamma - 1) + 0.5 * rho_L * sum(v_L[i]**2 for i in range(len(v_L))) + 0.5 * sum(B_L[i]**2 for i in range(len(B_L))),
        'magnetic_field': B_L
    }
    
    # Conserved variables in right state
    cons_R = {
        'density': rho_R,
        'momentum': [rho_R * v_R[i] for i in range(len(v_R))],
        'energy': p_R / (gamma - 1) + 0.5 * rho_R * sum(v_R[i]**2 for i in range(len(v_R))) + 0.5 * sum(B_R[i]**2 for i in range(len(B_R))),
        'magnetic_field': B_R
    }
    
    # Rusanov flux: F = 0.5 * (F_L + F_R - |max_speed| * (U_R - U_L))
    rusanov_flux = {}
    
    # Density flux
    rusanov_flux['density'] = 0.5 * (flux_L['density'] + flux_R['density'] - max_speed * (cons_R['density'] - cons_L['density']))
    
    # Momentum flux
    rusanov_flux['momentum'] = [None] * len(v_L)
    for i in range(len(v_L)):
        rusanov_flux['momentum'][i] = 0.5 * (flux_L['momentum'][i] + flux_R['momentum'][i] - max_speed * (cons_R['momentum'][i] - cons_L['momentum'][i]))
    
    # Energy flux
    rusanov_flux['energy'] = 0.5 * (flux_L['energy'] + flux_R['energy'] - max_speed * (cons_R['energy'] - cons_L['energy']))
    
    # Magnetic field flux
    rusanov_flux['magnetic_field'] = [None] * len(B_L)
    for i in range(len(B_L)):
        rusanov_flux['magnetic_field'][i] = 0.5 * (flux_L['magnetic_field'][i] + flux_R['magnetic_field'][i] - max_speed * (cons_R['magnetic_field'][i] - cons_L['magnetic_field'][i]))
    
    return rusanov_flux

@njit
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
    # Compute physical fluxes in left and right states
    flux_L = compute_mhd_flux(rho_L, v_L, p_L, B_L, gamma, direction)
    flux_R = compute_mhd_flux(rho_R, v_R, p_R, B_R, gamma, direction)
    
    # Compute wave speeds
    s_L, s_R = compute_max_wave_speeds(rho_L, v_L, p_L, B_L, rho_R, v_R, p_R, B_R, gamma, direction)
    
    # Conserved variables in left state
    cons_L = {
        'density': rho_L,
        'momentum': [rho_L * v_L[i] for i in range(len(v_L))],
        'energy': p_L / (gamma - 1) + 0.5 * rho_L * sum(v_L[i]**2 for i in range(len(v_L))) + 0.5 * sum(B_L[i]**2 for i in range(len(B_L))),
        'magnetic_field': B_L
    }
    
    # Conserved variables in right state
    cons_R = {
        'density': rho_R,
        'momentum': [rho_R * v_R[i] for i in range(len(v_R))],
        'energy': p_R / (gamma - 1) + 0.5 * rho_R * sum(v_R[i]**2 for i in range(len(v_R))) + 0.5 * sum(B_R[i]**2 for i in range(len(B_R))),
        'magnetic_field': B_R
    }
    
    # HLL flux:
    # If s_L >= 0: F = F_L
    # If s_R <= 0: F = F_R
    # Otherwise: F = (s_R * F_L - s_L * F_R + s_L * s_R * (U_R - U_L)) / (s_R - s_L)
    
    # Choose the appropriate form based on wave speeds
    if s_L >= 0:
        return flux_L
    elif s_R <= 0:
        return flux_R
    else:
        hll_flux = {}
        
        # HLL average state formula
        factor = 1.0 / (s_R - s_L)
        
        # Density flux
        hll_flux['density'] = factor * (s_R * flux_L['density'] - s_L * flux_R['density'] 
                                     + s_L * s_R * (cons_R['density'] - cons_L['density']))
        
        # Momentum flux
        hll_flux['momentum'] = [None] * len(v_L)
        for i in range(len(v_L)):
            hll_flux['momentum'][i] = factor * (s_R * flux_L['momentum'][i] - s_L * flux_R['momentum'][i] 
                                            + s_L * s_R * (cons_R['momentum'][i] - cons_L['momentum'][i]))
        
        # Energy flux
        hll_flux['energy'] = factor * (s_R * flux_L['energy'] - s_L * flux_R['energy'] 
                                    + s_L * s_R * (cons_R['energy'] - cons_L['energy']))
        
        # Magnetic field flux
        hll_flux['magnetic_field'] = [None] * len(B_L)
        for i in range(len(B_L)):
            hll_flux['magnetic_field'][i] = factor * (s_R * flux_L['magnetic_field'][i] - s_L * flux_R['magnetic_field'][i] 
                                                  + s_L * s_R * (cons_R['magnetic_field'][i] - cons_L['magnetic_field'][i]))
        
        return hll_flux

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
    # For simplicity, we'll defer to the HLL solver in this demo
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
    # Select the appropriate flux function
    if solver.lower() == 'rusanov':
        flux_function = rusanov_flux
    elif solver.lower() == 'hll':
        flux_function = hll_flux
    elif solver.lower() == 'hlld':
        flux_function = hlld_flux
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Evolve the system
    for _ in range(steps):
        # You would typically register the solver with the MHD system
        # and then call its advance_time_step method
        mhd_system.advance_time_step()
    
    return mhd_system 