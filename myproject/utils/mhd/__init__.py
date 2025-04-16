"""
Magnetohydrodynamics (MHD) module for iTensor.

This module implements MHD equations, solvers, and utilities for both symbolic and
numerical calculations of magnetohydrodynamic systems.
"""

# Import core components from submodules
from .core import MHDSystem
from .solvers import (
    compute_mhd_flux,
    rusanov_flux,
    hll_flux,
    hlld_flux,
    evolve_mhd
)
from .constrained_transport import (
    initialize_face_centered_b,
    compute_emf,
    update_face_centered_b,
    face_to_cell_centered_b,
    check_divergence_free,
    initialize_from_vector_potential
)
from .initial_conditions import (
    orszag_tang_vortex,
    magnetic_rotor,
    mhd_blast_wave,
    mhd_shock_tube,
    kelvin_helmholtz_mhd
)

# Export the public API
__all__ = [
    # Core MHD system
    'MHDSystem',
    
    # MHD equations and fluxes
    'compute_mhd_flux',
    
    # Riemann solvers
    'rusanov_flux',
    'hll_flux',
    'hlld_flux',
    'evolve_mhd',
    
    # Constrained transport
    'initialize_face_centered_b',
    'compute_emf',
    'update_face_centered_b',
    'face_to_cell_centered_b',
    'check_divergence_free',
    'initialize_from_vector_potential',
    
    # Initial conditions
    'orszag_tang_vortex',
    'magnetic_rotor',
    'mhd_blast_wave',
    'mhd_shock_tube',
    'kelvin_helmholtz_mhd',
] 