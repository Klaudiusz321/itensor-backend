"""
Magnetohydrodynamics (MHD) module for iTensor.

This module implements MHD equations, solvers, and utilities for both symbolic and
numerical calculations of magnetohydrodynamic systems.
"""

# Import core components from submodules
from .core import orszag_tang_vortex_2d as orszag_tang_vortex, MHDSystem

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
    magnetic_rotor,
    mhd_blast_wave,
    mhd_shock_tube,
    kelvin_helmholtz_mhd
)
from .grid import (
    create_grid,
    create_staggered_grid,
    metric_from_transformation,
    compute_christoffel_symbols,
    symbolic_gradient,
    symbolic_divergence,
    numerical_gradient,
    numerical_divergence,
    numerical_curl_2d,
    numerical_curl_3d,
    create_curvilinear_coordinates,
    lambdify_metric_functions,
    apply_boundary_conditions,
    apply_vector_boundary_conditions,
    apply_divB_preserving_boundary
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
    
    # Grid and coordinate functions
    'create_grid',
    'create_staggered_grid',
    'metric_from_transformation',
    'compute_christoffel_symbols',
    'symbolic_gradient',
    'symbolic_divergence',
    'numerical_gradient',
    'numerical_divergence',
    'numerical_curl_2d',
    'numerical_curl_3d',
    'create_curvilinear_coordinates',
    'lambdify_metric_functions',
    'apply_boundary_conditions',
    'apply_vector_boundary_conditions',
    'apply_divB_preserving_boundary',
]

"""
MHD (Magnetohydrodynamics) utilities package.

This package provides utilities for MHD simulations, including:
- Core MHD system classes and solvers
- Grid and coordinate system handling
- Data sanitization and repair functions
"""

from .sanitization import sanitize_array, detect_and_fix_mhd_issues, repair_mhd_inconsistencies 