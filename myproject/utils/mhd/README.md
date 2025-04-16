# MHD Module for iTensor

This module implements magnetohydrodynamics (MHD) simulations for the iTensor platform. It provides a comprehensive framework for solving and analyzing MHD equations in different coordinate systems.

## Features

- **Ideal and Resistive MHD**: Support for both ideal and resistive MHD simulations
- **Constrained Transport**: Implementation of constrained transport method to maintain div(B) = 0
- **Multiple Riemann Solvers**: Including HLL, HLLD, and Rusanov (local Lax-Friedrichs) solvers
- **Coordinate System Support**: Works with the differential operators module to support simulations in arbitrary curvilinear coordinates
- **Standard Test Problems**: Includes implementations of common test problems like Orszag-Tang vortex, magnetic rotor, MHD blast wave, and shock tubes
- **Performance Optimization**: Uses Numba JIT compilation for critical numerical routines
- **Visualization Tools**: Includes utilities for plotting and creating animations of results

## Module Structure

- `core.py`: Contains the main `MHDSystem` class that coordinates the MHD simulation
- `solvers.py`: Implements various Riemann solvers for computing numerical fluxes
- `constrained_transport.py`: Contains utilities for maintaining the div(B) = 0 constraint
- `initial_conditions.py`: Implements standard test problems and initial conditions
- `demo_orszag_tang.py` and `demo_magnetic_rotor.py`: Demonstration scripts

## Physical Background

Magnetohydrodynamics studies the flow of electrically conducting fluids in magnetic fields. The MHD equations combine the Navier-Stokes equations of fluid dynamics with Maxwell's equations of electromagnetism. In the ideal MHD limit, the fluid is assumed to have infinite electrical conductivity.

The equations are typically written in conservative form:

- Mass conservation: ∂ρ/∂t + ∇·(ρv) = 0
- Momentum conservation: ∂(ρv)/∂t + ∇·(ρvv + pI - BB) = 0
- Energy conservation: ∂E/∂t + ∇·[(E + p)v - B(v·B)] = 0
- Magnetic induction: ∂B/∂t + ∇×(v×B) = 0

With an additional constraint:
- Div(B) = 0

Where:
- ρ is density
- v is velocity
- p is pressure
- B is magnetic field
- E is total energy density: E = p/(γ-1) + ρv²/2 + B²/2
- γ is the adiabatic index (typically 5/3 for monatomic gas)

## Numerical Methods

The MHD module implements a finite volume method with the following features:

1. **Conservation Law Form**: The equations are solved in conservation law form to maintain mass, momentum, energy, and magnetic flux conservation.

2. **Method of Lines**: The spatial derivatives are discretized while keeping time continuous, then standard ODE integrators are used for time evolution.

3. **Riemann Solvers**: Interface fluxes are computed using approximate Riemann solvers (HLL, HLLD, Rusanov) that consider the wave structure of MHD.

4. **Constrained Transport**: To maintain the divergence-free constraint on the magnetic field, the module uses a staggered grid approach where magnetic field components are stored at cell faces.

5. **CFL Condition**: Time steps are chosen according to the CFL (Courant-Friedrichs-Lewy) condition based on the maximum wave speed.

## Running the Demos

### Prerequisites

- Python 3.6+
- NumPy
- Matplotlib
- Numba

### Orszag-Tang Vortex Demo

The Orszag-Tang vortex is a standard 2D MHD test problem that leads to complex shock interactions and vortical structures.

To run the demo:

```bash
python demo_orszag_tang.py --resolution 128 --time 1.0 --interval 0.1
```

Options:
- `--resolution`: Grid resolution (default: 128)
- `--time`: Final simulation time (default: 1.0)
- `--interval`: Output interval (default: 0.1)
- `--no-plot`: Disable plotting (use for performance testing)

### Magnetic Rotor Demo

The magnetic rotor problem involves a dense rotating disk in a uniform magnetic field, generating torsional Alfvén waves.

To run the demo:

```bash
python demo_magnetic_rotor.py --resolution 256 --time 0.4 --interval 0.05
```

Options:
- `--resolution`: Grid resolution (default: 256)
- `--time`: Final simulation time (default: 0.4)
- `--interval`: Output interval (default: 0.05)
- `--no-plot`: Disable plotting

## Example Usage

To use the MHD module in your own code:

```python
import numpy as np
from myproject.utils.mhd import MHDSystem, orszag_tang_vortex

# Define domain and grid
domain_size = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain [0,1] x [0,1]
resolution = (128, 128)

# Create coordinate system (Cartesian)
coordinate_system = {
    'name': 'cartesian',
    'coordinates': ['x', 'y'],
    'transformation': None
}

# Create MHD system
mhd = MHDSystem(coordinate_system, domain_size, resolution, gamma=5/3)

# Initialize with Orszag-Tang vortex
ot_init = orszag_tang_vortex(mhd.grid, gamma=5/3)

# Set initial conditions
mhd.set_initial_conditions(
    lambda x, y: ot_init['density'],
    [lambda x, y: ot_init['velocity'][0], lambda x, y: ot_init['velocity'][1]],
    lambda x, y: ot_init['pressure'],
    [lambda x, y: ot_init['magnetic_field'][0], lambda x, y: ot_init['magnetic_field'][1]]
)

# Evolve the system
final_state = mhd.evolve(final_time=1.0)
```

## Extending the Module

To add new features to the MHD module:

1. **New Initial Conditions**: Add functions to `initial_conditions.py` that return appropriate density, velocity, pressure, and magnetic field arrays.

2. **New Riemann Solvers**: Implement additional Riemann solvers in `solvers.py` following the patterns of the existing solvers.

3. **Alternative Time Integration**: Modify the `advance_time_step` method in `MHDSystem` to use different ODE integrators.

4. **Additional Physics**: To add physics like resistivity, gravity, or radiation, modify the `compute_rhs` method to include additional source terms.

## References

1. Stone, J. M., & Norman, M. L. (1992). ZEUS-2D: A radiation magnetohydrodynamics code for astrophysical flows in two space dimensions. I. The hydrodynamic algorithms and tests. The Astrophysical Journal Supplement Series, 80, 753-790.

2. Tóth, G. (2000). The ∇·B=0 Constraint in Shock-Capturing Magnetohydrodynamics Codes. Journal of Computational Physics, 161(2), 605-652.

3. Gardiner, T. A., & Stone, J. M. (2005). An unsplit Godunov method for ideal MHD via constrained transport. Journal of Computational Physics, 205(2), 509-539.

4. Orszag, S. A., & Tang, C.-M. (1979). Small-scale structure of two-dimensional magnetohydrodynamic turbulence. Journal of Fluid Mechanics, 90(1), 129-143.

## License

This module is part of the iTensor project and is available under the project's license terms. 