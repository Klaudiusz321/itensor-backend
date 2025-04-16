# iTensor: Tensor Operations for Differential Geometry

A Python module for tensor operations and differential geometry calculations, supporting both symbolic (using SymPy) and numeric (using NumPy) computations.

## Features

- **Coordinate Systems**: Define and work with various coordinate systems (Cartesian, spherical, cylindrical, etc.)
- **Metric Tensors**: Compute metric tensors and their inverses
- **Differential Operators**: Calculate gradients, divergences, curls, and Laplacians in any coordinate system
- **Tensor Operations**: Transform tensors between coordinate systems
- **Christoffel Symbols**: Compute Christoffel symbols and related geometric quantities
- **Visualization**: Built-in tools for visualizing scalar and vector fields

## Installation

```bash
git clone https://github.com/yourusername/Tensor-backend-calculator.git
cd Tensor-backend-calculator
pip install -e .
```

## Quick Start

```python
import numpy as np
import sympy as sp
from myproject.utils.differential_operators import (
    symbolic_gradient, symbolic_divergence, symbolic_laplacian,
    numeric_gradient, numeric_divergence, numeric_laplacian,
    spherical_coordinates, cartesian_to_spherical, spherical_to_cartesian
)

# Symbolic example (using SymPy)
# -----------------------------
# Define spherical coordinates
r, theta, phi = sp.symbols('r theta phi', real=True, positive=True)

# Define a scalar field
f = r**2 * sp.sin(theta)**2 * sp.cos(phi)

# Get the metric tensor for spherical coordinates
g = spherical_coordinates([r, theta, phi])

# Compute the gradient of the scalar field
grad_f = symbolic_gradient(f, g, [r, theta, phi])

# Compute the Laplacian
laplacian_f = symbolic_laplacian(f, g, [r, theta, phi])

# Numeric example (using NumPy)
# ----------------------------
# Create a 3D grid in spherical coordinates
r_vals = np.linspace(0.1, 2.0, 20)
theta_vals = np.linspace(0.1, np.pi-0.1, 20)
phi_vals = np.linspace(0, 2*np.pi-0.1, 20)
r_grid, theta_grid, phi_grid = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')
grid = [r_grid, theta_grid, phi_grid]

# Define a scalar field on the grid
scalar_field = r_grid**2 * np.sin(theta_grid)**2 * np.cos(phi_grid)

# Create the metric tensor
metric = np.zeros((3, 3) + r_grid.shape)
for i in range(r_grid.shape[0]):
    for j in range(r_grid.shape[1]):
        for k in range(r_grid.shape[2]):
            r_val, theta_val = r_grid[i,j,k], theta_grid[i,j,k]
            metric[0, 0, i, j, k] = 1.0  # g_rr = 1
            metric[1, 1, i, j, k] = r_val**2  # g_θθ = r²
            metric[2, 2, i, j, k] = r_val**2 * np.sin(theta_val)**2  # g_φφ = r²sin²θ

# Compute the inverse metric
metric_inverse = np.zeros_like(metric)
for i in range(r_grid.shape[0]):
    for j in range(r_grid.shape[1]):
        for k in range(r_grid.shape[2]):
            metric_inverse[:, :, i, j, k] = np.linalg.inv(metric[:, :, i, j, k])

# Compute the gradient
gradient = numeric_gradient(scalar_field, metric_inverse, grid)

# Compute the Laplacian
laplacian = numeric_laplacian(scalar_field, metric, grid)
```

## Module Structure

- `myproject/`
  - `utils/`
    - `differential_operators/`
      - `__init__.py` - Package initialization and imports
      - `symbolic.py` - Symbolic differential operators using SymPy
      - `numeric.py` - Numeric differential operators using NumPy
      - `transforms.py` - Coordinate transformations

## Running the Examples

Each module contains runnable examples that demonstrate its functionality. To run them:

```bash
# Run the main demo
python -m myproject.utils.differential_operators

# Run specific module examples
python -m myproject.utils.differential_operators.symbolic
python -m myproject.utils.differential_operators.numeric
python -m myproject.utils.differential_operators.transforms
```

## Advanced Usage

### Custom Coordinate Systems

You can define custom coordinate systems by specifying their metric tensors:

```python
import sympy as sp

# Define curvilinear coordinates
u, v, w = sp.symbols('u v w', real=True)

# Define the metric tensor components
g_uu = 1
g_vv = u**2
g_ww = u**2 * sp.sin(v)**2

# Create the metric tensor
g = sp.zeros(3, 3)
g[0, 0] = g_uu
g[1, 1] = g_vv
g[2, 2] = g_ww

# Now you can use this metric with any differential operator
```

### Vector Fields and Tensor Operations

```python
# Define a vector field in spherical coordinates
V = [r*sp.sin(theta), r*sp.cos(theta), 0]

# Compute the divergence
div_V = symbolic_divergence(V, g, [r, theta, phi])

# Compute the curl (for 3D vector fields)
curl_V = symbolic_curl(V, g, [r, theta, phi])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
