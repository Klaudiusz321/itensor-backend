# iTensor: Tensor Operations for Differential Geometry

A Python-based web service for tensor operations and differential geometry calculations, supporting both symbolic (using SymPy) and numeric (using NumPy) computations. The backend is built with Django and provides a RESTful API for tensor calculations.

## Features

- **Coordinate Systems**: Define and work with various coordinate systems (Cartesian, spherical, cylindrical, etc.)
- **Metric Tensors**: Compute metric tensors and their inverses
- **Differential Operators**: Calculate gradients, divergences, curls, and Laplacians in any coordinate system
- **Tensor Operations**: Transform tensors between coordinate systems
- **Christoffel Symbols**: Compute Christoffel symbols and related geometric quantities
- **Curvature Tensors**: Calculate Riemann tensor, Ricci tensor, scalar curvature, and Einstein tensor
- **Flat Metric Detection**: Automatically identify flat metrics in various coordinate systems
- **Visualization**: Built-in tools for visualizing scalar and vector fields

## Installation

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for containerized deployment)

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Tensor-backend-calculator.git
cd Tensor-backend-calculator

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start the development server
python manage.py runserver
```

## API Endpoints

The API provides several endpoints for tensor calculations:

- `/api/compute_tensors/`: Calculate tensor quantities for a given metric
- `/api/differential_operators/`: Compute differential operators on scalar/vector fields
- `/api/transformations/`: Transform between coordinate systems
- `/api/numeric/`: Perform numeric tensor calculations
- `/api/symbolic/`: Perform symbolic tensor calculations

For detailed API documentation, visit `/api/docs/` after starting the server.

## Core Module Structure

- `myproject/`
  - `api/`: API endpoints and serializers
  - `utils/`
    - `differential_operators/`: Differential geometry operators
      - `symbolic.py`: Symbolic differential operators using SymPy
      - `numeric.py`: Numeric differential operators using NumPy
      - `transforms.py`: Coordinate transformations
    - `symbolic/`: Symbolic tensor calculations
      - `compute_tensor.py`: Core tensor computation module
      - `simplification/`: Expression simplification utilities
    - `numerical/`: Numeric tensor calculations
    - `mhd/`: Magnetohydrodynamics simulation utilities
- `calculator/`: Django app for tensor calculations
  - `views.py`: Main view functions
  - `symbolic_views.py`: Symbolic calculation views
  - `numerical_views.py`: Numerical calculation views
  - `differential_operators_views.py`: Views for differential operators

## Example Usage

### Sample Input Metric (Spherical Coordinates)

```python
# Spherical coordinates
r, theta, phi = sp.symbols('r theta phi', real=True, positive=True)

# Metric tensor components
g = {
    (0, 0): 1,
    (1, 1): r**2,
    (2, 2): r**2 * sp.sin(theta)**2
}

# Compute tensor quantities
result = compute_tensors(coordinates=[r, theta, phi], metric=g)
```

### Output
```
Christoffel symbols (Γ^a_{bc}):
Γ^1_{11} = 0
Γ^1_{22} = -r*sin(θ)^2
Γ^1_{33} = -r
Γ^2_{12} = 1/r
Γ^2_{33} = -sin(θ)*cos(θ)
Γ^3_{13} = 1/r
Γ^3_{23} = cot(θ)

Riemann tensor (R_{abcd}):
All components = 0 (flat metric)

Ricci tensor (R_{ij}):
All components = 0

Scalar Curvature:
R = 0
```

## Testing

```bash
# Run unit tests
python manage.py test

# Run specific test file
python test_flat_metrics.py

# Test with Docker
./deploy_test.sh
```

## Deployment

The project includes Docker configurations for easy deployment:

```bash
# Build and deploy with Docker
docker build -t itensor-backend .
docker run -p 8000:8000 itensor-backend

# Production deployment
./deploy.sh

# Test deployment
./deploy_test.sh
```

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [SymPy](https://www.sympy.org/) - Python library for symbolic mathematics
- [NumPy](https://numpy.org/) - Fundamental package for scientific computing with Python
- [Django](https://www.djangoproject.com/) - High-level Python Web framework
- [Django REST Framework](https://www.django-rest-framework.org/) - Toolkit for building Web APIs
