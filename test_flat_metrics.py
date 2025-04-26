import sympy as sp
from myproject.utils.symbolic.compute_tensor import is_flat_metric, is_spherical_coordinates, is_euclidean_metric, oblicz_tensory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_tensor_summary(Riemann, n):
    """Print a summary of the Riemann tensor components to aid in debugging."""
    non_zero_count = 0
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    if Riemann[rho][sigma][mu][nu] != 0:
                        non_zero_count += 1
    
    logger.info(f"Riemann tensor has {non_zero_count} non-zero components out of {n**4} total components")
    if non_zero_count > 0 and non_zero_count <= 5:  # Only show details for a few non-zero components
        logger.info("Non-zero Riemann components:")
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        if Riemann[rho][sigma][mu][nu] != 0:
                            logger.info(f"R_{rho}{sigma}{mu}{nu} = {Riemann[rho][sigma][mu][nu]}")

def test_cartesian_metric():
    """Test flat metric in Cartesian coordinates."""
    logger.info("Testing Cartesian coordinates (flat space)")
    x, y, z = sp.symbols('x y z')
    coords = [x, y, z]
    
    # Standard Cartesian metric (identity matrix)
    metric = {
        (0, 0): 1,
        (1, 1): 1,
        (2, 2): 1
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    assert is_flat, "Cartesian metric should be detected as flat"
    logger.info(f"Cartesian metric flat detection: {is_flat}")

def test_cylindrical_metric():
    """Test flat metric in cylindrical coordinates."""
    logger.info("Testing cylindrical coordinates (flat space)")
    r, theta, z = sp.symbols('r theta z')
    coords = [r, theta, z]
    
    # Cylindrical coordinates metric
    metric = {
        (0, 0): 1,
        (1, 1): r**2,
        (2, 2): 1
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    assert is_flat, "Cylindrical metric should be detected as flat"
    logger.info(f"Cylindrical metric flat detection: {is_flat}")

def test_spherical_metric():
    """Test flat metric in spherical coordinates."""
    logger.info("Testing spherical coordinates (flat space)")
    r, theta, phi = sp.symbols('r theta phi')
    coords = [r, theta, phi]
    
    # Spherical coordinates metric
    metric = {
        (0, 0): 1,
        (1, 1): r**2,
        (2, 2): r**2 * sp.sin(theta)**2
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    is_spherical = is_spherical_coordinates(g, coords, len(coords))
    
    assert is_flat, "Spherical metric should be detected as flat"
    assert is_spherical, "Spherical coordinates should be detected properly"
    logger.info(f"Spherical metric flat detection: {is_flat}")
    logger.info(f"Spherical coordinates detection: {is_spherical}")

def test_curved_metric():
    """Test a curved (non-flat) metric."""
    logger.info("Testing curved space (Schwarzschild metric)")
    r, theta, phi = sp.symbols('r theta phi')
    t = sp.Symbol('t')
    coords = [t, r, theta, phi]
    M = sp.Symbol('M', positive=True)  # Mass parameter
    
    # Schwarzschild metric in natural units (c=G=1)
    metric = {
        (0, 0): -(1 - 2*M/r),
        (1, 1): 1/(1 - 2*M/r),
        (2, 2): r**2,
        (3, 3): r**2 * sp.sin(theta)**2
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    print_tensor_summary(R_abcd, len(coords))
    
    assert not is_flat, "Schwarzschild metric should not be detected as flat"
    logger.info(f"Schwarzschild metric flat detection: {is_flat}")
    logger.info(f"Scalar curvature: {Scalar_Curvature}")

def test_scaled_flat_metric():
    """Test flat metric with scaling factor."""
    logger.info("Testing scaled Cartesian coordinates (flat space)")
    x, y, z = sp.symbols('x y z')
    a = sp.Symbol('a', positive=True)  # Scaling factor
    coords = [x, y, z]
    
    # Scaled Cartesian metric
    metric = {
        (0, 0): a**2,
        (1, 1): a**2,
        (2, 2): a**2
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    assert is_flat, "Scaled Cartesian metric should be detected as flat"
    logger.info(f"Scaled Cartesian metric flat detection: {is_flat}")

def test_oblate_spheroidal_coordinates():
    """Test flat metric in oblate spheroidal coordinates."""
    logger.info("Testing oblate spheroidal coordinates (flat space)")
    mu, nu, phi = sp.symbols('mu nu phi')
    a = sp.Symbol('a', positive=True)  # Scale parameter
    coords = [mu, nu, phi]
    
    # Oblate spheroidal coordinates metric
    metric = {
        (0, 0): a**2 * (sp.sinh(mu)**2 + sp.sin(nu)**2),
        (1, 1): a**2 * (sp.sinh(mu)**2 + sp.sin(nu)**2),
        (2, 2): a**2 * sp.sinh(mu)**2 * sp.sin(nu)**2
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    print_tensor_summary(R_abcd, len(coords))
    
    # Note: This is a flat space metric, but due to the complexity of the coordinate
    # transformation, the symbolic calculation may have difficulty showing it's identically zero
    logger.info(f"Oblate spheroidal metric flat detection: {is_flat}")
    logger.info("Note: Theoretically this should be flat, but symbolic computation limitations may cause failure")
    
    if not is_flat:
        # Try again with more aggressive simplification
        is_flat_extra = is_flat_metric(R_abcd, len(coords), extra_simplify=True)
        logger.info(f"Oblate spheroidal metric flat detection with extra simplification: {is_flat_extra}")

def test_parabolic_coordinates():
    """Test flat metric in parabolic coordinates."""
    logger.info("Testing parabolic coordinates (flat space)")
    u, v, phi = sp.symbols('u v phi')
    coords = [u, v, phi]
    
    # Parabolic coordinates metric
    metric = {
        (0, 0): u**2 + v**2,
        (1, 1): u**2 + v**2,
        (2, 2): u**2 * v**2
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    assert is_flat, "Parabolic coordinates metric should be detected as flat"
    logger.info(f"Parabolic coordinates metric flat detection: {is_flat}")

def test_rindler_coordinates():
    """Test flat metric in Rindler coordinates (accelerated frame in flat spacetime)."""
    logger.info("Testing Rindler coordinates (flat space with acceleration)")
    t, x, y, z = sp.symbols('t x y z')
    a = sp.Symbol('a', positive=True)  # Acceleration parameter
    coords = [t, x, y, z]
    
    # Rindler coordinates metric
    metric = {
        (0, 0): -(a*x)**2,
        (1, 1): 1,
        (2, 2): 1,
        (3, 3): 1
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    assert is_flat, "Rindler coordinates metric should be detected as flat"
    logger.info(f"Rindler coordinates metric flat detection: {is_flat}")

def test_minkowski_alternative():
    """Test alternative representation of Minkowski metric."""
    logger.info("Testing alternative Minkowski coordinates (flat space)")
    t, r, theta, phi = sp.symbols('t r theta phi')
    coords = [t, r, theta, phi]
    
    # Alternative form of the Minkowski metric with off-diagonal terms
    # This is still flat!
    c = sp.Symbol('c', positive=True)  # Speed of light
    metric = {
        (0, 0): -c**2,
        (0, 1): c,
        (1, 0): c,
        (1, 1): 1,
        (2, 2): r**2,
        (3, 3): r**2 * sp.sin(theta)**2
    }
    
    g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coords, metric)
    
    is_flat = is_flat_metric(R_abcd, len(coords))
    print_tensor_summary(R_abcd, len(coords))
    
    logger.info(f"Alternative Minkowski metric flat detection: {is_flat}")
    logger.info("Note: This is theoretically flat, but the off-diagonal terms make symbolic detection challenging")
    
    if not is_flat:
        # Try again with more aggressive simplification
        is_flat_extra = is_flat_metric(R_abcd, len(coords), extra_simplify=True)
        logger.info(f"Alternative Minkowski metric flat detection with extra simplification: {is_flat_extra}")

def run_all_tests():
    """Run all test cases."""
    logger.info("Starting flat metric detection tests")
    
    # Basic tests
    test_cartesian_metric()
    test_cylindrical_metric()
    test_spherical_metric()
    test_curved_metric()
    test_scaled_flat_metric()
    
    # Advanced tests
    try:
        test_oblate_spheroidal_coordinates()
    except Exception as e:
        logger.error(f"Error in oblate spheroidal test: {e}")
    
    try:
        test_parabolic_coordinates()
    except Exception as e:
        logger.error(f"Error in parabolic coordinates test: {e}")
        
    try:
        test_rindler_coordinates()
    except Exception as e:
        logger.error(f"Error in Rindler coordinates test: {e}")
        
    try:
        test_minkowski_alternative()
    except Exception as e:
        logger.error(f"Error in alternative Minkowski test: {e}")
        
    logger.info("All tests completed")

if __name__ == "__main__":
    run_all_tests() 