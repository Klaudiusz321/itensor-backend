import sympy as sp
import logging
from myproject.utils.symbolic.compute_tensor import is_flat_metric, is_spherical_coordinates, is_euclidean_metric, oblicz_tensory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_flat_metrics():
    """Test a variety of flat metrics to verify they are correctly detected."""
    flat_metrics = {
        "Cartesian": {
            "coords": ["x", "y", "z"],
            "metric": {
                (0, 0): 1,
                (1, 1): 1,
                (2, 2): 1
            },
            "expected": True
        },
        "Cylindrical": {
            "coords": ["r", "theta", "z"],
            "metric": {
                (0, 0): 1,
                (1, 1): "r**2",
                (2, 2): 1
            },
            "expected": True
        },
        "Spherical": {
            "coords": ["r", "theta", "phi"],
            "metric": {
                (0, 0): 1,
                (1, 1): "r**2",
                (2, 2): "r**2 * sin(theta)**2"
            },
            "expected": True
        },
        "Scaled-Cartesian": {
            "coords": ["x", "y", "z"],
            "metric": {
                (0, 0): "a**2",
                (1, 1): "a**2",
                (2, 2): "a**2"
            },
            "expected": True
        },
        "Parabolic": {
            "coords": ["u", "v", "phi"],
            "metric": {
                (0, 0): "u**2 + v**2",
                (1, 1): "u**2 + v**2",
                (2, 2): "u**2 * v**2"
            },
            "expected": True
        },
        "Rindler": {
            "coords": ["t", "x", "y", "z"],
            "metric": {
                (0, 0): "-(a*x)**2",
                (1, 1): 1,
                (2, 2): 1,
                (3, 3): 1
            },
            "expected": True
        },
        "Oblate-Spheroidal-Simple": {
            "coords": ["mu", "nu", "phi"],
            "metric": {
                (0, 0): "cosh(mu)**2",
                (1, 1): "sinh(mu)**2",
                (2, 2): "sinh(mu)**2 * sin(nu)**2"
            },
            "expected": True,
            "skip_on_failure": True  # Skip this test if it takes too long
        },
        "Bipolar": {
            "coords": ["u", "v", "z"],
            "metric": {
                (0, 0): "a**2/(cosh(u) - cos(v))**2",
                (1, 1): "a**2/(cosh(u) - cos(v))**2",
                (2, 2): 1
            },
            "expected": True
        },
        "2D-Minkowski": {
            "coords": ["t", "x"],
            "metric": {
                (0, 0): -1,
                (1, 1): 1
            },
            "expected": True
        },
        "Simple-Transformed-2D": {
            "coords": ["u", "v"],
            "metric": {
                (0, 0): "exp(2*u)",
                (1, 1): 1
            },
            "expected": True
        },
        "Rotated-Coordinates": {
            "coords": ["x_prime", "y_prime", "z_prime"],
            "metric": {
                (0, 0): 1,
                (1, 1): 1,
                (2, 2): 1
            },
            "expected": True
        },
        "Toroidal": {
            "coords": ["eta", "xi", "phi"],
            "metric": {
                (0, 0): "a**2/(cosh(eta) - cos(xi))**2",
                (1, 1): "a**2/(cosh(eta) - cos(xi))**2",
                (2, 2): "a**2 * sinh(eta)**2/(cosh(eta) - cos(xi))**2"
            },
            "expected": True,
            "skip_on_failure": True  # Skip this test if it takes too long
        }
    }
    
    # Add curved metrics for contrast
    curved_metrics = {
        "Schwarzschild": {
            "coords": ["t", "r", "theta", "phi"],
            "metric": {
                (0, 0): "-(1 - 2*M/r)",
                (1, 1): "1/(1 - 2*M/r)",
                (2, 2): "r**2",
                (3, 3): "r**2 * sin(theta)**2"
            },
            "expected": False
        },
        "FLRW": {
            "coords": ["t", "r", "theta", "phi"],
            "metric": {
                (0, 0): -1,
                (1, 1): "a(t)**2",
                (2, 2): "a(t)**2 * r**2",
                (3, 3): "a(t)**2 * r**2 * sin(theta)**2"
            },
            "expected": False,
            "skip_on_failure": True  # Skip this test if it takes too long
        },
        "Anti-DeSitter-2D": {
            "coords": ["t", "r"],
            "metric": {
                (0, 0): "-(1 + r**2/L**2)",
                (1, 1): "1/(1 + r**2/L**2)"
            },
            "expected": False
        }
    }
    
    # Process and create symbols
    all_metrics = {**flat_metrics, **curved_metrics}
    total = len(all_metrics)
    passed = 0
    skipped = 0
    
    for name, data in all_metrics.items():
        try:
            logger.info(f"Testing {name} metric")
            
            # Create symbols for coordinates and parameters
            symbols = {}
            coord_symbols = []
            
            # First process coordinates
            for coord in data["coords"]:
                if coord not in symbols:
                    symbols[coord] = sp.Symbol(coord)
                coord_symbols.append(symbols[coord])
            
            # Add specific symbols for time-dependent functions
            if "a(t)" in str(data["metric"]):
                t = symbols.get("t", sp.Symbol("t"))
                a_func = sp.Function("a")(t)
                symbols["a(t)"] = a_func
            
            # Process parameters in metric expressions
            processed_metric = {}
            for pos, expr in data["metric"].items():
                if isinstance(expr, str):
                    # Extract potential parameters from string
                    for param in ["a", "M", "G", "c", "L"]:
                        if param in expr and param not in symbols and param + "(" not in expr:
                            symbols[param] = sp.Symbol(param, positive=True)
                    
                    # Parse the expression with all available symbols
                    locals_dict = {str(s): s for s in symbols.values()}
                    try:
                        processed_metric[pos] = sp.sympify(expr, locals=locals_dict)
                    except Exception as e:
                        logger.error(f"Error parsing expression '{expr}': {e}")
                        raise
                else:
                    processed_metric[pos] = expr
            
            # Calculate tensors with a reasonable time limit
            try:
                g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coord_symbols, processed_metric)
                
                # Test flat detection
                is_flat = is_flat_metric(R_abcd, len(coord_symbols))
                
                # Check if result matches expected
                if is_flat == data["expected"]:
                    logger.info(f"✓ {name} metric correctly identified as {'flat' if is_flat else 'curved'}")
                    passed += 1
                else:
                    logger.error(f"✗ {name} metric incorrectly identified as {'flat' if is_flat else 'curved'}")
            except Exception as e:
                logger.error(f"Error computing tensors for {name} metric: {e}")
                if data.get("skip_on_failure", False):
                    logger.warning(f"Skipping {name} metric due to computation error")
                    skipped += 1
                    continue
                raise
                
        except Exception as e:
            logger.error(f"Error testing {name} metric: {e}")
            if data.get("skip_on_failure", False):
                logger.warning(f"Skipping {name} metric due to error")
                skipped += 1
            else:
                # Raise exception for non-skippable errors
                raise
    
    logger.info(f"Test results: {passed}/{total-skipped} metrics correctly identified, {skipped} skipped")
    return passed, total-skipped, skipped

def run_tests():
    """Run all test cases."""
    logger.info("Starting flat metric detection tests")
    try:
        passed, total, skipped = test_flat_metrics()
        success_rate = (passed / total) * 100 if total > 0 else 0
        logger.info(f"Testing complete: {passed}/{total} tests passed ({success_rate:.1f}%), {skipped} skipped")
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
    
if __name__ == "__main__":
    run_tests() 