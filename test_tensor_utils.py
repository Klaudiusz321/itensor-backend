#!/usr/bin/env python
"""
Test script for tensor_utils.py component function evaluation.
This script tests the improved error handling in the tensor_utils.py module.
"""

import sys
import numpy as np
import warnings
from myproject.utils.numerical.tensor_utils import create_component_function

def run_tests():
    """Run comprehensive tests for component_func."""
    # Suppress divide by zero warnings for our tests
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    print("Testing tensor_utils.py component function evaluation...")
    
    coordinates = ['t', 'r', 'θ', 'φ']
    test_coords = np.array([0.0, 3.0, np.pi/2, 0.0])  # Using r=3 to avoid singularity
    
    # Test cases for valid expressions
    valid_expressions = [
        ("1", 1.0),
        ("r", 3.0),
        ("r**2", 9.0),
        ("r**2 * sin(θ)**2", 9.0),
        ("sin(θ)", 1.0),
        ("1/(1-2/r)", 3.0),
        ("sqrt(r)", np.sqrt(3.0))
    ]
    
    print("\nTesting valid expressions:")
    print("--------------------------")
    for expr, expected in valid_expressions:
        try:
            func = create_component_function(expr, coordinates)
            result = func(test_coords)
            print(f"Expression: '{expr}' → Result: {result} (Expected: {expected})")
            assert abs(result - expected) < 1e-10, f"Result {result} does not match expected {expected}"
        except Exception as e:
            print(f"ERROR: Expression '{expr}' failed: {str(e)}")
    
    # Test cases for invalid expressions
    invalid_expressions = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("r**", "Incomplete exponentiation"),
        ("sin(r", "Unclosed parenthesis"),
        ("unknown_var", "Unknown variable"),
        ("unknown_func(r)", "Unknown function")
    ]
    
    print("\nTesting invalid expressions (should raise ValueError):")
    print("-----------------------------------------------------")
    for expr, desc in invalid_expressions:
        try:
            func = create_component_function(expr, coordinates)
            result = func(test_coords)
            print(f"UNEXPECTED SUCCESS ({desc}): '{expr}' → {result}")
        except ValueError as e:
            print(f"Expected error ({desc}): {str(e)}")

    # Additional tests with Schwarzschild metric
    print("\nTesting with Schwarzschild metric components:")
    print("--------------------------------------------")
    schwarzschild_coords = np.array([0.0, 10.0, np.pi/2, 0.0])  # r=10, well outside the Schwarzschild radius
    
    # Schwarzschild metric components
    schwarzschild = [
        ("-(1 - 2/r)", "g_00"),  # g_tt
        ("1/(1 - 2/r)", "g_11"),  # g_rr
        ("r**2", "g_22"),  # g_θθ
        ("r**2 * sin(θ)**2", "g_33")  # g_φφ
    ]
    
    print(f"Evaluating at coordinates: t={schwarzschild_coords[0]}, r={schwarzschild_coords[1]}, θ={schwarzschild_coords[2]}, φ={schwarzschild_coords[3]}")
    for expr, name in schwarzschild:
        try:
            func = create_component_function(expr, coordinates)
            result = func(schwarzschild_coords)
            print(f"{name}: '{expr}' → {result}")
        except ValueError as e:
            print(f"ERROR in {name}: {str(e)}")

    # Test handling of singular expressions
    print("\nTesting expressions at singular points:")
    print("-------------------------------------")
    singular_coords = np.array([0.0, 2.0, np.pi/2, 0.0])  # r=2 is Schwarzschild radius for M=1
    print(f"Evaluating at Schwarzschild radius: t={singular_coords[0]}, r={singular_coords[1]}, θ={singular_coords[2]}, φ={singular_coords[3]}")
    
    for expr, name in schwarzschild:
        try:
            func = create_component_function(expr, coordinates)
            result = func(singular_coords)
            print(f"{name}: '{expr}' → {result}")
            if name == "g_11" and np.isinf(result):
                print(f"  Note: As expected, {name} is infinite at the Schwarzschild radius (r=2)")
        except Exception as e:
            print(f"ERROR in {name}: {str(e)}")

def main():
    """Main entry point."""
    run_tests()
    print("\nAll tests completed.")

if __name__ == "__main__":
    main() 