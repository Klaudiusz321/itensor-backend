#!/usr/bin/env python
"""
Script to test the numerical tensor calculation API endpoint.
This sends a test request to the running Django server.
"""

import json
import requests
import numpy as np
import pprint

def test_numerical_endpoint():
    """Test the numerical calculation endpoint with valid data."""
    # Test with a simple Schwarzschild metric
    valid_request = {
        "dimension": 4,
        "coordinates": ["t", "r", "θ", "φ"],
        "metric": [
            ["-1*(1 - 2/r)", "0", "0", "0"],
            ["0", "1/(1 - 2/r)", "0", "0"],
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(θ)**2"]
        ],
        "evaluation_point": [0, 10, np.pi/2, 0],  # r=10, far from Schwarzschild radius
        "calculation_types": ["christoffel_symbols", "ricci_tensor", "ricci_scalar"]
    }
    
    print("Sending valid request to /api/tensors/numerical/")
    try:
        response = requests.post(
            "http://localhost:8000/api/tensors/numerical/",
            json=valid_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Success! Response received:")
            print(f"Dimension: {data.get('dimension')}")
            print(f"Coordinates: {data.get('coordinates')}")
            print(f"Ricci scalar: {data.get('ricci_scalar')}")
            
            # Check for Christoffel symbols
            christoffel = data.get('christoffel_symbols', {})
            nonzero_symbols = {k: v for k, v in christoffel.items() if abs(v) > 1e-10}
            print(f"Non-zero Christoffel symbols: {len(nonzero_symbols)}")
            if nonzero_symbols:
                print("First few Christoffel symbols:")
                for i, (idx, val) in enumerate(list(nonzero_symbols.items())[:5]):
                    print(f"  Γ_{idx} = {val}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Error making request: {e}")

def test_invalid_request():
    """Test the numerical calculation endpoint with invalid data."""
    # Test with empty metric component
    invalid_request = {
        "dimension": 4,
        "coordinates": ["t", "r", "θ", "φ"],
        "metric": [
            ["-1*(1 - 2/r)", "0", "0", "0"],
            ["0", "", "0", "0"],  # Empty component
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(θ)**2"]
        ],
        "evaluation_point": [0, 10, np.pi/2, 0]
    }
    
    print("\nSending invalid request with empty metric component")
    try:
        response = requests.post(
            "http://localhost:8000/api/tensors/numerical/",
            json=invalid_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error making request: {e}")
    
    # Test with syntax error in expression
    invalid_syntax_request = {
        "dimension": 4,
        "coordinates": ["t", "r", "θ", "φ"],
        "metric": [
            ["-1*(1 - 2/r)", "0", "0", "0"],
            ["0", "1/(1 - 2/r", "0", "0"],  # Missing closing parenthesis
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(θ)**2"]
        ],
        "evaluation_point": [0, 10, np.pi/2, 0]
    }
    
    print("\nSending invalid request with syntax error in expression")
    try:
        response = requests.post(
            "http://localhost:8000/api/tensors/numerical/",
            json=invalid_syntax_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    print("Testing numerical tensor calculation API")
    test_numerical_endpoint()
    test_invalid_request() 