import requests
import json
import sys

def test_symbolic_endpoint():
    """Test the symbolic tensor calculation endpoint"""
    url = "http://localhost:8000/api/tensors/symbolic/"
    
    # Test with Schwarzschild metric
    payload = {
        "dimension": 4,
        "coordinates": ["t", "r", "theta", "phi"],
        "metric": [
            ["-1 * (1 - 2*M/r)", "0", "0", "0"],
            ["0", "1/(1 - 2*M/r)", "0", "0"],
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(theta)**2"]
        ],
        "calculations": [
            "christoffel_symbols",
            "riemann_tensor",
            "ricci_tensor",
            "ricci_scalar",
            "einstein_tensor",
            "weyl_tensor"
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"Testing symbolic endpoint with Schwarzschild metric")
    print("Request payload:")
    print(json.dumps(payload, indent=2))
    print("Sending request to:", url)
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"\nStatus code: {response.status_code}")
        
        if response.status_code == 200:
            print("\nSuccess! Symbolic calculation completed.")
            
            data = response.json()
            
            # Print basic info
            print(f"\nDimension: {data.get('dimension')}")
            print(f"Coordinates: {', '.join(data.get('coordinates', []))}")
            print(f"Scalar curvature: {data.get('scalar_curvature')}")
            
            # Count non-zero components
            metric_count = len(data.get('metric_components', {}))
            christoffel_count = len(data.get('christoffel_symbols', {}))
            riemann_count = len(data.get('riemann_tensor', {}))
            ricci_count = len(data.get('ricci_tensor', {}))
            einstein_count = len(data.get('einstein_tensor', {}))
            weyl_count = len(data.get('weyl_tensor', {}))
            
            print("\nNon-zero components:")
            print(f"Metric tensor: {metric_count}")
            print(f"Christoffel symbols: {christoffel_count}")
            print(f"Riemann tensor: {riemann_count}")
            print(f"Ricci tensor: {ricci_count}")
            print(f"Einstein tensor: {einstein_count}")
            print(f"Weyl tensor: {weyl_count}")
            
            # Print first few Christoffel symbols
            if christoffel_count > 0:
                print("\nSample Christoffel symbols:")
                for i, (key, value) in enumerate(data.get('christoffel_symbols', {}).items()):
                    print(f"Γ_{key} = {value}")
                    if i >= 2:  # Only show first 3
                        print("...")
                        break
            
            # Print first few Ricci tensor components
            if ricci_count > 0:
                print("\nSample Ricci tensor components:")
                for i, (key, value) in enumerate(data.get('ricci_tensor', {}).items()):
                    print(f"R_{key} = {value}")
                    if i >= 2:  # Only show first 3
                        print("...")
                        break
            
            return True
        else:
            print("\nError response:")
            print(response.text)
            return False
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False

def test_simple_metric():
    """Test the symbolic endpoint with a very simple metric"""
    url = "http://localhost:8000/api/tensors/symbolic/"
    
    # Test with flat Minkowski metric
    payload = {
        "dimension": 4,
        "coordinates": ["t", "x", "y", "z"],
        "metric": [
            ["-1", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
            ["0", "0", "0", "1"]
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"\nTesting symbolic endpoint with simple Minkowski metric")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Simple metric calculation completed.")
            
            data = response.json()
            print(f"Scalar curvature: {data.get('scalar_curvature')}")
            
            return True
        else:
            print("Error response:")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_incomplete_payload():
    """Test the endpoint with an incomplete payload"""
    url = "http://localhost:8000/api/tensors/symbolic/"
    
    # Test with minimal payload
    payload = {
        "coordinates": ["t", "r", "theta", "phi"]
        # Missing metric field
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"\nTesting symbolic endpoint with incomplete payload")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status code: {response.status_code}")
        
        if response.status_code != 200:
            print("Received expected error for incomplete payload:")
            print(response.text)
            return True
        else:
            print("Unexpected success with incomplete payload!")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Symbolic API Endpoint Test Script")
    print("=================================")
    print("Make sure the Django backend is running at http://localhost:8000\n")
    
    success = True
    
    if not test_symbolic_endpoint():
        print("\nMain symbolic endpoint test failed.")
        success = False
    
    if not test_simple_metric():
        print("\nSimple metric test failed.")
        success = False
    
    if not test_incomplete_payload():
        print("\nIncomplete payload test failed.")
        success = False
    
    if success:
        print("\nAll tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the symbolic API endpoint.")
        sys.exit(1) 