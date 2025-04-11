import requests
import json
import sys

def test_numerical_endpoint():
    url = "http://localhost:8000/api/tensors/numerical/"
    payload = {
        "dimension": 4,
        "coordinates": ["t", "r", "θ", "φ"],
        "metric": [
            ["-1", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(θ)**2"]
        ],
        "evaluation_point": [0, 10, 1.5708, 0],
        "calculation_types": ["christoffel_symbols", "ricci_tensor", "ricci_scalar"]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        print("Testing numerical endpoint (primary URL: /api/tensors/numerical/)...")
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response received:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print("Error response:")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_numeric_endpoint_alias():
    url = "http://localhost:8000/api/tensors/numeric/"
    payload = {
        "dimension": 4,
        "coordinates": ["t", "r", "θ", "φ"],
        "metric": [
            ["-1", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(θ)**2"]
        ],
        "evaluation_point": [0, 10, 1.5708, 0],
        "calculation_types": ["christoffel_symbols", "ricci_tensor", "ricci_scalar"]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        print("\nTesting numerical endpoint (alias URL: /api/tensors/numeric/)...")
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response received:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print("Error response:")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_symbolic_endpoint():
    url = "http://localhost:8000/api/tensors/symbolic/"
    payload = {
        "dimension": 4,
        "coordinates": ["t", "r", "θ", "φ"],
        "metric": [
            ["-1", "0", "0", "0"],
            ["0", "1/(1-2*M/r)", "0", "0"],
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(θ)**2"]
        ],
        "calculations": ["christoffel_symbols", "ricci_tensor", "ricci_scalar"]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        print("\nTesting symbolic endpoint...")
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response received:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print("Error response:")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_health_endpoint():
    url = "http://localhost:8000/api/health/"
    
    try:
        print("\nTesting health endpoint...")
        response = requests.get(url)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response received:")
            print(response.text)
            return True
        else:
            print("Error response:")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("API Endpoint Test Script")
    print("------------------------")
    print("Make sure the Django backend is running at http://localhost:8000\n")
    
    success = True
    
    if not test_health_endpoint():
        print("\nHealth endpoint test failed. Make sure the backend server is running.")
        sys.exit(1)
    
    if not test_numerical_endpoint():
        print("\nNumerical endpoint (primary URL) test failed.")
        success = False
    
    if not test_numeric_endpoint_alias():
        print("\nNumerical endpoint (alias URL) test failed.")
        success = False
    
    if not test_symbolic_endpoint():
        print("\nSymbolic endpoint test failed.")
        success = False
    
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the API endpoints.") 