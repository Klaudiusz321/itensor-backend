import json
import logging
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import numpy as np

# Import our tensor utilities
from myproject.utils.numerical.tensor_utils import calculate_all_tensors

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def numerical_calculate_view(request):
    """
    API endpoint for numerical tensor calculations.
    
    Expected JSON input format:
    {
        "dimension": 4,                      # Dimension of the manifold
        "coordinates": ["t", "r", "θ", "φ"], # Names of coordinates
        "metric": [                          # Metric tensor components as a nested list
            ["-1", "0", "0", "0"],           # Components can be numbers or string expressions
            ["0", "1/(1-2*M/r)", "0", "0"],
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(θ)**2"]
        ],
        "evaluation_point": [0, 10, 1.5708, 0], # Point at which to evaluate tensors
        "calculation_types": ["christoffel_symbols", "riemann_tensor", "ricci_tensor", 
                              "ricci_scalar", "einstein_tensor", "weyl_tensor"]
    }
    
    Returns:
    {
        "success": true,
        "dimension": 4,
        "coordinates": ["t", "r", "θ", "φ"],
        "evaluation_point": [0, 10, 1.5708, 0],
        "metric": [...],                    # Evaluated metric as 2D array
        "inverse_metric": [...],            # Inverse metric as 2D array
        "christoffel_symbols": {            # Non-zero Christoffel symbols as dictionary
            "0,1,0": 0.11111,               # Format: "upper_index,lower_index_1,lower_index_2": value
            ...
        },
        "riemann_tensor": {                 # Non-zero Riemann tensor components
            "0,1,0,1": 0.22222,             # Format: "upper_index,lower_index_1,lower_index_2,lower_index_3": value
            ...
        },
        "ricci_tensor": [...],              # Ricci tensor as 2D array
        "ricci_scalar": 0.0,                # Ricci scalar (scalar curvature)
        "einstein_tensor": [...],           # Einstein tensor as 2D array
        "weyl_tensor": {...}                # Weyl tensor components
    }
    """
    try:
        # Log request details
        logger.info(f"Received numerical calculation request from {request.META.get('REMOTE_ADDR')}")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Content type: {request.META.get('CONTENT_TYPE')}")
        
        # Parse JSON request
        try:
            data = json.loads(request.body)
            logger.info(f"Parsed request data: {data}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            return JsonResponse({
                "success": False,
                "error": f"Invalid JSON format: {str(e)}"
            }, status=400)
        
        # Validate required fields
        required_fields = ["dimension", "coordinates", "metric", "evaluation_point"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return JsonResponse({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }, status=400)
        
        # Extract data
        dimension = data["dimension"]
        coordinates = data["coordinates"]
        metric_data = data["metric"]
        evaluation_point = data["evaluation_point"]
        calculation_types = data.get("calculation_types", ["all"])
        
        logger.info(f"Dimension: {dimension}")
        logger.info(f"Coordinates: {coordinates}")
        logger.info(f"Metric data shape: {len(metric_data)}x{len(metric_data[0]) if metric_data and metric_data[0] else 0}")
        logger.info(f"Evaluation point: {evaluation_point}")
        logger.info(f"Calculation types: {calculation_types}")
        
        # Validate dimension consistency
        if len(coordinates) != dimension:
            logger.error(f"Dimension mismatch: coordinates length ({len(coordinates)}) != dimension ({dimension})")
            return JsonResponse({
                "success": False,
                "error": f"Number of coordinates ({len(coordinates)}) does not match dimension ({dimension})"
            }, status=400)
            
        if len(metric_data) != dimension or any(len(row) != dimension for row in metric_data):
            logger.error(f"Dimension mismatch: metric dimensions mismatch with specified dimension {dimension}")
            return JsonResponse({
                "success": False,
                "error": f"Metric dimensions ({len(metric_data)}x{len(metric_data[0]) if metric_data else 0}) do not match the specified dimension ({dimension}x{dimension})"
            }, status=400)
            
        if len(evaluation_point) != dimension:
            logger.error(f"Dimension mismatch: evaluation point length ({len(evaluation_point)}) != dimension ({dimension})")
            return JsonResponse({
                "success": False,
                "error": f"Evaluation point has {len(evaluation_point)} coordinates, expected {dimension}"
            }, status=400)
        
        # Additional validation of the metric data
        for i in range(dimension):
            for j in range(dimension):
                if i >= len(metric_data) or j >= len(metric_data[i]):
                    logger.error(f"Missing metric component at position [{i}][{j}]")
                    return JsonResponse({
                        "success": False,
                        "error": f"Missing metric component at position [{i}][{j}]"
                    }, status=400)
                
                component = metric_data[i][j]
                # The component can be None, empty string, numeric, or a valid string expression
                # Actual handling of these cases is in the tensor_utils.py module
                logger.debug(f"Metric component [{i}][{j}] = {component} (type: {type(component).__name__})")
        
        # Convert evaluation_point elements to float
        try:
            evaluation_point = [float(x) if x is not None else 0.0 for x in evaluation_point]
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid evaluation point value: {str(e)}")
            return JsonResponse({
                "success": False,
                "error": f"Evaluation point contains invalid values: {str(e)}"
            }, status=400)
        
        # Calculate all tensors
        logger.info(f"Calculating numerical tensors for {dimension}D metric at point {evaluation_point}")
        try:
            result = calculate_all_tensors(metric_data, coordinates, evaluation_point)
            logger.info("Tensor calculation successful")
        except Exception as calc_error:
            logger.error(f"Calculation error: {str(calc_error)}")
            logger.error(traceback.format_exc())
            return JsonResponse({
                "success": False,
                "error": f"Calculation error: {str(calc_error)}"
            }, status=500)
        
        # Filter results based on requested calculation types
        if "all" not in calculation_types:
            filtered_result = {
                "success": True,
                "dimension": result["dimension"],
                "coordinates": result["coordinates"],
                "evaluation_point": result["evaluation_point"],
                "metric": result["metric"],
                "inverse_metric": result["inverse_metric"]
            }
            
            tensor_types = {
                "christoffel_symbols": "christoffel_symbols",
                "riemann_tensor": "riemann_tensor",
                "ricci_tensor": "ricci_tensor",
                "ricci_scalar": "ricci_scalar",
                "einstein_tensor": "einstein_tensor",
                "weyl_tensor": "weyl_tensor"
            }
            
            for calc_type, result_key in tensor_types.items():
                if calc_type in calculation_types and result_key in result:
                    filtered_result[result_key] = result[result_key]
                    
            result = filtered_result
        else:
            result["success"] = True
        
        # Return the result
        logger.info("Returning successful response")
        return JsonResponse(result, safe=False)
            
    except ValueError as e:
        logger.error(f"Value error in numerical calculation: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "success": False,
            "error": f"Invalid input data: {str(e)}"
        }, status=400)
    except Exception as e:
        logger.error(f"Unexpected error in numerical calculation: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }, status=500)

@csrf_exempt
@require_POST
def calculate_schwarzschild_christoffel(request):
    """
    API endpoint for calculating Christoffel symbols for Schwarzschild metric.
    Returns the symbols in both array and textual formats.
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        G = data.get('G', 6.67430e-11)  # Gravitational constant
        M = data.get('M', 1.0)  # Mass
        c = data.get('c', 299792458.0)  # Speed of light
        
        logger.info(f"Calculating Schwarzschild Christoffel symbols with G={G}, M={M}, c={c}")
        
        # Calculate Christoffel symbols for Schwarzschild metric
        # Create the full 4×4×4 array structure (0-3 for t,r,θ,φ)
        christoffel_array = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
        
        # Fill in the non-zero components
        # Γ^t_tr = Γ^t_rt
        christoffel_array[0][0][1] = f"G*M/(-2*G*M*r + c**2*r**2)"
        christoffel_array[0][1][0] = f"G*M/(-2*G*M*r + c**2*r**2)"
        
        # Γ^r_tt
        christoffel_array[1][0][0] = f"(-2*G**2*M**2 + G*M*c**2*r)/(c**2*r**3)"
        
        # Γ^r_rr
        christoffel_array[1][1][1] = f"-G*M/(-2*G*M*r + c**2*r**2)"
        
        # Γ^r_θθ
        christoffel_array[1][2][2] = f"2*G*M/c**2 - r"
        
        # Γ^r_φφ
        christoffel_array[1][3][3] = f"2*G*M*sin(theta)**2/c**2 - r*sin(theta)**2"
        
        # Γ^θ_rθ = Γ^θ_θr
        christoffel_array[2][1][2] = f"1/r"
        christoffel_array[2][2][1] = f"1/r"
        
        # Γ^θ_φφ
        christoffel_array[2][3][3] = f"-sin(2*theta)/2"
        
        # Γ^φ_rφ = Γ^φ_φr
        christoffel_array[3][1][3] = f"1/r"
        christoffel_array[3][3][1] = f"1/r"
        
        # Γ^φ_θφ = Γ^φ_φθ
        christoffel_array[3][2][3] = f"1/tan(theta)"
        christoffel_array[3][3][2] = f"1/tan(theta)"
        
        # Create textual representation
        textual_symbols = [
            "Γ^(0)_(01) = G*M/(-2*G*M*r + c**2*r**2)",
            "Γ^(0)_(10) = G*M/(-2*G*M*r + c**2*r**2)",
            "Γ^(1)_(00) = (-2*G**2*M**2 + G*M*c**2*r)/(c**2*r**3)",
            "Γ^(1)_(11) = -G*M/(-2*G*M*r + c**2*r**2)",
            "Γ^(1)_(22) = 2*G*M/c**2 - r",
            "Γ^(1)_(33) = 2*G*M*sin(theta)**2/c**2 - r*sin(theta)**2",
            "Γ^(2)_(12) = 1/r",
            "Γ^(2)_(21) = 1/r",
            "Γ^(2)_(33) = -sin(2*theta)/2",
            "Γ^(3)_(13) = 1/r",
            "Γ^(3)_(31) = 1/r",
            "Γ^(3)_(23) = 1/tan(theta)",
            "Γ^(3)_(32) = 1/tan(theta)"
        ]
        
        return JsonResponse({
            'success': True,
            'christoffel_array': christoffel_array,
            'christoffel_textual': textual_symbols,
            'metadata': {
                'G': G,
                'M': M,
                'c': c,
                'metric_type': 'Schwarzschild'
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON format'}, status=400)
    except Exception as e:
        logger.error(f"Error in Schwarzschild calculation: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({'success': False, 'error': str(e)}, status=500) 