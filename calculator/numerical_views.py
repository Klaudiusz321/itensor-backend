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