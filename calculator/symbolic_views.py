import json
import logging
import traceback
import sympy as sp
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from myproject.utils.symbolic import oblicz_tensory, compute_einstein_tensor, compute_weyl_tensor, generate_output

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def symbolic_calculation_view(request):
    """
    API endpoint for symbolic tensor calculations.
    
    Expected JSON input format:
    {
        "dimension": 4,
        "coordinates": ["t", "r", "θ", "φ"],
        "metric": [
            ["-1", "0", "0", "0"],
            ["0", "1/(1-2*M/r)", "0", "0"],
            ["0", "0", "r**2", "0"],
            ["0", "0", "0", "r**2 * sin(θ)**2"]
        ],
        "calculations": ["christoffel_symbols", "riemann_tensor", "ricci_tensor", "ricci_scalar"]
    }
    
    Returns:
    {
        "success": true,
        "message": "Symbolic calculations completed",
        "metric_components": { ... },
        "christoffel_symbols": { ... },
        "riemann_tensor": { ... },
        "ricci_tensor": { ... },
        "ricci_scalar": "..."
    }
    """
    try:
        # Log request details
        logger.info(f"Received symbolic calculation request from {request.META.get('REMOTE_ADDR')}")
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
        
        # Check for alternate payload format from frontend
        if "matrix" in data and "metric" not in data:
            logger.info("Converting 'matrix' field to 'metric' field")
            data["metric"] = data["matrix"]
        
        if "metric_type" in data:
            logger.info(f"Metric type provided: {data['metric_type']}")
        
        # Validate required fields
        required_fields = ["metric"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return JsonResponse({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }, status=400)
        
        # Extract data
        metric_data = data["metric"]
        dimension = data.get("dimension", len(metric_data))
        coordinates = data.get("coordinates", ["t", "r", "θ", "φ"][:dimension])
        calculations = data.get("calculations", ["all"])
        
        logger.info(f"Dimension: {dimension}")
        logger.info(f"Coordinates: {coordinates}")
        logger.info(f"Metric data shape: {len(metric_data)}x{len(metric_data[0]) if metric_data and metric_data[0] else 0}")
        logger.info(f"Requested calculations: {calculations}")
        
        # Validate dimension consistency
        if len(coordinates) != dimension:
            coordinates = coordinates[:dimension]
            logger.warning(f"Coordinates list truncated to match dimension {dimension}")
        
        if len(metric_data) != dimension or any(len(row) != dimension for row in metric_data):
            logger.error(f"Dimension mismatch: metric dimensions {len(metric_data)}x{len(metric_data[0]) if metric_data else 0} don't match specified dimension {dimension}x{dimension}")
            return JsonResponse({
                "success": False,
                "error": f"Metric dimensions ({len(metric_data)}x{len(metric_data[0]) if metric_data else 0}) do not match the specified dimension ({dimension}x{dimension})"
            }, status=400)
        
        # Convert metric data to sympy symbols
        try:
            logger.info("Converting metric data to sympy symbols")
            
            # Create symbol dictionary for variables
            symbols_dict = {}
            for coord in coordinates:
                symbols_dict[coord] = sp.Symbol(coord)
            
            # Add common symbols used in GR
            for sym in ['M', 'a', 'b', 'c', 'r', 'theta', 'phi', 'sin', 'cos', 'tan', 'exp', 'log']:
                if sym not in symbols_dict:
                    if sym in ['sin', 'cos', 'tan', 'exp', 'log']:
                        symbols_dict[sym] = getattr(sp, sym)
                    else:
                        symbols_dict[sym] = sp.Symbol(sym)
            
            # Convert metric string expressions to sympy expressions
            metric_dict = {}
            for i in range(dimension):
                for j in range(dimension):
                    try:
                        expr = metric_data[i][j]
                        if expr:  # Skip empty strings
                            metric_dict[(i, j)] = sp.sympify(expr, locals=symbols_dict)
                    except Exception as e:
                        logger.error(f"Error parsing metric component [{i}][{j}]: {expr}, Error: {str(e)}")
                        return JsonResponse({
                            "success": False,
                            "error": f"Error parsing metric component [{i}][{j}]: {expr}, Error: {str(e)}"
                        }, status=400)
            
            # Perform tensor calculations
            logger.info("Starting symbolic tensor calculations")
            try:
                g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coordinates, metric_dict)
                
                # Calculate Einstein tensor
                g_inv = g.inv()
                G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, dimension)
                
                # Calculate Weyl tensor if dimension > 3
                Weyl = None
                if dimension > 3:
                    Weyl = compute_weyl_tensor(R_abcd, Ricci, Scalar_Curvature, g, dimension)
                
                # Generate LaTeX output
                logger.info("Generating LaTeX output")
                latex_output = generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, dimension, Weyl)
                
                # Prepare response data
                response = {
                    "success": True,
                    "message": "Symbolic calculations completed successfully",
                    "latex_output": latex_output,
                    "metric_components": {},
                    "christoffel_symbols": {},
                    "riemann_tensor": {},
                    "ricci_tensor": {},
                    "einstein_tensor": {},
                    "scalar_curvature": str(Scalar_Curvature),
                    "dimension": dimension,
                    "coordinates": coordinates
                }
                
                # Fill in metric components
                for i in range(dimension):
                    for j in range(dimension):
                        response["metric_components"][f"{i},{j}"] = str(g[i, j])
                
                # Fill in Christoffel symbols
                for i in range(dimension):
                    for j in range(dimension):
                        for k in range(dimension):
                            if Gamma[i][j][k] != 0:
                                response["christoffel_symbols"][f"{i},{j},{k}"] = str(Gamma[i][j][k])
                
                # Fill in Riemann tensor components
                for i in range(dimension):
                    for j in range(dimension):
                        for k in range(dimension):
                            for l in range(dimension):
                                if R_abcd[i][j][k][l] != 0:
                                    response["riemann_tensor"][f"{i},{j},{k},{l}"] = str(R_abcd[i][j][k][l])
                
                # Fill in Ricci tensor components
                for i in range(dimension):
                    for j in range(dimension):
                        response["ricci_tensor"][f"{i},{j}"] = str(Ricci[i, j])
                
                # Fill in Einstein tensor components
                for i in range(dimension):
                    for j in range(dimension):
                        response["einstein_tensor"][f"{i},{j}"] = str(G_lower[i, j])
                
                # Add Weyl tensor if calculated
                if Weyl:
                    response["weyl_tensor"] = {}
                    for i in range(dimension):
                        for j in range(dimension):
                            for k in range(dimension):
                                for l in range(dimension):
                                    if Weyl[i][j][k][l] != 0:
                                        response["weyl_tensor"][f"{i},{j},{k},{l}"] = str(Weyl[i][j][k][l])
                
                logger.info("Returning successful symbolic calculation response")
                return JsonResponse(response)
                
            except Exception as calc_error:
                logger.error(f"Calculation error: {str(calc_error)}")
                logger.error(traceback.format_exc())
                return JsonResponse({
                    "success": False,
                    "error": f"Calculation error: {str(calc_error)}"
                }, status=500)
                
        except Exception as e:
            logger.error(f"Error converting metric data: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({
                "success": False,
                "error": f"Error converting metric data: {str(e)}"
            }, status=400)
            
    except Exception as e:
        logger.error(f"Unexpected error in symbolic calculation: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }, status=500) 