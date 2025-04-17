import json
import logging
import traceback
import sympy as sp
import re
import inspect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from myproject.utils.symbolic import oblicz_tensory, compute_einstein_tensor, compute_weyl_tensor, generate_output

logger = logging.getLogger(__name__)

def _fix_fn_expo(expr_str):
    """
    Fix function exponentiation syntax like "sin**2(theta)" to "sin(theta)**2".
    
    Uses regex to find patterns like "name**exponent(args)" and rewrites as "name(args)**exponent".
    """
    # Match pattern like sin**2(theta) and rewrite as sin(theta)**2
    pattern = r'([a-zA-Z]+)\*\*(\d+)\(([^)]*)\)'
    fixed_expr = re.sub(pattern, r'\1(\3)**\2', expr_str)
    
    # Log if we made a change
    if fixed_expr != expr_str:
        logging.debug(f"Fixed expression from '{expr_str}' to '{fixed_expr}'")
    
    return fixed_expr

@csrf_exempt
@require_POST
def symbolic_calculation_view(request):
    """
    API endpoint for symbolic tensor calculations.
    
    Expected JSON input format (new format):
    {
        "coordinates": ["r", "theta"],
        "parameters": { "R2": 9.0 },
        "metric": {
            "0,0": "R2",
            "0,1": "0",
            "1,0": "0",
            "1,1": "R2 * (sin(theta))**2"
        }
    }

    
    Alternative format (backward compatibility):
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
        "einstein_tensor": { ... },
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
        
        # Check for metric field
        if "metric" not in data:
            if "matrix" in data:
                # Convert 'matrix' field to 'metric' field
                logger.info("Converting 'matrix' field to 'metric' field")
                data["metric"] = data["matrix"]
            else:
                logger.error("Missing required field: metric")
                return JsonResponse({
                    "success": False,
                    "error": "Missing required field: metric"
                }, status=400)
                
        # Handle parameters from request
        parameters = data.get("parameters", {})
        logger.info(f"Parameters: {parameters}")
        
        # Create dictionary for symbolic calculations
        symbols_dict = {}
        
        # Add parameters to symbols dict
        for param_name, param_value in parameters.items():
            symbols_dict[param_name] = param_value
            logger.info(f"Added parameter {param_name} = {param_value}")
        
        # Handle different input formats
        metric_format = "matrix" if isinstance(data["metric"], list) else "dict"
        logger.info(f"Detected metric format: {metric_format}")
        
        # Extract coordinates
        coordinates = data.get("coordinates", [])
        
        # Determine dimension
        if metric_format == "matrix":
            # For matrix format, use length of matrix
            dimension = data.get("dimension", len(data["metric"]))
            if not coordinates:
                # Default coordinates for backward compatibility
                coordinates = ["t", "r", "θ", "φ"][:dimension]
        else:
            # For dict format, determine dimension from keys
            max_i = max_j = 0
            for key in data["metric"].keys():
                if ',' in key:
                    i, j = map(int, key.split(','))
                    max_i = max(max_i, i)
                    max_j = max(max_j, j)
            dimension = max(max_i, max_j) + 1
            
            # Ensure coordinates has correct length
            if len(coordinates) < dimension:
                logger.warning(f"Coordinates list length {len(coordinates)} is less than dimension {dimension}")
                # If no coordinates provided or too few, use defaults
                if not coordinates:
                    # For 2D spherical metric, default to r, theta
                    if dimension == 2:
                        coordinates = ["r", "theta"]
                    else:
                        coordinates = [f"x{i}" for i in range(dimension)]
                else:
                    # Pad with default coordinate names
                    coordinates.extend([f"x{i}" for i in range(len(coordinates), dimension)])
        
        logger.info(f"Using coordinates: {coordinates}")
        logger.info(f"Dimension: {dimension}")
        
        # Add coordinates to symbols_dict
        for coord in coordinates:
            symbols_dict[coord] = sp.Symbol(coord)
        
        # Add common symbols used in GR and trigonometric functions
        for sym in ['M', 'a', 'b', 'c', 'r', 'theta', 'phi', 'R', 'R2']:
            if sym not in symbols_dict:
                symbols_dict[sym] = sp.Symbol(sym)
        
        # Add trigonometric and other functions
        for func_name in ['sin', 'cos', 'tan', 'exp', 'log']:
            symbols_dict[func_name] = getattr(sp, func_name)
        
        # Convert metric to sympy expressions
        metric_dict = {}
        
        if metric_format == "matrix":
            # Matrix format: [[g00, g01, ...], [g10, g11, ...], ...]
            # Check dimension consistency
            if len(data["metric"]) != dimension:
                logger.error(f"Metric row count {len(data['metric'])} doesn't match dimension {dimension}")
                return JsonResponse({
                    "success": False,
                    "error": f"Metric dimensions mismatch: expected {dimension}x{dimension} but got {len(data['metric'])}x?"
                }, status=400)
                
            for i in range(dimension):
                row = data["metric"][i]
                if len(row) != dimension:
                    logger.error(f"Metric row {i} has {len(row)} elements, expected {dimension}")
                    return JsonResponse({
                        "success": False,
                        "error": f"Metric dimensions mismatch: expected {dimension}x{dimension} but row {i} has {len(row)} elements"
                    }, status=400)
                
                for j in range(dimension):
                    expr_str = row[j]
                    if not expr_str:  # Skip empty strings
                        continue
                    
                    try:
                        # Fix function exponent syntax
                        expr_str_fixed = _fix_fn_expo(expr_str)
                        logger.info(f"Original expression: {expr_str} -> Fixed: {expr_str_fixed}")
                        
                        # Convert string expression to sympy expression
                        metric_dict[(i, j)] = sp.sympify(expr_str_fixed, locals=symbols_dict)
                        logger.info(f"Parsed metric component [{i}][{j}]: {expr_str_fixed} -> {metric_dict[(i, j)]}")
                    except Exception as e:
                        logger.error(f"Error parsing metric component [{i}][{j}]: {expr_str_fixed}, Error: {str(e)}")
                        return JsonResponse({
                            "success": False,
                            "error": f"Error parsing metric component [{i}][{j}]: {expr_str_fixed}. {str(e)}"
                        }, status=400)
        else:
            # Dict format: {"0,0": "expr1", "0,1": "expr2", ...}
            for key, expr_str in data["metric"].items():
                if not expr_str:  # Skip empty strings
                    continue
                
                try:
                    if ',' in key:
                        i, j = map(int, key.split(','))
                    else:
                        # Try to parse as tuple string "(i,j)"
                        key = key.replace('(', '').replace(')', '').replace(' ', '')
                        i, j = map(int, key.split(','))
                    
                    # Check bounds
                    if i >= dimension or j >= dimension:
                        logger.error(f"Index out of bounds: ({i},{j}) for dimension {dimension}")
                        return JsonResponse({
                            "success": False,
                            "error": f"Index out of bounds: ({i},{j}) for dimension {dimension}"
                        }, status=400)
                    
                    # Fix function exponent syntax
                    expr_str_fixed = _fix_fn_expo(expr_str)
                    logger.info(f"Original expression: {expr_str} -> Fixed: {expr_str_fixed}")
                    
                    # Convert string expression to sympy expression
                    try:
                        metric_dict[(i, j)] = sp.sympify(expr_str_fixed, locals=symbols_dict)
                        # Apply symmetry
                        metric_dict[(j, i)] = metric_dict[(i, j)]
                        logger.info(f"Parsed metric component [{i}][{j}]: {expr_str_fixed} -> {metric_dict[(i, j)]}")
                    except Exception as e:
                        logger.error(f"Error parsing metric component {key}: {expr_str_fixed}, Error: {str(e)}")
                        return JsonResponse({
                            "success": False,
                            "error": f"Error parsing metric component {key}: {expr_str_fixed}. {str(e)}"
                        }, status=400)
                except Exception as e:
                    logger.error(f"Error parsing metric component {key}: {expr_str}, Error: {str(e)}")
                    return JsonResponse({
                        "success": False,
                        "error": f"Error parsing metric component {key}: {expr_str}. {str(e)}"
                    }, status=400)
        
        # Ensure all metric components are defined
        for i in range(dimension):
            for j in range(dimension):
                if (i, j) not in metric_dict:
                    if i == j:
                        # Diagonal defaults to 1
                        metric_dict[(i, j)] = 1
                    else:
                        # Off-diagonal defaults to 0
                        metric_dict[(i, j)] = 0
                        
        # Substitute parameter values
        for (i, j), expr in list(metric_dict.items()):
            for param, value in parameters.items():
                if param in str(expr):
                    metric_dict[(i, j)] = expr.subs(sp.Symbol(param), value)
            
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
                "ricci_scalar": str(Scalar_Curvature),
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
                                    
            logger.info("Symbolic calculation completed successfully")
            return JsonResponse(response)
            
        except Exception as e:
            logger.error(f"Error in tensor calculations: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({
                "success": False,
                "error": f"Error in tensor calculations: {str(e)}"
            }, status=500)
            
    except Exception as e:
        logger.error(f"Unexpected error in symbolic_calculation_view: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }, status=500) 