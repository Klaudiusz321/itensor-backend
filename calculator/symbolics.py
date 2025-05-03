import logging
import sympy as sp
import traceback
from myproject.utils.symbolic.compute_tensor import oblicz_tensory
from myproject.utils.symbolic.simplification.custom_simplify import custom_simplify

logger = logging.getLogger(__name__)

def compute_symbolic(dimension, coords, metric, evaluation_point=None):
    """
    Compute symbolic tensor quantities for the given metric.
    
    Args:
        dimension (int): Dimension of the space
        coords (list): List of coordinate symbols
        metric (list or dict): Either a 2D list/array of metric components or a dictionary with keys like "00", "01", etc.
        evaluation_point (dict, optional): Dictionary mapping coordinates to values for evaluation
        
    Returns:
        dict: Dictionary containing computed tensor quantities
    """
    try:
        logger.info(f"Computing symbolic tensors for {dimension}D metric")
        
        # Convert string coordinates to sympy symbols
        coord_symbols = []
        for coord in coords:
            if coord:
                coord_symbols.append(sp.Symbol(coord))
            else:
                return {"error": "Empty coordinate name provided"}
        
        # Convert metric to the format expected by oblicz_tensory
        metric_dict = {}
        
        # Check if metric is a 2D list/array
        if isinstance(metric, list):
            if len(metric) != dimension:
                return {"error": f"Metric dimensions don't match: expected {dimension}x{dimension}, got {len(metric)}x?"}
            
            for i, row in enumerate(metric):
                if len(row) != dimension:
                    return {"error": f"Metric row {i} has {len(row)} elements, expected {dimension}"}
                
                for j, value in enumerate(row):
                    try:
                        # Parse the expression string to a sympy expression
                        expr_str = str(value)
                        if expr_str == "0":
                            continue  # Skip zero components to save time
                        
                        expr = sp.sympify(expr_str)
                        metric_dict[(i, j)] = expr
                    except Exception as e:
                        return {"error": f"Could not parse metric expression at [{i}][{j}]: {value}. Error: {str(e)}"}
        
        # Check if metric is a dictionary
        elif isinstance(metric, dict):
            for key, value in metric.items():
                if len(key) != 2:
                    return {"error": f"Invalid metric key: {key}. Should be 2 digits."}
                
                try:
                    i, j = int(key[0]), int(key[1])
                    if i >= dimension or j >= dimension:
                        return {"error": f"Metric component {key} out of bounds for dimension {dimension}"}
                    
                    # Parse the expression string to a sympy expression
                    expr_str = str(value)
                    if expr_str == "0":
                        continue  # Skip zero components to save time
                    
                    expr = sp.sympify(expr_str)
                    metric_dict[(i, j)] = expr
                except ValueError:
                    return {"error": f"Invalid metric key format: {key}. Should be digits."}
        else:
            return {"error": "Metric must be either a 2D list/array or a dictionary"}
        
        # Make sure the metric is symmetric
        for i in range(dimension):
            for j in range(dimension):
                if (i, j) in metric_dict and (j, i) not in metric_dict:
                    metric_dict[(j, i)] = metric_dict[(i, j)]
        
        # Compute tensor quantities
        g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(coord_symbols, metric_dict)
        
        # Format results
        n = dimension
        
        # Format Christoffel symbols
        christoffel_components = {}
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    value = Gamma[a][b][c]
                    if value != 0:
                        simplified = custom_simplify(value)
                        if simplified != 0:
                            christoffel_components[f"{a}_{{{b}{c}}}"] = str(simplified)
        
        # Format Riemann tensor components
        riemann_components = {}
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        value = R_abcd[i][j][k][l]
                        if value != 0:
                            simplified = custom_simplify(value)
                            if simplified != 0:
                                riemann_components[f"R_{{{i}{j}{k}{l}}}"] = str(simplified)
        
        # Format Ricci tensor components
        ricci_components = {}
        for i in range(n):
            for j in range(n):
                value = Ricci[i, j]
                if value != 0:
                    simplified = custom_simplify(value)
                    if simplified != 0:
                        ricci_components[f"R_{{{i}{j}}}"] = str(simplified)
        
        # Evaluate at a specific point if provided
        evaluated_results = None
        if evaluation_point:
            try:
                substitutions = {}
                for coord, value in evaluation_point.items():
                    if coord in coords:
                        symbol = sp.Symbol(coord)
                        substitutions[symbol] = value
                
                # Evaluate tensor components at the specified point
                evaluated_results = {
                    "christoffel": {},
                    "riemann": {},
                    "ricci": {},
                    "scalar_curvature": None
                }
                
                # Evaluate Christoffel symbols
                for key, expr_str in christoffel_components.items():
                    expr = sp.sympify(expr_str)
                    evaluated = expr.subs(substitutions)
                    if evaluated != 0:
                        evaluated_results["christoffel"][key] = float(evaluated)
                
                # Evaluate Riemann tensor
                for key, expr_str in riemann_components.items():
                    expr = sp.sympify(expr_str)
                    evaluated = expr.subs(substitutions)
                    if evaluated != 0:
                        evaluated_results["riemann"][key] = float(evaluated)
                
                # Evaluate Ricci tensor
                for key, expr_str in ricci_components.items():
                    expr = sp.sympify(expr_str)
                    evaluated = expr.subs(substitutions)
                    if evaluated != 0:
                        evaluated_results["ricci"][key] = float(evaluated)
                
                # Evaluate scalar curvature
                evaluated_scalar = Scalar_Curvature.subs(substitutions)
                evaluated_results["scalar_curvature"] = float(evaluated_scalar)
            except Exception as e:
                logger.error(f"Error evaluating at point: {str(e)}")
                evaluated_results = {"error": f"Error evaluating at point: {str(e)}"}
        
        # Prepare the final result
        result = {
            "success": True,
            "dimension": dimension,
            "coordinates": coords,
            "tensors": {
                "christoffel": christoffel_components,
                "riemann": riemann_components,
                "ricci": ricci_components,
                "scalar_curvature": str(Scalar_Curvature)
            }
        }
        
        # Add evaluated results if available
        if evaluated_results:
            result["evaluated"] = evaluated_results
        
        return result
        
    except Exception as e:
        logger.error(f"Error in symbolic computation: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Error in symbolic computation: {str(e)}"} 