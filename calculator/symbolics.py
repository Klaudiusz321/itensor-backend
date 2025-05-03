import logging
import sympy as sp
import traceback
from myproject.utils.symbolic.compute_tensor import oblicz_tensory, compute_einstein_tensor
from myproject.utils.symbolic.simplification.custom_simplify import custom_simplify
from .models import Tensor
from django.db import OperationalError

logger = logging.getLogger(__name__)

def compute_symbolic(dimension, coords, metric, evaluation_point=None):
    """
    Compute symbolic tensor quantities for the given metric.
    Cache results for reuse to avoid redundant calculations.
    
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
        
        # Check for cached results - but don't fail if caching is unavailable
        try:
            # Generate hash for this calculation to check cache
            metric_hash = Tensor.generate_metric_hash(dimension, coords, metric)
            logger.info(f"Generated metric hash: {metric_hash}")
            
            # Check if we already have cached results for this metric
            cached_tensor = Tensor.objects.filter(metric_hash=metric_hash).first()
            if cached_tensor and hasattr(cached_tensor, 'christoffel_symbols'):
                logger.info(f"Found cached result for metric with hash {metric_hash}")
                
                # Prepare the cached result
                result = {
                    "success": True,
                    "coordinates": coords,
                    "dimension": dimension,
                    "christoffelSymbols": cached_tensor.christoffel_symbols,
                    "riemannTensor": cached_tensor.riemann_tensor,
                    "ricciTensor": cached_tensor.ricci_tensor,
                    "scalarCurvature": cached_tensor.scalar_curvature,
                    "einsteinTensor": cached_tensor.einstein_tensor,
                    "weylTensor": [],
                    "cached": True,
                    "rawData": {
                        "dimension": dimension,
                        "coordinates": coords,
                        "tensors": {
                            "christoffel": cached_tensor.christoffel_symbols,
                            "riemann": cached_tensor.riemann_tensor,
                            "ricci": cached_tensor.ricci_tensor,
                            "scalar_curvature": cached_tensor.scalar_curvature,
                            "einstein": cached_tensor.einstein_tensor
                        }
                    }
                }
                
                # If evaluation point is provided, evaluate the tensors
                if evaluation_point:
                    result = evaluate_at_point(result, coords, evaluation_point)
                    
                return result
        except (AttributeError, OperationalError, Exception) as e:
            # If any error occurs during cache lookup, just log it and continue with calculation
            logger.warning(f"Cache lookup failed, will calculate from scratch: {str(e)}")
        
        logger.info("No cached result found or caching unavailable, computing tensors...")
        
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
        
        # Compute Einstein tensor
        g_inv = g.inv()
        G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, dimension)
        
        # Format results
        n = dimension
        
        # Format Christoffel symbols in the format expected by frontend
        christoffel_symbols = []
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    value = Gamma[a][b][c]
                    if value != 0:
                        simplified = custom_simplify(value)
                        if simplified != 0:
                            christoffel_symbols.append({
                                "indices": f"{a}_{{{b}{c}}}",
                                "value": str(simplified)
                            })
        
        # Format Riemann tensor components
        riemann_tensor = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        value = R_abcd[i][j][k][l]
                        if value != 0:
                            simplified = custom_simplify(value)
                            if simplified != 0:
                                riemann_tensor.append({
                                    "indices": f"R_{{{i}{j}{k}{l}}}",
                                    "value": str(simplified)
                                })
        
        # Format Ricci tensor components
        ricci_tensor = []
        for i in range(n):
            for j in range(n):
                value = Ricci[i, j]
                if value != 0:
                    simplified = custom_simplify(value)
                    if simplified != 0:
                        ricci_tensor.append({
                            "indices": f"R_{{{i}{j}}}",
                            "value": str(simplified)
                        })
        
        # Format Einstein tensor components
        einstein_tensor = []
        for i in range(n):
            for j in range(n):
                value = G_lower[i, j]
                if value != 0:
                    simplified = custom_simplify(value)
                    if simplified != 0:
                        einstein_tensor.append({
                            "indices": f"G_{{{i}{j}}}",
                            "value": str(simplified)
                        })
        
        # Format Weyl tensor (placeholder - would need to implement actual calculation)
        weyl_tensor = []
        
        # Try to cache the results in database, but don't fail if caching is unavailable
        try:
            name = f"Metric in {', '.join(coords)} coordinates"
            if coords[0] == 't' and coords[1] == 'r' and any('theta' in c for c in coords):
                # Try to identify common metrics
                if '2*M/r' in str(metric) and 'sin(theta)' in str(metric):
                    name = "Schwarzschild metric"
                elif 'a(t)' in str(metric) and ('sin(theta)' in str(metric) or 'sin(psi)' in str(metric)):
                    name = "FLRW metric"
            
            # Create a simple tensor object if caching is unavailable
            try:
                # Create and save the tensor with full caching fields
                tensor = Tensor(
                    name=name,
                    metric_hash=Tensor.generate_metric_hash(dimension, coords, metric),
                    dimension=dimension,
                    coordinates=coords,
                    metric_data=metric,
                    christoffel_symbols=christoffel_symbols,
                    riemann_tensor=riemann_tensor,
                    ricci_tensor=ricci_tensor,
                    scalar_curvature=str(Scalar_Curvature),
                    einstein_tensor=einstein_tensor
                )
                tensor.save()
                logger.info(f"Saved tensor calculation to database")
            except OperationalError as e:
                # If missing columns, fall back to the basic model fields
                logger.warning(f"Caching with full fields failed, trying minimal save: {str(e)}")
                tensor = Tensor(
                    name=name,
                    components={"info": "Basic save without caching support"},
                    description=f"Tensor calculation for {coords} coordinates"
                )
                tensor.save()
                logger.info("Saved basic tensor info (caching not available)")
        except Exception as e:
            logger.error(f"Error caching results: {str(e)}")
            # Don't fail the whole calculation if caching fails
        
        # Prepare the final result in the format expected by the frontend
        result = {
            "success": True,
            "coordinates": coords,
            "dimension": dimension,
            "christoffelSymbols": christoffel_symbols,
            "riemannTensor": riemann_tensor,
            "ricciTensor": ricci_tensor,
            "scalarCurvature": str(Scalar_Curvature),
            "einsteinTensor": einstein_tensor,
            "weylTensor": weyl_tensor,
            "cached": False,
            "rawData": {
                "dimension": dimension,
                "coordinates": coords,
                "tensors": {
                    "christoffel": christoffel_symbols,
                    "riemann": riemann_tensor,
                    "ricci": ricci_tensor,
                    "scalar_curvature": str(Scalar_Curvature),
                    "einstein": einstein_tensor
                }
            }
        }
        
        # Evaluate at a specific point if provided
        if evaluation_point:
            result = evaluate_at_point(result, coords, evaluation_point)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in symbolic computation: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Error in symbolic computation: {str(e)}"}

def evaluate_at_point(result, coords, evaluation_point):
    """Helper function to evaluate symbolic results at a specific point"""
    try:
        substitutions = {}
        for coord, value in evaluation_point.items():
            if coord in coords:
                symbol = sp.Symbol(coord)
                substitutions[symbol] = value
        
        # Evaluate scalar curvature
        scalar_expr = sp.sympify(result["scalarCurvature"])
        evaluated_scalar = scalar_expr.subs(substitutions)
        
        # Add evaluated results
        result["evaluated"] = {
            "evaluationPoint": evaluation_point,
            "scalarCurvature": float(evaluated_scalar) if evaluated_scalar.is_Number else str(evaluated_scalar)
        }
        
        return result
    except Exception as e:
        logger.error(f"Error evaluating at point: {str(e)}")
        result["evaluated"] = {"error": f"Error evaluating at point: {str(e)}"}
        return result 