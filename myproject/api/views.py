from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
import sympy as sp
import numpy as np
import traceback
import logging

logger = logging.getLogger(__name__)

@api_view(['POST'])
def differential_operators(request):
    """
    Calculate differential operators using pre-calculated tensors.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            calculation_mode = data.get('calculation_mode', 'symbolic')
            metric = data.get('metric', [])
            christoffel_symbols = data.get('christoffel_symbols', [])
            coordinates = data.get('coordinates', [])
            vector_field = data.get('vector_field', [])
            scalar_field = data.get('scalar_field', '')
            selected_operators = data.get('selected_operators', [])
            enable_consistency_checks = data.get('enable_consistency_checks', False)
            
            logger.info(f"Processing differential operators request: mode={calculation_mode}, operators={selected_operators}")
            
            # Initialize result dictionary
            result = {}
            
            # Choose between symbolic and numeric calculation modes
            if calculation_mode == 'symbolic':
                # Import symbolic modules
                from myproject.utils.differential_operators.symbolic import (
                    gradient, 
                    divergence, 
                    curl, 
                    laplacian, 
                    covariant_derivative
                )
                from myproject.utils.differential_operators.consistency_checks import (
                    check_christoffel_symmetry,
                    check_metric_compatibility
                )
                
                # Convert inputs to sympy objects for symbolic calculations
                try:
                    # Convert coordinates to sympy symbols
                    coord_symbols = sp.symbols(coordinates)
                    
                    # Convert scalar field to sympy expression if provided
                    if scalar_field:
                        scalar_expr = sp.sympify(scalar_field)
                    else:
                        scalar_expr = None
                        
                    # Convert vector field components to sympy expressions if provided
                    vector_expr = []
                    if vector_field and any(vector_field):
                        for component in vector_field:
                            if component and component.strip():
                                vector_expr.append(sp.sympify(component))
                            else:
                                vector_expr.append(sp.S.Zero)  # SymPy zero
                    
                    # Convert metric to sympy matrix
                    metric_matrix = sp.Matrix(metric)
                    
                    # Process selected operators
                    if 'gradient' in selected_operators and scalar_expr:
                        grad_result = gradient(scalar_expr, coordinates, metric_matrix, christoffel_symbols)
                        result['gradient'] = [str(comp) for comp in grad_result]
                        
                    if 'divergence' in selected_operators and vector_expr:
                        div_result = divergence(vector_expr, coordinates, metric_matrix, christoffel_symbols)
                        result['divergence'] = str(div_result)
                        
                    if 'curl' in selected_operators and vector_expr:
                        # Curl only makes sense in 3D
                        if len(coordinates) != 3:
                            result['curl_error'] = "Curl operation requires 3D space"
                        else:
                            curl_result = curl(vector_expr, coordinates, metric_matrix, christoffel_symbols)
                            result['curl'] = [str(comp) for comp in curl_result]
                        
                    if 'laplacian' in selected_operators and scalar_expr:
                        lap_result = laplacian(scalar_expr, coordinates, metric_matrix, christoffel_symbols)
                        result['laplacian'] = str(lap_result)
                        
                    if 'covariant-derivative' in selected_operators and vector_expr:
                        cov_result = covariant_derivative(vector_expr, coordinates, metric_matrix, christoffel_symbols)
                        # Convert 2D array of sympy expressions to 2D array of strings
                        result['covariantDerivative'] = [[str(comp) for comp in row] for row in cov_result]
                    
                    # Perform consistency checks if enabled
                    if enable_consistency_checks:
                        consistency_checks = {}
                        consistency_checks['christoffelSymmetry'] = check_christoffel_symmetry(christoffel_symbols)
                        consistency_checks['metricCompatibility'] = check_metric_compatibility(
                            metric_matrix, christoffel_symbols, coord_symbols
                        )
                        result['consistencyChecks'] = consistency_checks
                        
                except Exception as e:
                    logger.error(f"Error in symbolic calculations: {str(e)}")
                    logger.error(traceback.format_exc())
                    result['error'] = f"Symbolic calculation error: {str(e)}"
                    
            else:  # Numeric mode
                # Import numeric modules
                from myproject.utils.differential_operators.numeric import (
                    evaluate_gradient, 
                    evaluate_divergence, 
                    evaluate_curl, 
                    evaluate_laplacian
                )
                
                try:
                    # Convert inputs to numeric arrays for calculations
                    
                    # Convert metric to numpy array
                    metric_array = np.array(metric, dtype=float)
                    
                    # Create an evaluation point (using point [1,1,1,...] for now)
                    # In a complete implementation, this would come from the request
                    eval_point = np.ones(len(coordinates))
                    
                    # Create a small grid around the evaluation point
                    # This is a simplified approach for demonstration
                    grid = [np.linspace(p * 0.9, p * 1.1, 3) for p in eval_point]
                    
                    # Convert scalar field to numpy function if provided
                    if scalar_field:
                        # Parse and evaluate the scalar field expression
                        scalar_func = lambda *args: eval(scalar_field, 
                                                      {coord: arg for coord, arg in zip(coordinates, args)})
                        # Create a grid function
                        scalar_grid = np.zeros([3] * len(coordinates))
                    else:
                        scalar_func = None
                        scalar_grid = None
                        
                    # Convert vector field components to numpy functions if provided
                    vector_func = []
                    vector_grid = []
                    if vector_field and any(vector_field):
                        for component in vector_field:
                            if component and component.strip():
                                vector_func.append(
                                    lambda *args, expr=component: eval(expr, 
                                                              {coord: arg for coord, arg in zip(coordinates, args)})
                                )
                                vector_grid.append(np.zeros([3] * len(coordinates)))
                            else:
                                vector_func.append(lambda *args: 0)
                                vector_grid.append(np.zeros([3] * len(coordinates)))
                    
                    # Calculate inverse metric
                    metric_inverse = np.linalg.inv(metric_array)
                    
                    # Process selected operators
                    if 'gradient' in selected_operators and scalar_func:
                        # Simplified implementation for now
                        grad_result = [f"∂({scalar_field})/∂{coord}" for coord in coordinates]
                        result['gradient'] = grad_result
                        
                    if 'divergence' in selected_operators and vector_func:
                        # Simplified implementation for now
                        div_terms = [f"∂({comp})/∂{coord}" for comp, coord in zip(vector_field, coordinates)]
                        result['divergence'] = " + ".join(div_terms)
                        
                    if 'curl' in selected_operators and vector_func:
                        # Curl only makes sense in 3D
                        if len(coordinates) != 3:
                            result['curl_error'] = "Curl operation requires 3D space"
                        else:
                            # Simplified implementation for now
                            result['curl'] = [
                                f"∂({vector_field[2]})/∂{coordinates[1]} - ∂({vector_field[1]})/∂{coordinates[2]}",
                                f"∂({vector_field[0]})/∂{coordinates[2]} - ∂({vector_field[2]})/∂{coordinates[0]}",
                                f"∂({vector_field[1]})/∂{coordinates[0]} - ∂({vector_field[0]})/∂{coordinates[1]}"
                            ]
                        
                    if 'laplacian' in selected_operators and scalar_func:
                        # Simplified implementation for now
                        lap_terms = [f"∂²({scalar_field})/∂{coord}²" for coord in coordinates]
                        result['laplacian'] = " + ".join(lap_terms)
                        
                    if 'covariant-derivative' in selected_operators and vector_func:
                        # Simplified implementation for now
                        cov_result = [[f"∂({vf})/∂{coord}" for vf in vector_field] for coord in coordinates]
                        result['covariantDerivative'] = cov_result
                    
                except Exception as e:
                    logger.error(f"Error in numeric calculations: {str(e)}")
                    logger.error(traceback.format_exc())
                    result['error'] = f"Numeric calculation error: {str(e)}"
            
            return Response(result)
        except Exception as e:
            logger.error(f"General error in differential operators endpoint: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({'error': str(e)}, status=400)
    else:
        return Response({'error': 'Only POST requests are allowed'}, status=405) 