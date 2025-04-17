from rest_framework.decorators import api_view
from rest_framework.response import Response
import json

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
            
            # Choose between symbolic and numeric calculation modes
            if calculation_mode == 'symbolic':
                from myproject.utils.differential_operators.symbolic import gradient, divergence, curl, laplacian, covariant_derivative
                from myproject.utils.tensor_utils import check_christoffel_symbol_symmetry, check_metric_compatibility
            else:
                from myproject.utils.differential_operators.numeric import gradient, divergence, curl, laplacian, covariant_derivative
            
            # Initialize result dictionary
            result = {}
            
            # Process selected operators
            if 'gradient' in selected_operators and scalar_field:
                result['gradient'] = gradient(metric, scalar_field, coordinates, christoffel_symbols)
            
            if 'divergence' in selected_operators and vector_field:
                result['divergence'] = divergence(metric, vector_field, coordinates, christoffel_symbols)
            
            if 'curl' in selected_operators and vector_field:
                # Curl only makes sense in 3D
                if len(coordinates) != 3:
                    result['error'] = "Curl operation requires 3D space"
                else:
                    result['curl'] = curl(metric, vector_field, coordinates, christoffel_symbols)
            
            if 'laplacian' in selected_operators and scalar_field:
                result['laplacian'] = laplacian(metric, scalar_field, coordinates, christoffel_symbols)
            
            if 'covariant-derivative' in selected_operators and vector_field:
                result['covariantDerivative'] = covariant_derivative(metric, vector_field, coordinates, christoffel_symbols)
            
            # Perform consistency checks if enabled (symbolic mode only)
            if enable_consistency_checks and calculation_mode == 'symbolic':
                consistency_checks = {}
                consistency_checks['christoffelSymmetry'] = check_christoffel_symbol_symmetry(christoffel_symbols)
                consistency_checks['metricCompatibility'] = check_metric_compatibility(metric, christoffel_symbols, coordinates)
                result['consistencyChecks'] = consistency_checks
            
            return Response(result)
        except Exception as e:
            return Response({'error': str(e)}, status=400)
    else:
        return Response({'error': 'Only POST requests are allowed'}, status=405) 