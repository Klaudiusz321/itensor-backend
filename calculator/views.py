from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import logging
from myproject.utilis.calcualtion import (
    oblicz_tensory, 
    compute_einstein_tensor, 
    wczytaj_metryke_z_tekstu, 
    generate_output
)
import sympy as sp

logger = logging.getLogger(__name__)

def convert_sympy_obj(obj):
    if hasattr(obj, 'free_symbols') and obj.free_symbols:
        return str(obj)
    elif hasattr(obj, 'evalf'):
        try:
            value = float(obj.evalf())
            if value.is_integer():
                return int(value)
            return value
        except Exception:
            return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_sympy_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_sympy_obj(i) for i in obj]
    elif isinstance(obj, sp.Matrix):
        return [[convert_sympy_obj(obj[i,j]) for j in range(obj.cols)] 
                for i in range(obj.rows)]
    else:
        return obj

def parse_metric_output(output_text: str, g, Gamma, R_abcd, Ricci, Scalar_Curvature, wspolrzedne, parametry) -> dict:
    try:
        # Filtrowanie niezerowych komponentów metryki
        n = len(wspolrzedne)
        metryka_dict = {}
        for i in range(n):
            for j in range(n):
                val = convert_sympy_obj(g[i,j])
                if val != 0 and val != "0":
                    metryka_dict[f"{i},{j}"] = val

        # Filtrowanie niezerowych komponentów tensora Ricciego
        ricci_nonzero = {}
        for i in range(n):
            for j in range(n):
                val = convert_sympy_obj(Ricci[i,j])
                if val != 0 and val != "0":
                    ricci_nonzero[f"{i},{j}"] = val

        result = {
            'coordinates': [str(coord) for coord in wspolrzedne],
            'parameters': [str(param) for param in parametry],
            'metryka': metryka_dict,
            'scalarCurvature': convert_sympy_obj(Scalar_Curvature),
            'Ricci': ricci_nonzero,
            'success': True
        }

        return result

    except Exception as e:
        logger.error(f"Output parsing error: {e}", exc_info=True)
        return {
            'error': 'Output parsing error',
            'details': str(e)
        }

@csrf_exempt
@require_POST
def calculate_view(request):
    try:
        data = json.loads(request.body)
        metric_text = data.get('metric_text')
        
        if not metric_text:
            return JsonResponse({'error': 'Missing metric_text'}, status=400)

        # Parsowanie metryki
        try:
            wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        except Exception as e:
            return JsonResponse({
                'error': f"Metric parsing error: {str(e)}"
            }, status=400)

        # Obliczenia tensorów
        try:
            n = len(wspolrzedne)
            g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
            
            if g.det() == 0:
                return JsonResponse({
                    'error': "Metric tensor is singular"
                }, status=400)
                
            g_inv = g.inv()
            G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n)
            
            output = generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n)
            result = parse_metric_output(
                output, g, Gamma, R_abcd, Ricci, Scalar_Curvature,
                wspolrzedne, parametry
            )
            
            if result.get('error'):
                return JsonResponse(result, status=400)
            
            return JsonResponse({
                'status': 'completed',
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': str(e)
            }, status=400)
        
    except Exception as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        return JsonResponse({
            'error': str(e)
        }, status=400)

