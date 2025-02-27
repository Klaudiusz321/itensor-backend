from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import logging
from myproject.utilis.calcualtion import (
    oblicz_tensory, 
    compute_einstein_tensor, 
    wczytaj_metryke_z_tekstu
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
    else:
        return str(obj)

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
            
            # Przygotuj wynik w formacie zgodnym z frontendem
            result = {
                'result': {
                    'coordinates': [str(coord) for coord in wspolrzedne],
                    'parameters': [str(param) for param in parametry],
                    'metric': [f"g_{{{i}{j}}} = {convert_sympy_obj(g[i,j])}" 
                             for i in range(n) for j in range(n) 
                             if g[i,j] != 0],
                    'christoffel': [f"\\Gamma^{{{k}}}_{{{i}{j}}} = {convert_sympy_obj(Gamma[k][i][j])}"
                                  for k in range(n) 
                                  for i in range(n) 
                                  for j in range(n) 
                                  if Gamma[k][i][j] != 0],
                    'riemann': [f"R_{{{a}{b}{c}{d}}} = {convert_sympy_obj(R_abcd[a][b][c][d])}"
                               for a in range(n) 
                               for b in range(n) 
                               for c in range(n) 
                               for d in range(n) 
                               if R_abcd[a][b][c][d] != 0],
                    'ricci': [f"R_{{{i}{j}}} = {convert_sympy_obj(Ricci[i,j])}"
                             for i in range(n) 
                             for j in range(n) 
                             if Ricci[i,j] != 0],
                    'einstein': [f"G_{{{i}{j}}} = {convert_sympy_obj(G_lower[i,j])}"
                                for i in range(n) 
                                for j in range(n) 
                                if G_lower[i,j] != 0],
                    'scalar': [f"R = {convert_sympy_obj(Scalar_Curvature)}"],
                    'g': g.tolist(),
                    'Gamma': Gamma,
                    'R_abcd': R_abcd,
                    'Ricci': Ricci.tolist(),
                    'status': 'completed'
                }
            }
            
            return JsonResponse(result)
            
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

