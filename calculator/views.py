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
        sections = {
            'metric': [],
            'christoffel': [],
            'riemann': [],
            'ricci': [],
            'einstein': [],
            'scalar': []
        }
        
        current_section = None
        lines = output_text.split('\n')
        
        latex_sections = {
            'christoffelLatex': [],
            'riemannLatex': [],
            'ricciLatex': [],
            'einsteinLatex': []
        }
        
        for line in lines:
            if '\\(' in line:
                latex_content = line[line.find('\\('):line.find('\\)')+2]
                if current_section == 'christoffel':
                    latex_sections['christoffelLatex'].append(latex_content)
                elif current_section == 'riemann':
                    latex_sections['riemannLatex'].append(latex_content)
                elif current_section == 'ricci':
                    latex_sections['ricciLatex'].append(latex_content)
                elif current_section == 'einstein':
                    latex_sections['einsteinLatex'].append(latex_content)
                
                if current_section:
                    sections[current_section].append(line.strip())
            
            if "Metric tensor components" in line:
                current_section = 'metric'
            elif "Christoffel symbols" in line:
                current_section = 'christoffel'
            elif "Riemann tensor" in line:
                current_section = 'riemann'
            elif "Ricci tensor" in line:
                current_section = 'ricci'
            elif "Einstein tensor" in line:
                current_section = 'einstein'
            elif "Scalar curvature" in line:
                current_section = 'scalar'

        result = {
            'metric': sections['metric'],
            'christoffel': sections['christoffel'],
            'riemann': sections['riemann'],
            'ricci': sections['ricci'],
            'einstein': sections['einstein'],
            'scalar': sections['scalar'],
            'coordinates': [str(coord) for coord in wspolrzedne],
            'parameters': [str(param) for param in parametry],
            'metryka': {f"{i},{j}": convert_sympy_obj(g[i,j]) 
                       for i in range(len(wspolrzedne)) 
                       for j in range(len(wspolrzedne))},
            'scalarCurvature': convert_sympy_obj(Scalar_Curvature),
            'scalarCurvatureLatex': f"\\({sp.latex(Scalar_Curvature)}\\)",
            'christoffelLatex': latex_sections['christoffelLatex'],
            'riemannLatex': latex_sections['riemannLatex'],
            'ricciLatex': latex_sections['ricciLatex'],
            'einsteinLatex': latex_sections['einsteinLatex'],
            'outputText': output_text,
            'g': convert_sympy_obj(g),
            'Gamma': convert_sympy_obj(Gamma),
            'R_abcd': convert_sympy_obj(R_abcd),
            'Ricci': convert_sympy_obj(Ricci),
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

