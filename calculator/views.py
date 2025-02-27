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

        # Filtrowanie niezerowych komponentów metryki
        n = len(wspolrzedne)
        metryka_dict = {}
        for i in range(n):
            for j in range(n):
                val = convert_sympy_obj(g[i,j])
                if val != 0 and val != "0":
                    metryka_dict[f"{i},{j}"] = val

        # Filtrowanie niezerowych komponentów Gamma
        gamma_nonzero = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    val = convert_sympy_obj(Gamma[i][j][k])
                    if val != 0 and val != "0":
                        gamma_nonzero.append({
                            'indices': [i,j,k],
                            'value': val
                        })

        # Filtrowanie niezerowych komponentów tensora Riemanna
        riemann_nonzero = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        val = convert_sympy_obj(R_abcd[i][j][k][l])
                        if val != 0 and val != "0":
                            riemann_nonzero.append({
                                'indices': [i,j,k,l],
                                'value': val
                            })

        # Filtrowanie niezerowych komponentów tensora Ricciego
        ricci_nonzero = {}
        for i in range(n):
            for j in range(n):
                val = convert_sympy_obj(Ricci[i,j])
                if val != 0 and val != "0":
                    ricci_nonzero[f"{i},{j}"] = val

        result = {
            'metric': sections['metric'],
            'christoffel': sections['christoffel'],
            'riemann': sections['riemann'],
            'ricci': sections['ricci'],
            'einstein': sections['einstein'],
            'scalar': sections['scalar'],
            'coordinates': [str(coord) for coord in wspolrzedne],
            'parameters': [str(param) for param in parametry],
            'metryka': metryka_dict,
            'scalarCurvature': convert_sympy_obj(Scalar_Curvature),
            'scalarCurvatureLatex': f"\\({sp.latex(Scalar_Curvature)}\\)",
            'christoffelLatex': latex_sections['christoffelLatex'],
            'riemannLatex': latex_sections['riemannLatex'],
            'ricciLatex': latex_sections['ricciLatex'],
            'einsteinLatex': latex_sections['einsteinLatex'],
            'outputText': output_text,
            'g': metryka_dict,
            'Gamma': gamma_nonzero,
            'R_abcd': riemann_nonzero,
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

