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

def parse_metric_output(output_text: str, g, Gamma, R_abcd, Ricci, Scalar_Curvature, wspolrzedne, parametry) -> dict:
    try:
        if not output_text:
            raise ValueError("Empty output text")
            
        sections = {
            'metric': [],
            'christoffel': [],
            'riemann': [],
            'ricci': [],
            'einstein': [],
            'scalar': []
        }
        
        # Parsowanie sekcji z output_text
        current_section = None
        lines = output_text.split('\n')
        
        latex_sections = {
            'christoffelLatex': [],
            'riemannLatex': [],
            'ricciLatex': [],
            'einsteinLatex': []
        }
        
        for line in lines:
            if '\\(' in line:  # Linia zawiera LaTeX
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
            
            # Aktualizacja current_section
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

        # Przygotowanie danych zgodnie z interfejsem MetricData
        result = {
            'metric': sections['metric'],
            'christoffel': sections['christoffel'],
            'riemann': sections['riemann'],
            'ricci': sections['ricci'],
            'einstein': sections['einstein'],
            'scalar': sections['scalar'],
            'coordinates': [str(coord) for coord in wspolrzedne],
            'parameters': [str(param) for param in parametry],
            'metryka': {f"{i},{j}": str(g[i,j]) for i in range(len(wspolrzedne)) for j in range(len(wspolrzedne))},
            'scalarCurvature': str(Scalar_Curvature),
            'scalarCurvatureLatex': f"\\({sp.latex(Scalar_Curvature)}\\)",
            'christoffelLatex': latex_sections['christoffelLatex'],
            'riemannLatex': latex_sections['riemannLatex'],
            'ricciLatex': latex_sections['ricciLatex'],
            'einsteinLatex': latex_sections['einsteinLatex'],
            'outputText': output_text,
            'g': g.tolist() if hasattr(g, 'tolist') else [[str(g[i,j]) for j in range(len(wspolrzedne))] for i in range(len(wspolrzedne))],
            'Gamma': [[[str(Gamma[i][j][k]) for k in range(len(wspolrzedne))] for j in range(len(wspolrzedne))] for i in range(len(wspolrzedne))],
            'R_abcd': [[[[str(R_abcd[i][j][k][l]) for l in range(len(wspolrzedne))] for k in range(len(wspolrzedne))] for j in range(len(wspolrzedne))] for i in range(len(wspolrzedne))],
            'Ricci': Ricci.tolist() if hasattr(Ricci, 'tolist') else [[str(Ricci[i,j]) for j in range(len(wspolrzedne))] for i in range(len(wspolrzedne))],
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
        logger.info("=== Starting calculate_view ===")
        
        # 1. Parsowanie JSON
        try:
            data = json.loads(request.body)
            logger.info(f"Received data: {data}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JsonResponse({
                'error': 'Invalid JSON format',
                'details': str(e)
            }, status=400)

        # 2. Sprawdzenie metric_text
        metric_text = data.get('metric_text')
        if not metric_text:
            logger.error("Missing metric_text")
            return JsonResponse({
                'error': 'Missing metric_text'
            }, status=400)

        logger.info(f"Processing metric_text:\n{metric_text}")

        # 3. Parsowanie metryki
        try:
            wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
            logger.info(f"Parsed coordinates: {[str(w) for w in wspolrzedne]}")
            logger.info(f"Parsed parameters: {[str(p) for p in parametry]}")
            logger.info(f"Parsed metric components: {len(metryka)} components")
        except Exception as e:
            logger.error(f"Metric parsing error: {e}", exc_info=True)
            return JsonResponse({
                'error': 'Metric parsing error',
                'details': str(e)
            }, status=400)

        # 4. Obliczenia tensorów
        try:
            n = len(wspolrzedne)
            logger.info(f"Starting tensor calculations for {n} dimensions")
            
            g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
            logger.info("Basic tensors calculated")
            
            # Sprawdź czy macierz metryczna jest odwracalna
            if g.det() == 0:
                raise ValueError("Metric tensor is singular (not invertible)")
            
            g_inv = g.inv()
            logger.info("Metric inverse calculated")
            
            G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n)
            logger.info("Einstein tensor calculated")
            
        except Exception as e:
            logger.error(f"Calculation error: {e}", exc_info=True)
            return JsonResponse({
                'error': 'Calculation error',
                'details': str(e)
            }, status=400)

        # 5. Generowanie wyniku
        try:
            output = generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n)
            result = parse_metric_output(
                output, 
                g, 
                Gamma, 
                R_abcd, 
                Ricci, 
                Scalar_Curvature,
                wspolrzedne,
                parametry
            )
            logger.info("Output generated successfully")
            return JsonResponse(result)
        except Exception as e:
            logger.error(f"Output generation error: {e}", exc_info=True)
            return JsonResponse({
                'error': 'Output generation error',
                'details': str(e)
            }, status=400)

    except Exception as e:
        logger.error(f"Unexpected server error: {e}", exc_info=True)
        return JsonResponse({
            'error': 'Server error',
            'details': str(e)
        }, status=500)

