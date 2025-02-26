from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_POST
import json
from myproject.utilis.calcualtion import oblicz_tensory, compute_einstein_tensor, wczytaj_metryke_z_tekstu, generate_output, generate_numerical_curvature
from myproject.utilis.calcualtion.derivative import numeric_derivative, partial_derivative, total_derivative
import logging
from django.core.cache import cache
import time

logger = logging.getLogger(__name__)

def parse_metric_output(output_text: str) -> dict:
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
    
    for line in lines:
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
        elif line.strip() and current_section:
            # Dodaj tylko niepuste linie zawierające LaTeX
            if '\\(' in line:
                sections[current_section].append(line.strip())

    return {
        'metric': sections['metric'],
        'christoffel': sections['christoffel'],
        'riemann': sections['riemann'],
        'ricci': sections['ricci'],
        'einstein': sections['einstein'],
        'scalar': sections['scalar'],
        'success': True
    }

@csrf_exempt
@require_POST
def calculate_view(request):
    try:
        print("\n=== Starting calculate_view ===")
        
        # 1. Walidacja requestu
        if not request.body:
            return JsonResponse({
                'error': 'Empty request body'
            }, status=400)

        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            return JsonResponse({
                'error': 'Invalid JSON',
                'details': str(e)
            }, status=400)

        # 2. Walidacja metric_text
        metric_text = data.get('metric_text')
        if not metric_text or not isinstance(metric_text, str):
            return JsonResponse({
                'error': 'Missing or invalid metric_text'
            }, status=400)

        print(f"Received metric_text: {metric_text}")

        # 3. Parsowanie metryki
        try:
            wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        except ValueError as e:
            return JsonResponse({
                'error': 'Metric parsing error',
                'details': str(e)
            }, status=400)
        except Exception as e:
            print(f"Unexpected error in metric parsing: {e}")
            return JsonResponse({
                'error': 'Server error during metric parsing',
                'details': str(e)
            }, status=500)

        # 4. Obliczenia
        try:
            n = len(wspolrzedne)
            g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
            g_inv = g.inv()
            G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n)
        except Exception as e:
            print(f"Calculation error: {e}")
            return JsonResponse({
                'error': 'Calculation error',
                'details': str(e)
            }, status=400)

        # 5. Generowanie wyniku
        try:
            output = generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n)
            result = parse_metric_output(output)
            return JsonResponse(result)
        except Exception as e:
            print(f"Output generation error: {e}")
            return JsonResponse({
                'error': 'Output generation error',
                'details': str(e)
            }, status=500)

    except Exception as e:
        print(f"Unexpected server error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': 'Server error',
            'details': str(e)
        }, status=500)

def compute_full_tensors(metric_text):
    """
    Oblicza wszystkie tensory dla danej metryki.
    """
    try:
        wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        n = len(wspolrzedne)
        g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
        g_inv = g.inv()
        G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n)
        
        # Zwracamy dane w formacie odpowiednim dla transform_to_curvature_data
        return {
            'scalar_curvature': Scalar_Curvature,  # Wyrażenie symboliczne krzywizny skalarnej
            'coordinates': wspolrzedne,            # Lista symboli współrzędnych
            'parameters': parametry,               # Lista parametrów
            'metric_data': {                       # Dodatkowe dane metryczne
                'metric': g,
                'christoffel': Gamma,
                'riemann': R_abcd,
                'ricci': Ricci,
                'einstein_upper': G_upper,
                'einstein_lower': G_lower
            }
        }
    except Exception as e:
        print(f"Error in compute_full_tensors: {e}")
        return None

@csrf_exempt
@require_POST
def visualize_view(request):
    try:
        data = json.loads(request.body)
        metric_text = data.get("metric_text")
        
        if not metric_text:
            return JsonResponse({
                'error': 'Missing metric_text'
            }, status=400)

        # Podstawowe obliczenia
        wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        n = len(wspolrzedne)
        g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)

        # Generowanie wykresu
        result = generate_numerical_curvature(
            Scalar_Curvature,
            wspolrzedne,
            parametry,
            ranges=[[-2, 2]] * len(wspolrzedne),  # mniejszy zakres
            points_per_dim=15  # mniej punktów
        )

        if not result:
            return JsonResponse({
                'error': 'Error generating plot'
            }, status=400)

        return JsonResponse({
            'plot': result['plot'],
            'coordinates': result['coordinates']
        })

    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)

