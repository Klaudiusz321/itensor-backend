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
    """
    Endpoint dla obliczeń symbolicznych (wzory LaTeX)
    """
    try:
        data = json.loads(request.body)
        metric_text = data.get('metric_text')
        
        if not metric_text:
            return JsonResponse({
                'error': 'Brak tekstu metryki',
                'detail': 'Pole metric_text jest wymagane'
            }, status=400)

        # Obliczenia symboliczne
        wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        n = len(wspolrzedne)
        g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
        g_inv = g.inv()
        G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n)
        
        # Generowanie wyniku symbolicznego
        output = generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n)
        return JsonResponse(parse_metric_output(output))
        
    except Exception as e:
        print("Błąd:", str(e))
        return JsonResponse({
            'error': 'Błąd serwera',
            'detail': str(e)
        }, status=400)

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
    start_time = time.time()
    try:
        logger.info("Starting visualization...")
        
        data = json.loads(request.body)
        metric_text = data.get("metric_text")
        
        # Sprawdź cache
        cache_key = f"visualization_{hash(metric_text)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return JsonResponse(cached_result)
            
        logger.info("Generating visualization...")
        wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        n = len(wspolrzedne)
        g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
        result = generate_numerical_curvature(
            Scalar_Curvature,
            wspolrzedne,
            parametry,
            ranges=[[-2, 2]] * len(wspolrzedne),
            points_per_dim=15
        )

        if not result:
            logger.error("Failed to generate visualization")
            return JsonResponse({'error': 'Visualization generation failed'}, status=500)

        logger.info("Visualization generated successfully")
        
        # Zapisz do cache
        cache.set(cache_key, result, timeout=3600)
        
        # Po każdym głównym kroku
        logger.info(f"Step X completed in {time.time() - start_time:.2f}s")
        
        # Jeśli przekroczono czas
        if time.time() - start_time > 25:  # Heroku ma limit 30s
            logger.error("Timeout approaching")
            return JsonResponse({'error': 'Operation timeout'}, status=408)
        
        return JsonResponse({
            'plot': result['plot'],
            'coordinates': result['coordinates']
        })

    except Exception as e:
        logger.exception("Error in visualize_view")
        return JsonResponse({'error': str(e)}, status=500)

