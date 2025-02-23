from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from myproject.utilis.calcualtion import oblicz_tensory, compute_einstein_tensor, wczytaj_metryke_z_tekstu, generate_output, generate_numerical_curvature


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
@require_http_methods(["POST"])
def calculate(request):
    try:
        print("Otrzymano żądanie:", request.body)
        
        data = json.loads(request.body)
        metric_text = data.get('metric_text')
        numerical_ranges = data.get('ranges', None)
        
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
        parsed_result = parse_metric_output(output)
        
        if numerical_ranges:
            numerical_data = generate_numerical_curvature(
                Scalar_Curvature,
                wspolrzedne,
                parametry,
                numerical_ranges
            )
            if numerical_data:
                parsed_result['numerical_data'] = numerical_data
        
        return JsonResponse(parsed_result)
        
    except json.JSONDecodeError as e:
        return JsonResponse({
            'error': 'Nieprawidłowy format JSON',
            'detail': str(e)
        }, status=400)
    except Exception as e:
        print("Błąd:", str(e))
        return JsonResponse({
            'error': 'Błąd serwera',
            'detail': str(e)
        }, status=400)
