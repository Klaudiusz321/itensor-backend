from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from myproject.utilis.tensor_calculations import (
    wczytaj_metryke_z_tekstu,
    oblicz_tensory,
    compute_einstein_tensor,
    generate_output,
    process_latex
)

@csrf_exempt
@require_http_methods(["POST"])
def calculate(request):
    try:
        # Dodaj logowanie dla debugowania
        print("Otrzymano żądanie:", request.body)
        
        data = json.loads(request.body)
        metric_text = data.get('metric_text')
        
        if not metric_text:
            return JsonResponse({
                'error': 'Brak tekstu metryki',
                'detail': 'Pole metric_text jest wymagane'
            }, status=400)

        # Obliczenia
        wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        n = len(wspolrzedne)
        g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
        g_inv = g.inv()
        G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n)
        
        # Generowanie wyniku
        output = generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n)
        
        return JsonResponse({
            'result': output,
            'success': True
        })
        
    except json.JSONDecodeError as e:
        return JsonResponse({
            'error': 'Nieprawidłowy format JSON',
            'detail': str(e)
        }, status=400)
    except Exception as e:
        print("Błąd:", str(e))  # Dodaj logowanie błędów
        return JsonResponse({
            'error': 'Błąd serwera',
            'detail': str(e)
        }, status=400)
