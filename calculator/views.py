import json
from django.http import HttpResponse
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from myproject.utilis.tensor_calculations import (
    wczytaj_metryke_z_tekstu,
    oblicz_tensory,
    compute_einstein_tensor,
    generate_christoffel_latex,
    generate_riemann_latex,
    generate_ricci_latex,
    generate_einstein_latex
)
import sympy as sp


def index(request):
    return HttpResponse("Hello from index page!")
@csrf_exempt  # Wyłączamy CSRF dla uproszczenia API – w produkcji warto zaimplementować właściwą ochronę
def calculate(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed")
    try:
        data = json.loads(request.body)
        metric_text = data.get("metric_text")
        if not metric_text:
            return HttpResponseBadRequest("No metric_text provided")
        
        # Wywołanie funkcji obliczeniowych
        wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        if not (wspolrzedne and metryka):
            return HttpResponseBadRequest("Invalid metric data")
        
        g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
        g_inv = g.inv()
        G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, len(wspolrzedne))
        
        # Generowanie LaTeX
        christoffel_latex = generate_christoffel_latex(Gamma, len(wspolrzedne))
        riemann_latex = generate_riemann_latex(R_abcd, len(wspolrzedne))
        ricci_latex = generate_ricci_latex(Ricci, len(wspolrzedne))
        einstein_latex = generate_einstein_latex(G_lower, len(wspolrzedne))
        scalar_curv_latex = sp.latex(Scalar_Curvature)
        
        response_data = {
            "coordinates": [str(w) for w in wspolrzedne],
            "parameters": [str(p) for p in parametry],
            "metryka": {f"{k[0]},{k[1]}": str(expr) for k, expr in metryka.items()},
            "scalarCurvature": str(Scalar_Curvature),
            "scalarCurvatureLatex": scalar_curv_latex,
            "christoffelLatex": christoffel_latex,
            "riemannLatex": riemann_latex,
            "ricciLatex": ricci_latex,
            "einsteinLatex": einstein_latex,
        }
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
