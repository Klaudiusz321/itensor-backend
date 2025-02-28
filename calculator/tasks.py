
from celery import shared_task

import logging
from celery import shared_task
from myproject.utilis.calcualtion.compute_tensor import (
    oblicz_tensory, 
    compute_weyl_tensor, 
    compute_einstein_tensor,
    compute_weyl_tensor,
)
import sympy as sp
from myproject.utilis.calcualtion.prase_metric import wczytaj_metryke_z_tekstu

def convert_to_latex(obj):
    if isinstance(obj, (sp.Basic, sp.Expr, sp.Matrix)):
        return sp.latex(obj)
    return str(obj)

logger = logging.getLogger(__name__)

@shared_task
def compute_tensors_task(metric_text: str):
    """
    Funkcja uruchamiana w tle (worker) – zawiera długotrwałe obliczenia.
    Zwraca słownik z wynikami (musi być serializowalny do JSON).
    """
    try:
        # Parsuj metric_text i oblicz
        wspolrzedne, parametry, metryka = wczytaj_metryke_z_tekstu(metric_text)
        n = len(wspolrzedne)

        g, Gamma, R_abcd, Ricci, Scalar_Curvature = oblicz_tensory(wspolrzedne, metryka)
        Weyl = compute_weyl_tensor(R_abcd, Ricci, Scalar_Curvature, g, n)

        # Sprawdź, czy macierz metryczna nie jest osobliwa
        if g.det() == 0:
            return {'error': 'Metric tensor is singular'}

        g_inv = g.inv()
        G_upper, G_lower = compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n)

        result = {
            'coordinates': [str(coord) for coord in wspolrzedne],
            'parameters': [str(param) for param in parametry],
            'metric': [
                f"g_{{{i}{j}}} = {convert_to_latex(g[i,j])}"
                for i in range(n) for j in range(n) if g[i,j] != 0
            ],
            'christoffel': [
                f"\\Gamma^{{{k}}}_{{{i}{j}}} = {convert_to_latex(Gamma[k][i][j])}"
                for k in range(n) for i in range(n) for j in range(n)
                if Gamma[k][i][j] != 0
            ],
            'riemann': [
                f"R_{{{a}{b}{c}{d}}} = {convert_to_latex(R_abcd[a][b][c][d])}"
                for a in range(n) for b in range(n)
                for c in range(n) for d in range(n)
                if R_abcd[a][b][c][d] != 0
            ],
            'ricci': [
                f"R_{{{i}{j}}} = {convert_to_latex(Ricci[i,j])}"
                for i in range(n) for j in range(n)
                if Ricci[i,j] != 0
            ],
            'einstein': [
                f"G_{{{i}{j}}} = {convert_to_latex(G_lower[i,j])}"
                for i in range(n) for j in range(n)
                if G_lower[i,j] != 0
            ],
            'scalar': [
                f"R = {convert_to_latex(Scalar_Curvature)}"
            ],
            'Weyl': [
                f"C_{{{i}{j}{k}{l}}} = {convert_to_latex(Weyl[i][j][k][l])}"
                for i in range(n) for j in range(n)
                for k in range(n) for l in range(n)
                if Weyl[i][j][k][l] != 0
            ]
        }
        return result

    except Exception as e:
        logger.error(f"Error in compute_tensors_task: {e}", exc_info=True)
        return {'error': str(e)}
