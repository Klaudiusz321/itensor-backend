# calculator/views/numerical/numerical_views.py

import json
import logging
import traceback
import datetime

import numpy as np
import sympy
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from calculator.models import Tensor
from myproject.utils.numerical.core import NumericTensorCalculator

logger = logging.getLogger(__name__)


@csrf_exempt
@require_POST
def numerical_calculate_view(request):
    # 1) Parsowanie JSON
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"success": False, "error": f"Invalid JSON: {e}"},
            status=400
        )

    # 2) Walidacja obecności wymaganych pól
    for field in ("dimension", "metric", "evaluation_point", "coordinates"):
        if field not in payload:
            return JsonResponse(
                {"success": False, "error": f"Missing field: {field}"},
                status=400
            )

    n = payload["dimension"]
    metric_data = payload["metric"]
    eval_pt = payload["evaluation_point"]
    coords = payload["coordinates"]

    # 3) Podstawowe sprawdzenie rozmiarów macierzy i list
    if len(metric_data) != n or any(len(row) != n for row in metric_data):
        return JsonResponse(
            {"success": False, "error": f"Metric must be {n}×{n}"},
            status=400
        )
    if len(eval_pt) != n:
        return JsonResponse(
            {"success": False, "error": f"Evaluation point must have {n} entries"},
            status=400
        )
    if len(coords) != n:
        return JsonResponse(
            {"success": False, "error": f"Coordinates must have {n} names"},
            status=400
        )

    try:
        # 4) Tworzymy symbole Sympy i słownik do sympify
        coord_syms = sympy.symbols(coords)
        sym_locals = {name: sym for name, sym in zip(coords, coord_syms)}

        # 5) Parsujemy każdy entry: liczba → Expr, string → sympify
        expr_matrix = []
        for row in metric_data:
            expr_row = []
            for entry in row:
                if isinstance(entry, str):
                    expr_row.append(sympy.sympify(entry, locals=sym_locals))
                else:
                    expr_row.append(sympy.sympify(entry))
            expr_matrix.append(expr_row)

        # 6) Lambdify całej macierzy do funkcji NumPy
        f_metric = sympy.lambdify(coord_syms, expr_matrix, modules=["numpy"])

        # 7) Wrapper: NumericTensorCalculator oczekuje g_func: np.ndarray → np.ndarray
        def g_func(x: np.ndarray) -> np.ndarray:
            raw = f_metric(*x)  # może być list of lists lub ndarray
            return np.array(raw, dtype=float)

        # 8) Właściwe obliczenia
        calc = NumericTensorCalculator(g_func, h=payload.get("h", 1e-6))
        for method in ("compute_all", "calculate_all", "compute"):
            if hasattr(calc, method):
                results = getattr(calc, method)(np.array(eval_pt, dtype=float))
                break
        else:
            raise AttributeError(
                "NumericTensorCalculator has no compute_all/calculate_all/compute method"
            )

    except Exception as exc:
        logger.error(traceback.format_exc())
        return JsonResponse({"success": False, "error": str(exc)}, status=500)

    # 9) Budujemy odpowiedź
    out = {
        "success": True,
        "dimension": n,
        "coordinates": coords,
        "evaluation_point": eval_pt,
        "metric": results["metric"].tolist(),
        "inverse_metric": results["metric_inv"].tolist(),
    }
    # Dodajemy tensory
    for key, out_key in (
        ("christoffel", "christoffel"),
        ("riemann_lower", "riemann_lower"),
        ("ricci", "ricci"),
        ("scalar", "scalar"),
        ("einstein_lower", "einstein"),
        ("einstein_upper", "einstein_upper"),
    ):
        val = results.get(key)
        out[out_key] = val.tolist() if isinstance(val, np.ndarray) else float(val)

    # 10) Deduplica i zapis do bazy
    existing = Tensor.objects.filter(
        dimension=n,
        coordinates=coords,
        metric_data=metric_data,
    ).first()

    if existing:
        out["tensor_id"] = existing.id
        return JsonResponse(out, status=200)

    tensor = Tensor.objects.create(
        name=f"Numerical @ {datetime.datetime.now().isoformat()}",
        dimension=n,
        coordinates=coords,
        metric_data=metric_data,
        christoffel_symbols=out["christoffel"],
        riemann_tensor=out["riemann_lower"],
        ricci_tensor=out["ricci"],
        scalar_curvature=str(out["scalar"]),
        einstein_tensor=out["einstein"],
    )
    out["tensor_id"] = tensor.id
    return JsonResponse(out, status=201)
