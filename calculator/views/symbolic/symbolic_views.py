# calculator/views/symbolic/symbolic_views.py

import json
import logging
import traceback
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from myproject.utils.symbolic.compute_tensor import ComputeTensor
from calculator.models import Tensor

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def symbolic_calculate_view(request):
    """
    API endpoint for symbolic tensor calculations.
    Expects JSON with:
      - "dimension": int
      - "coordinates": list[str]
      - "metric": 2D list or dict describing the metric
      - optional "evaluation_point": list[float]
    Caches results in the Tensor model so identical requests return stored result.
    """
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"success": False, "error": f"Invalid JSON: {e}"},
            status=400
        )

    # 1) Walidacja wymaganych pól
    missing = [f for f in ("dimension", "coordinates", "metric") if f not in payload]
    if missing:
        return JsonResponse(
            {"success": False, "error": f"Missing fields: {', '.join(missing)}"},
            status=400
        )

    dim    = payload["dimension"]
    coords = payload["coordinates"]
    metric = payload["metric"]
    eval_pt = payload.get("evaluation_point") or payload.get("evaluationPoint") or [0]*dim

    if not isinstance(eval_pt, (list, tuple)) or len(eval_pt) != dim:
        return JsonResponse(
            {"success": False, "error": f"evaluation_point must be a list of length {dim}"},
            status=400
        )

    # 2) Sprawdź, czy już mamy w bazie
    existing = Tensor.objects.filter(
        dimension=dim,
        coordinates=coords,
        metric_data=metric
    ).first()
    if existing:
        # Zakładamy, że components zawiera już gotowy wynik compute_tensor.to_dict()
        return JsonResponse({
            "success": True,
            "cached": True,
            "db_id": existing.id,
            **existing.components
        })

    # 3) Wykonaj obliczenia symboliczne
    try:
        tensor_calc = ComputeTensor(
            coords           = coords,
            metric           = metric,
            dimension        = dim,
            evaluation_point = eval_pt
        )
        result = tensor_calc.to_dict()  # np. {'christoffelSymbols':..., 'ricciTensor':..., ...}
    except Exception as err:
        logger.error("Error in symbolic computation:\n%s", traceback.format_exc())
        return JsonResponse(
            {"success": False, "error": str(err)},
            status=500
        )

    # 4) Zapisz nowy rekord w DB
    try:
        tensor = Tensor.objects.create(
            name          = f"Symbolic @ {datetime.now().isoformat()}",
            description   = payload.get("description", ""),
            dimension     = dim,
            coordinates   = coords,
            metric_data   = metric,
            components    = result
        )
        result["db_id"] = tensor.id
    except Exception as db_err:
        logger.error("Could not save Tensor to DB: %s", traceback.format_exc())
        result["db_save_error"] = str(db_err)

    # 5) Zwróć wynik
    return JsonResponse({"success": True, "cached": False, **result})
