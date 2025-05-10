# calculator/views/symbolic/symbolic_views.py

import json
import logging
import traceback

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from myproject.utils.symbolic.compute_tensor import ComputeTensor
from calculator.models import Tensor  # załóżmy, że model jest w calculator/models.py

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def symbolic_calculate_view(request):
    """
    API endpoint for symbolic tensor calculations.
    Expects JSON with at least:
      - "dimension": int
      - "coordinates": list[str]
      - "metric": dict or matrix
    """
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"success": False, "error": f"Invalid JSON: {e}"}, 
            status=400
        )

    # Prosta walidacja
    missing = [f for f in ("dimension","coordinates","metric") if f not in payload]
    if missing:
        return JsonResponse(
            {"success": False, "error": f"Missing fields: {', '.join(missing)}"},
            status=400
        )

    try:
        # Wywołujemy główną logikę
        tensor_calc = ComputeTensor(
            coords           = payload["coordinates"],
            metric           = payload["metric"],
            dimension        = payload["dimension"],
            evaluation_point = payload.get("evaluation_point", {})
        )
        # Konwertujemy wynik do słownika
        result = tensor_calc.to_dict()
    except Exception as err:
        logger.error("Error in symbolic computation:\n%s", traceback.format_exc())
        return JsonResponse(
            {"success": False, "error": str(err)},
            status=500
        )

    # --- ZAPIS DO BAZY ---
    try:
        existing = Tensor.objects.filter(
            dimension=payload["dimension"], 
            coordinates=payload["coordinates"],
            metric_data=payload["metric"],
        ).first()
        if existing:
            tensor = existing
        else:
            tensor = Tensor.objects.create(
                name                = f"Symbolic @ {request.META.get('REMOTE_ADDR')}",
            dimension           = payload["dimension"],
            coordinates         = payload["coordinates"],
            metric_data         = payload["metric"],
            components          = result,            # możesz filtrować, co chcesz zachować
            description         = payload.get("description", "")
        )
        # dodajemy do odpowiedzi jego id
        result["db_id"] = tensor.id
    except Exception as db_err:
        logger.error("Could not save Tensor to DB: %s", traceback.format_exc())
        # ale nawet jeśli się nie uda, zwracamy wynik obliczeń
        result["db_save_error"] = str(db_err)

    return JsonResponse(result)
