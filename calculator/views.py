from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import logging
from celery.result import AsyncResult
import sympy as sp
from calculator.tasks import compute_tensors_task
from django.views.decorators.http import require_GET
from celery.result import AsyncResult

logger = logging.getLogger(__name__)

def convert_to_latex(obj):
    if isinstance(obj, (sp.Basic, sp.Expr, sp.Matrix)):
        return sp.latex(obj)
    return str(obj)

@csrf_exempt
@require_POST
def calculate_view(request):
    """
    Widok: przyjmuje metric_text, uruchamia asynchroniczne zadanie Celery,
    zwraca task_id do sprawdzenia statusu lub pobrania wyniku.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            metric_text = data.get('metric_text')
            if not metric_text:
                return JsonResponse({'error': 'Missing metric_text'}, status=400)

            # Uruchom zadanie Celery w tle
            task = compute_tensors_task.delay(metric_text)

            # Zwróć ID zadania, aby klient mógł sprawdzić status
            return JsonResponse({'task_id': task.id, 'status': 'processing'}, status=202)

        except Exception as e:
            logger.error(f"Request error: {str(e)}", exc_info=True)
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


def task_status_view(request, task_id):
    """
    Endpoint do sprawdzania stanu zadania Celery (opcjonalny).
    """
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        # Zadanie nie rozpoczęło się jeszcze
        return JsonResponse({'task_id': task_id, 'state': 'PENDING'})

    elif task_result.state == 'FAILURE':
        # Zadanie zakończyło się niepowodzeniem
        return JsonResponse({
            'task_id': task_id,
            'state': 'FAILURE',
            'error': str(task_result.result),
        })

    elif task_result.state == 'SUCCESS':
        # Zadanie zakończyło się sukcesem, w .result mamy dane
        return JsonResponse({
            'task_id': task_id,
            'state': 'SUCCESS',
            'result': task_result.result  # Twój słownik z tensorami
        })

    else:
        # IN PROGRESS lub RETRY
        return JsonResponse({'task_id': task_id, 'state': task_result.state})
    
@require_GET
def task_status_view(request, task_id):
    task_result = AsyncResult(task_id)
    return JsonResponse({
        'task_id': task_id,
        'state': task_result.state,
        'result': task_result.result,
    })