from django.http import JsonResponse
from django.views.decorators.http import require_GET
import time, logging
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


logger = logging.getLogger(__name__)



@require_GET
def health_check(request):
    return JsonResponse({'status':'ok','timestamp': time.time()})

# proxy do differential-operators, jeżeli Ci to potrzebne
@csrf_exempt
@require_POST
def differential_operators(request):
    from .diffrential.differential_operators_views import differential_operators
    return differential_operators(request)

@csrf_exempt
@require_POST
def numerical_calculate_view(request):
    from .numerical.numerical_views import numerical_calculate_view
    return numerical_calculate_view(request)

@csrf_exempt
@require_POST
def symbolic_calculate_view(request):
    from .symbolic.symbolic_views import symbolic_calculate_view
    return symbolic_calculate_view(request)

@csrf_exempt
@require_POST
def mhd_simulation(request):
    from .mhd.mhd import mhd_simulation
    return mhd_simulation(request)
