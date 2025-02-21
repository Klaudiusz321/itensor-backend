import json
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from myproject.utilis.symbolic_calculations import compute_metric_schwarzschild

@csrf_exempt
def compute_metric(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        M_val = data.get('M', 1)
        r_val = data.get('r', 3)
        theta_val = data.get('theta', np.pi / 2)
        phi_val = data.get('phi', 0)

        numeric_result = compute_metric_schwarzschild(M_val, r_val, theta_val, phi_val)

        return JsonResponse({
            'scalar_curvature': float(numeric_result)
        })
    else:
        return JsonResponse({'error': 'POST request required'}, status=400)
