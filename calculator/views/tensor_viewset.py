# calculator/views/tensor_viewset.py

from datetime import datetime
import logging

import numpy as np
import sympy
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
import json
from calculator.views.views import differential_operators as diff_ops_view
from calculator.models import Tensor
from calculator.serializers import TensorSerializer
from calculator.symbolics import compute_symbolic
from calculator.views.views import differential_operators
from myproject.utils.numerical.core import NumericTensorCalculator

logger = logging.getLogger(__name__)


class TensorViewSet(ModelViewSet):
    queryset = Tensor.objects.all()
    serializer_class = TensorSerializer
    http_method_names = ['get', 'post', 'put', 'patch', 'delete']

    @action(detail=False, methods=['post'], url_path='find-similar')
    def find_similar(self, request):
        data = request.data
        # Akceptujemy zarówno 'metric' jak i 'metric_data'
        metric = data.get('metric_data') or data.get('metric')
        missing = [f for f in ('dimension', 'coordinates',) if f not in data] + ([] if metric is not None else ['metric_data/metric'])
        if missing:
            return Response({'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        existing = Tensor.objects.filter(
            dimension=data['dimension'],
            coordinates=data['coordinates'],
            metric_data=metric,
        ).first()
        if existing:
            return Response({'success': True, 'found': True, 'tensor': TensorSerializer(existing).data})
        return Response({'success': True, 'found': False})

    @action(detail=False, methods=['post'], url_path='symbolic')
    def symbolic(self, request):
        data = request.data

        # 0) Cache hit?
        existing = Tensor.objects.filter(
            dimension    = data.get('dimension'),
            coordinates  = data.get('coordinates'),
            metric_data  = data.get('metric'),
            computation  = 'symbolic'  # jeśli masz pole computation
        ).first()
        if existing:
            # zwracamy zapisane components + id
            return Response({
                **existing.components,
                'tensor_id': existing.id,
                'cached': True,
                'success': True
            }, status=status.HTTP_200_OK)

        # 1) Walidacja
        missing = [f for f in ('dimension', 'coordinates', 'metric') if f not in data]
        if missing:
            return Response(
                {'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        dim    = data['dimension']
        coords = data['coordinates']
        metric = data['metric']
        eval_pt = data.get('evaluation_point') \
               or data.get('evaluationPoint') \
               or [0]*dim

        if not isinstance(eval_pt, (list, tuple)) or len(eval_pt) != dim:
            return Response(
                {'success': False, 'error': f"evaluation_point must be a list of length {dim}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2) Wykonanie obliczeń
        try:
            result = compute_symbolic(
                dimension        = dim,
                coords           = coords,
                metric           = metric,
                evaluation_point = eval_pt
            )
        except Exception as e:
            logger.error("Symbolic calculation error", exc_info=True)
            return Response(
                {'success': False, 'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3) Zapis do bazy
        try:
            tensor = Tensor.objects.create(
                name          = f"Symbolic @ {datetime.now().isoformat()}",
                description   = data.get('description', ''),
                dimension     = dim,
                coordinates   = coords,
                metric_data   = metric,
                components    = result,
                computation   = 'symbolic'
            )
            tensor_id = tensor.id
        except Exception as db_err:
            logger.error("Failed to save symbolic Tensor: %s", db_err, exc_info=True)
            tensor_id = None
            result['db_save_error'] = str(db_err)

        # 4) Zwracamy wynik
        return Response({
            **result,
            'tensor_id': tensor_id,
            'cached': False,
            'success': True
        }, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'], url_path='numerical')
    def numerical(self, request):
        data = request.data
        existing = Tensor.objects.filter(
            dimension=data.get('dimension'),
            coordinates=data.get('coordinates'),
            metric_data=data.get('metric'),
        ).first()
        if existing:
            serializer = self.get_serializer(existing)
            return Response(serializer.data, status=status.HTTP_200_OK)

        missing = [f for f in ('dimension', 'coordinates', 'metric') if f not in data]
        if missing:
            return Response({'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        n       = data['dimension']
        coords  = data['coordinates']
        metric  = data['metric']
        eval_pt = data.get('evaluation_point') or data.get('evaluationPoint') or [0.0]*n
        if len(eval_pt) != n:
            return Response({'success': False, 'error': f"evaluation_point must be length {n}"},
                            status=status.HTTP_400_BAD_REQUEST)
        eval_pt = list(map(float, eval_pt))

        # Parsujemy metric przez Sympy → g_func
        coord_syms = sympy.symbols(coords)
        sym_locals = dict(zip(coords, coord_syms))
        exprs = [
            [sympy.sympify(entry, locals=sym_locals) if isinstance(entry, str) else sympy.sympify(entry)
             for entry in row]
            for row in metric
        ]
        f_metric = sympy.lambdify(coord_syms, exprs, modules=["numpy"])
        def g_func(x_arr):
            return np.array(f_metric(*x_arr), dtype=float)

        # Sprawdzamy, czy metryka nie jest osobliwa
        try:
            eig = np.linalg.eigvals(g_func(np.array(eval_pt)))
            if np.any(np.isclose(eig, 0.0, atol=1e-12)):
                return Response({'success': False,
                                 'error': f"Metric is singular at {eval_pt}. Eigenvalues: {eig.tolist()}"},
                                status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            pass

        try:
            calc = NumericTensorCalculator(g_func, h=data.get('h', 1e-6))
            results = calc.compute_all(np.array(eval_pt))
        except Exception as e:
            logger.error("NumericTensorCalculator error", exc_info=True)
            return Response({'success': False, 'error': str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        out = {
            'success': True,
            'dimension': n,
            'coordinates': coords,
            'evaluation_point': eval_pt,
            'metric':         results['metric'].tolist(),
            'inverse_metric': results['metric_inv'].tolist(),
            'christoffelSymbols': results['christoffel'].tolist(),
            'riemannTensor':      results['riemann_lower'].tolist(),
            'ricciTensor':        results['ricci'].tolist(),
            'scalarCurvature':    float(results['scalar']),
            'einsteinTensor':     results['einstein_lower'].tolist(),
            'weylTensor':         [],
        }

        # zapis / deduplikacja
       
        tensor = Tensor.objects.create(
            name=f"Numerical @ {datetime.now().isoformat()}",
            dimension=n,
            coordinates=coords,
            metric_data=metric,
            christoffel_symbols=out['christoffelSymbols'],
            riemann_tensor=out['riemannTensor'],
            ricci_tensor=out['ricciTensor'],
            scalar_curvature=str(out['scalarCurvature']),
            einstein_tensor=out['einsteinTensor'],
        )
        out['tensor_id'] = tensor.id
        return Response(out, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'], url_path='differential-operators')
    def differential_operators(self, request):
        data = request.data
        logger.info(f"Differential operators request: {data}")

        # 0) Sprawdź, czy taki sam zestaw parametrów już gdzieś liczyliśmy:
        existing = Tensor.objects.filter(
            dimension          = data.get('dimension'),
            coordinates        = data.get('coordinates'),
            metric_data        = data.get('metric'),
            components__vector = data.get('vector_field'),
            components__scalar = data.get('scalar_field'),
            computation        = 'differential_operators'
        ).first()
        if existing:
            return Response({
                **existing.components,
                'tensor_id': existing.id,
                'cached': True,
                'success': True
            }, status=status.HTTP_200_OK)

        # 1) Walidacja wymaganych pól
        missing = [f for f in (
            'dimension',
            'coordinates',
            'metric',
            'vector_field',
            'scalar_field',
            'selected_operators'
        ) if f not in data]
        if missing:
            return Response(
                {'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2) Przekaż obsługę do funkcji różniczkowej
        #    ona zwraca Django JsonResponse
        json_resp = diff_ops_view(request)

        # 3) Przekonwertuj Django JsonResponse na DRF Response
        try:
            payload = json.loads(json_resp.content)
        except Exception:
            logger.error("Nie udało się sparsować odpowiedzi differential_operators", exc_info=True)
            return Response(
                {'success': False, 'error': 'Invalid response from differential_operators'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        if json_resp.status_code != 200:
            return Response(payload, status=json_resp.status_code)

        # 4) Zapisz wynik w bazie
        try:
            tensor = Tensor.objects.create(
                name         = f"Differential Ops @ {datetime.now().isoformat()}",
                description  = data.get('description', ''),
                dimension    = data['dimension'],
                coordinates  = data['coordinates'],
                metric_data  = data['metric'],
                components   = payload,            # zapisujemy gradient/divergence/laplacian itp.
                computation  = 'differential_operators'
            )
            payload['tensor_id'] = tensor.id
        except Exception as db_err:
            logger.error("Błąd zapisu differential_operators do DB", exc_info=True)
            payload['db_save_error'] = str(db_err)

        payload['cached'] = False
        payload['success'] = True
        return Response(payload, status=status.HTTP_201_CREATED)