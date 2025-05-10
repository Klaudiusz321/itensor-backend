# calculator/views/tensor_viewset.py

from datetime import datetime
import logging

import numpy as np
import sympy
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet

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
        missing = [f for f in ('dimension', 'coordinates', 'metric') if f not in data]
        if missing:
            return Response({'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        dim     = data['dimension']
        coords  = data['coordinates']
        metric  = data['metric']
        eval_pt = data.get('evaluation_point') or data.get('evaluationPoint') or [0]*dim
        if len(eval_pt) != dim:
            return Response({'success': False, 'error': f"evaluation_point must be length {dim}"},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            result = compute_symbolic(
                dimension=dim,
                coords=coords,
                metric=metric,
                evaluation_point=eval_pt
            )
        except Exception as e:
            logger.error("Symbolic error", exc_info=True)
            return Response({'success': False, 'error': str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(result)

    @action(detail=False, methods=['post'], url_path='numerical')
    def numerical(self, request):
        data = request.data
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
        existing = Tensor.objects.filter(
            dimension=n,
            coordinates=coords,
            metric_data=metric
        ).first()
        if existing:
            out['tensor_id'] = existing.id
            return Response(out)

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
        """
        Proxy do Twojej istniejącej funkcji differential_operators(request):
        kalkuluje gradient, dywergencję, laplasjan itp.
        """
        # wystarczy przekazać request dalej
        return differential_operators(request)
