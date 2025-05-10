# calculator/views/tensor_viewset.py

from datetime import datetime
import json
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
from calculator.views.views import differential_operators as diff_ops_view
from myproject.utils.numerical.core import NumericTensorCalculator

logger = logging.getLogger(__name__)

class TensorViewSet(ModelViewSet):
    queryset = Tensor.objects.all()
    serializer_class = TensorSerializer
    http_method_names = ['get', 'post', 'put', 'patch', 'delete']

    def create(self, request, *args, **kwargs):
        data = request.data.copy()

        # 1) jeśli ktoś przesłał "metric" zamiast "metric_data"
        if 'metric' in data and 'metric_data' not in data:
            data['metric_data'] = data['metric']

        # 2) jeżeli front wysłał "results": rozpakuj je do pól DB
        #    (to są pola, których używa Twój defaults w nagłówku)
        results = data.get('results')
        if isinstance(results, dict):
            # sam wynik zostawiamy w components
            data['components'] = results
            # a każdy element rozdzielamy do właściwych kolumn
            data['christoffel_symbols'] = results.get('christoffelSymbols', [])
            data['riemann_tensor']      = results.get('riemannTensor', [])
            data['ricci_tensor']        = results.get('ricciTensor', [])
            # scalar jako string
            data['scalar_curvature']    = str(results.get('scalarCurvature', '')) 
            data['einstein_tensor']     = results.get('einsteinTensor', [])
            data['weyl_tensor']         = results.get('weylTensor', [])

        # 3) oblicz hash po metric_data/coods/dimension
        metric_hash = Tensor.generate_metric_hash(
            data.get('dimension'),
            data.get('coordinates'),
            data.get('metric_data')
        )
        data['metric_hash'] = metric_hash

        # 4) przygotuj "defaults" do get_or_create
        defaults = {
            'name':              data.get('name', f"Tensor @ {datetime.now().isoformat()}"),
            'description':       data.get('description', ''),
            'components':        data.get('components', {}),
            'dimension':         data.get('dimension'),
            'coordinates':       data.get('coordinates'),
            'metric_data':       data.get('metric_data'),
            'christoffel_symbols': data.get('christoffel_symbols', []),
            'riemann_tensor':    data.get('riemann_tensor', []),
            'ricci_tensor':      data.get('ricci_tensor', []),
            'scalar_curvature':  data.get('scalar_curvature'),
            'einstein_tensor':   data.get('einstein_tensor', []),
            'weyl_tensor':       data.get('weyl_tensor', []),
        }

        # 5) get_or_create
        tensor, created = Tensor.objects.get_or_create(
            metric_hash=metric_hash,
            defaults=defaults
        )

        serializer = self.get_serializer(tensor)
        status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return Response(serializer.data, status=status_code)

    @action(detail=False, methods=['post'], url_path='find-similar')
    def find_similar(self, request):
        data   = request.data
        metric = data.get('metric_data') or data.get('metric')
        missing = [f for f in ('dimension','coordinates') if f not in data]
        if metric is None:
            missing.append('metric_data/metric')
        if missing:
            return Response({'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        existing = Tensor.objects.filter(
            dimension    = data['dimension'],
            coordinates  = data['coordinates'],
            metric_data  = metric,
        ).first()

        if existing:
            return Response({
                'success': True,
                'found':   True,
                'tensor':  TensorSerializer(existing).data
            }, status=status.HTTP_200_OK)

        return Response({'success': True, 'found': False}, status=status.HTTP_200_OK)

    @action(detail=False, methods=['post'], url_path='symbolic')
    def symbolic(self, request):
        data   = request.data
        dim    = data.get('dimension')
        coords = data.get('coordinates')
        metric = data.get('metric')

        # 0) cache hit?
        existing = Tensor.objects.filter(
            dimension   = dim,
            coordinates = coords,
            metric_data = metric,
            # jeśli masz pole 'computation', możesz je tu sprawdzić
        ).first()
        if existing:
            return Response({
                **existing.components,
                'tensor_id': existing.id,
                'cached':    True,
                'success':   True
            }, status=status.HTTP_200_OK)

        # 1) validate
        missing = [f for f in ('dimension','coordinates','metric') if data.get(f) is None]
        if missing:
            return Response({'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        eval_pt = data.get('evaluation_point') or data.get('evaluationPoint') or [0]*dim
        if not isinstance(eval_pt, (list,tuple)) or len(eval_pt)!=dim:
            return Response({'success': False,
                             'error': f"evaluation_point must be list of length {dim}"},
                            status=status.HTTP_400_BAD_REQUEST)

        # 2) compute
        try:
            result = compute_symbolic(
                dimension        = dim,
                coords           = coords,
                metric           = metric,
                evaluation_point = eval_pt
            )
        except Exception as e:
            logger.error("Symbolic calculation error", exc_info=True)
            return Response({'success':False,'error':str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 3) save
        try:
            tensor = Tensor.objects.create(
                name         = f"Symbolic @ {datetime.now().isoformat()}",
                description  = data.get('description',''),
                dimension    = dim,
                coordinates  = coords,
                metric_data  = metric,
                components   = result,
            )
            tensor_id = tensor.id
        except Exception as db_err:
            logger.error("Failed to save symbolic Tensor", exc_info=True)
            tensor_id = None
            result['db_save_error'] = str(db_err)

        # 4) return
        return Response({
            **result,
            'tensor_id': tensor_id,
            'cached':    False,
            'success':   True
        }, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'], url_path='numerical')
    def numerical(self, request):
        data   = request.data
        dim    = data.get('dimension')
        coords = data.get('coordinates')
        metric = data.get('metric')

        # 0) cache hit?
        existing = Tensor.objects.filter(
            dimension    = dim,
            coordinates  = coords,
            metric_data  = metric,
        ).first()
        if existing:
            return Response(TensorSerializer(existing).data, status=status.HTTP_200_OK)

        # 1) validate
        missing = [f for f in ('dimension','coordinates','metric') if data.get(f) is None]
        if missing:
            return Response({'success':False,'error':f"Missing fields: {', '.join(missing)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        eval_pt = data.get('evaluation_point') or data.get('evaluationPoint') or [0.0]*dim
        if not isinstance(eval_pt,(list,tuple)) or len(eval_pt)!=dim:
            return Response({'success':False,'error':f"evaluation_point must be length {dim}"},
                            status=status.HTTP_400_BAD_REQUEST)
        eval_pt = list(map(float, eval_pt))

        # 2) build g_func via Sympy
        coord_syms = sympy.symbols(coords)
        sym_locals = dict(zip(coords, coord_syms))
        exprs = [
            [ sympy.sympify(entry, locals=sym_locals)
              if isinstance(entry,str)
              else sympy.sympify(entry)
              for entry in row ]
            for row in metric
        ]
        f_metric = sympy.lambdify(coord_syms, exprs, modules=["numpy"])
        def g_func(x_arr):
            return np.array(f_metric(*x_arr), dtype=float)

        # 3) check singularity
        try:
            eig = np.linalg.eigvals(g_func(np.array(eval_pt)))
            if np.any(np.isclose(eig,0.0,atol=1e-12)):
                return Response({'success':False,
                                 'error':f"Metric singular at {eval_pt}. Eigenvalues: {eig.tolist()}"},
                                status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            pass

        # 4) numeric
        try:
            calc    = NumericTensorCalculator(g_func, h=data.get('h',1e-6))
            results = calc.compute_all(np.array(eval_pt))
        except Exception as e:
            logger.error("NumericTensorCalculator error", exc_info=True)
            return Response({'success':False,'error':str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 5) prepare payload
        out = {
            'success':           True,
            'dimension':         dim,
            'coordinates':       coords,
            'evaluation_point':  eval_pt,
            'metric':            results['metric'].tolist(),
            'inverse_metric':    results['metric_inv'].tolist(),
            'christoffelSymbols':results['christoffel'].tolist(),
            'riemannTensor':     results['riemann_lower'].tolist(),
            'ricciTensor':       results['ricci'].tolist(),
            'scalarCurvature':   float(results['scalar']),
            'einsteinTensor':    results['einstein_lower'].tolist(),
            'weylTensor':        [],  # jeżeli będziesz
        }

        # 6) save
        try:
            tensor = Tensor.objects.create(
                name             = f"Numerical @ {datetime.now().isoformat()}",
                dimension        = dim,
                coordinates      = coords,
                metric_data      = metric,
                christoffel_symbols = out['christoffelSymbols'],
                riemann_tensor      = out['riemannTensor'],
                ricci_tensor        = out['ricciTensor'],
                scalar_curvature    = str(out['scalarCurvature']),
                einstein_tensor     = out['einsteinTensor'],
            )
            out['tensor_id'] = tensor.id
        except Exception as db_err:
            logger.error("Failed saving numerical Tensor", exc_info=True)
            out['tensor_id'] = None
            out['db_save_error'] = str(db_err)

        return Response(out, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'], url_path='differential-operators')
    def differential_operators(self, request):
        data = request.data
        logger.info(f"Differential operators request: {data}")

        # 0) cache hit?
        existing = Tensor.objects.filter(
            dimension    = data.get('dimension'),
            coordinates  = data.get('coordinates'),
            metric_data  = data.get('metric'),
            components__vector = data.get('vector_field'),
            components__scalar = data.get('scalar_field'),
        ).first()
        if existing:
            return Response({
                **existing.components,
                'tensor_id': existing.id,
                'cached':    True,
                'success':   True
            }, status=status.HTTP_200_OK)

        # 1) delegate to your function returning Django JsonResponse
        json_resp = diff_ops_view(request)

        # 2) parse
        try:
            payload = json.loads(json_resp.content)
        except Exception:
            logger.error("Failed to parse differential_operators response", exc_info=True)
            return Response({'success':False,'error':'Invalid diff_ops response'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if json_resp.status_code != 200:
            return Response(payload, status=json_resp.status_code)

        # 3) save
        try:
            tensor = Tensor.objects.create(
                name        = f"Differential Ops @ {datetime.now().isoformat()}",
                description = data.get('description',''),
                dimension   = data['dimension'],
                coordinates = data['coordinates'],
                metric_data = data['metric'],
                components  = payload,
            )
            payload['tensor_id'] = tensor.id
        except Exception as db_err:
            logger.error("Failed saving diff_ops Tensor", exc_info=True)
            payload['db_save_error'] = str(db_err)

        payload['cached'] = False
        payload['success'] = True
        return Response(payload, status=status.HTTP_201_CREATED)
