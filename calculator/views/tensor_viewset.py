# calculator/views/tensor_viewset.py

from datetime import datetime
import logging
import numpy as np
import sympy

from rest_framework import status
from rest_framework.decorators import action
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response

from myproject.utils.numerical.core import NumericTensorCalculator
from calculator.models import Tensor
from calculator.serializers import TensorSerializer
from calculator.symbolics import compute_symbolic

logger = logging.getLogger(__name__)

class TensorViewSet(ModelViewSet):
    queryset = Tensor.objects.all()
    serializer_class = TensorSerializer
    http_method_names = ['get', 'post', 'put', 'patch', 'delete']

    @action(detail=False, methods=['post'], url_path='find-similar')
    def find_similar(self, request):
        """
        Find similar tensor calculations based on coordinates and metric
        """
        data = request.data
        logger.info(f"Find similar request data: {data}")
        
        # Validate basic fields
        missing = [f for f in ('dimension', 'coordinates', 'metric_data') if f not in data]
        if missing:
            return Response(
                {'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Find matching tensor in database
        existing = Tensor.objects.filter(
            dimension=data['dimension'],
            coordinates=data['coordinates'],
            metric_data=data['metric_data'],
        ).first()
        
        if existing:
            serializer = self.get_serializer(existing)
            return Response({
                'success': True,
                'found': True,
                'tensor': serializer.data
            })
        
        return Response({
            'success': True,
            'found': False
        })

    @action(detail=False, methods=['post'], url_path='symbolic')
    def symbolic(self, request):
        """
        POST /api/tensors/symbolic/
        Obliczenia symboliczne (nie zapisujemy w bazie tutaj).
        """
        data = request.data
        # 1) Walidacja wymaganych pól
        missing = [f for f in ('dimension', 'coordinates', 'metric') if f not in data]
        if missing:
            return Response(
                {'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        dim   = data['dimension']
        coords = data['coordinates']
        metric = data['metric']
        # obsługa ewentualnego camelCase
        eval_pt = data.get('evaluation_point') or data.get('evaluationPoint')

        # 2) Domyśl punkt ewaluacji, jeśli nie podano
        if eval_pt is None:
            # przyjmij zero dla każdego wymiaru
            eval_pt = [0] * dim
        # sprawdź długość
        if not isinstance(eval_pt, (list, tuple)) or len(eval_pt) != dim:
            return Response(
                {'success': False, 'error': f"evaluation_point must be a list of length {dim}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 3) Wywołaj funkcję compute_symbolic
        try:
            result = compute_symbolic(
                dimension=dim,
                coords=coords,
                metric=metric,
                evaluation_point=eval_pt
            )
        except Exception as e:
            # błąd w obliczeniach symbolicznych
            return Response(
                {'success': False, 'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 4) Zwróć wynik w formacie camelCase
        return Response(result, status=status.HTTP_200_OK)

    @action(detail=False, methods=['post'], url_path='numerical')
    def numerical(self, request):
        data = request.data
        logger.info(f"Numerical request data: {data}")

        # 1) Validate basic fields
        missing = [f for f in ('dimension', 'coordinates', 'metric') if f not in data]
        if missing:
            return Response(
                {'success': False, 'error': f"Missing fields: {', '.join(missing)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        n      = data['dimension']
        coords = data['coordinates']
        metric = data['metric']

        # 2) Get evaluation point (snake or camel)
        eval_pt = data.get('evaluation_point') \
               or data.get('evaluationPoint') \
               or [0.0] * n
        if not isinstance(eval_pt, (list, tuple)) or len(eval_pt) != n:
            return Response(
                {'success': False, 'error': f"evaluation_point must be length {n}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        eval_pt = list(map(float, eval_pt))
        logger.info(f"Using evaluation_point: {eval_pt}")

        # 3) Build symbolic metric expressions
        coord_syms = sympy.symbols(coords)
        sym_locals = dict(zip(coords, coord_syms))

        expr_matrix = []
        for row in metric:
            expr_row = []
            for entry in row:
                if isinstance(entry, str):
                    expr_row.append(sympy.sympify(entry, locals=sym_locals))
                else:
                    expr_row.append(sympy.sympify(entry))
            expr_matrix.append(expr_row)

        # 4) Lambdify to get a numeric g_func(x)
        f_metric = sympy.lambdify(coord_syms, expr_matrix, modules=["numpy"])
        def g_func(x_arr):
            raw = f_metric(*x_arr)
            return np.array(raw, dtype=float)

        # 5) Check singularity at eval_pt
        try:
            g_at_pt = g_func(np.array(eval_pt, dtype=float))
            eig = np.linalg.eigvals(g_at_pt)
            if np.any(np.isclose(eig, 0.0, atol=1e-12)):
                return Response(
                    {'success': False,
                     'error': (
                         f"The metric is singular at {eval_pt}. "
                         f"Eigenvalues: {eig.tolist()}. Try a different point."
                     )},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except Exception as e:
            logger.error(f"Error checking metric singularity: {e}", exc_info=True)

        # 6) Perform numeric calculations
        try:
            calc = NumericTensorCalculator(g_func, h=data.get('h', 1e-6))
            results = calc.compute_all(np.array(eval_pt, dtype=float))
            logger.info("NumericTensorCalculator succeeded")
        except Exception as e:
            logger.error(f"Error in NumericTensorCalculator: {e}", exc_info=True)
            return Response(
                {'success': False, 'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 7) Prepare response payload
        out = {
            'success': True,
            'dimension': n,
            'coordinates': coords,
            'evaluation_point': eval_pt,
            'metric': results['metric'].tolist(),
            'inverse_metric': results['metric_inv'].tolist(),
            'christoffelSymbols': results['christoffel'].tolist(),
            'riemannTensor': results['riemann_lower'].tolist(),
            'ricciTensor': results['ricci'].tolist(),
            'scalarCurvature': float(results['scalar']),
            'einsteinTensor': results['einstein_lower'].tolist(),
            'weylTensor': [],  # optional
        }

        # 8) Deduplicate and save if new
        existing = Tensor.objects.filter(
            dimension=n,
            coordinates=coords,
            metric_data=metric
        ).first()
        if existing:
            out['tensor_id'] = existing.id
            return Response(out, status=status.HTTP_200_OK)

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
