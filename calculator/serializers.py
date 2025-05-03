from rest_framework import serializers
from .models import Tensor

class TensorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tensor
        fields = [
            'id', 'name', 'created_at', 'components', 'description',
            'metric_hash', 'dimension', 'coordinates', 'metric_data',
            'christoffel_symbols', 'riemann_tensor', 'ricci_tensor',
            'scalar_curvature', 'einstein_tensor'
        ]
        read_only_fields = ['created_at', 'metric_hash'] 