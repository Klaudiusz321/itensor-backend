from rest_framework import serializers
from .models import Tensor
import numpy as np

class TensorSerializer(serializers.ModelSerializer):
    # Make all extended fields optional by explicitly declaring them
    metric_hash = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    dimension = serializers.IntegerField(required=False)
    coordinates = serializers.JSONField(required=False)
    metric_data = serializers.JSONField(required=False)
    christoffel_symbols = serializers.JSONField(required=False)
    riemann_tensor = serializers.JSONField(required=False)
    ricci_tensor = serializers.JSONField(required=False)
    scalar_curvature = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    einstein_tensor = serializers.JSONField(required=False)
    
    class Meta:
        model = Tensor
        fields = [
            'id', 'name', 'created_at', 'components', 'description',
            'metric_hash', 'dimension', 'coordinates', 'metric_data',
            'christoffel_symbols', 'riemann_tensor', 'ricci_tensor',
            'scalar_curvature', 'einstein_tensor'
        ]
        read_only_fields = ['created_at']
    
    
    
    def validate(self, data):
        """
        Custom validation to handle missing columns in the database
        """
        # Get the list of valid fields by checking the model's concrete fields
        model_fields = [f.name for f in Tensor._meta.get_fields()]
        
        # Filter out fields that don't exist in the model
        filtered_data = {}
        for field_name, value in data.items():
            if field_name in model_fields:
                filtered_data[field_name] = value
        
        return filtered_data 
    