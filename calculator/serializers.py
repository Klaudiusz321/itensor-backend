from rest_framework import serializers
from .models import Tensor

class TensorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tensor
        fields = ['id', 'name', 'created_at', 'components', 'description']
        read_only_fields = ['created_at'] 