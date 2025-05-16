from rest_framework import serializers
from .models import Tensor

# ------------------- Tensor Model Serializer -------------------
class TensorSerializer(serializers.ModelSerializer):
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
        # Filter out any keys not in the model
        model_fields = {f.name for f in Tensor._meta.get_fields()}
        return {k: v for k, v in data.items() if k in model_fields}


# ------------------- MHD Simulation Serializers -------------------

class GeneralParametersSerializer(serializers.Serializer):
    grid_x    = serializers.IntegerField(min_value=1)
    grid_y    = serializers.IntegerField(min_value=1)
    grid_z    = serializers.IntegerField(min_value=1)
    duration  = serializers.FloatField()
    time_step = serializers.FloatField()
    method    = serializers.IntegerField()

class FluidPropertiesSerializer(serializers.Serializer):
    initial_density     = serializers.FloatField()
    initial_pressure    = serializers.FloatField()
    initial_temperature = serializers.FloatField()
    # Extend with function mode or spatial gradients as needed

class MagneticFieldSerializer(serializers.Serializer):
    initial_magnetic_field = serializers.ListField(
        child=serializers.FloatField(), min_length=3, max_length=3
    )
    magnetic_conductivity  = serializers.FloatField()
    magnetic_viscosity     = serializers.FloatField()

class BoundaryConditionsSerializer(serializers.Serializer):
    boundary_type              = serializers.IntegerField()
    custom_boundary_conditions = serializers.DictField(child=serializers.CharField(), required=False)

class SimulationModeSerializer(serializers.Serializer):
    simulation_mode       = serializers.IntegerField()
    disturbance_settings  = serializers.DictField(required=False)
    dynamic_field_changes = serializers.DictField(required=False)
    random_disturbances   = serializers.DictField(required=False)

class MHDSimulationConfigSerializer(serializers.Serializer):
    """
    Wraps all MHD simulation parameters:
      - general (grid, time, method),
      - fluid_properties (density, pressure, temperature),
      - magnetic_field, boundary, simulation_mode.
    """
    general          = GeneralParametersSerializer()
    fluid_properties = FluidPropertiesSerializer()
    magnetic_field   = MagneticFieldSerializer()
    boundary         = BoundaryConditionsSerializer()
    simulation_mode  = SimulationModeSerializer()

    def validate(self, data):
        # Optional cross-parameter checks can go here
        return data
