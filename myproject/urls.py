from django.contrib import admin
from django.urls import path
from calculator.views import calculate_view, health_check
from calculator.numerical_views import numerical_calculate_view
from calculator.symbolic_views import symbolic_calculation_view

# Define all URL patterns for the tensor calculator API
# Note: We maintain both 'numerical' and 'numeric' endpoints for backward compatibility
# 'numerical' is the primary endpoint, 'numeric' is an alias that some frontend components may use
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/calculate/', calculate_view),
    path('api/calculate', calculate_view),
    path('api/tensors/numerical/', numerical_calculate_view, name='numerical_calculation'),
    path('api/tensors/numerical', numerical_calculate_view, name='numerical_calculation_no_slash'),
    path('api/tensors/numeric/', numerical_calculate_view, name='numerical_calculation_alias'),  # Alias for common misspelling
    path('api/tensors/numeric', numerical_calculate_view, name='numerical_calculation_alias_no_slash'),  # Alias without trailing slash
    path('api/tensors/symbolic/', symbolic_calculation_view, name='symbolic_calculation'),
    path('api/tensors/symbolic', symbolic_calculation_view, name='symbolic_calculation_no_slash'),
    path('api/health/', health_check, name='health_check'),
]
