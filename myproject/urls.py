from django.contrib import admin
from django.urls import path, include
from calculator.views import calculate_view, health_check, calculate_flrw_view
from calculator.numerical_views import numerical_calculate_view
from calculator.symbolic_views import symbolic_calculation_view
from myproject.api.views import differential_operators, mhd_simulation, mhd_snapshot, mhd_field_plots
import logging

# Setup debug logging for URLs
logger = logging.getLogger(__name__)

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
    path('api/tensors/flrw/', calculate_flrw_view, name='flrw_calculation'),
    path('api/tensors/flrw', calculate_flrw_view, name='flrw_calculation_no_slash'),
    path('api/health/', health_check, name='health_check'),
    
    # Direct URL for differential operators
    path('api/differential-operators/', differential_operators, name='differential_operators'),
    
    # MHD API endpoints
    path('api/mhd/simulation/', mhd_simulation, name='mhd_simulation'),
    path('api/mhd/snapshot/', mhd_snapshot, name='mhd_snapshot'),
    path('api/mhd/field-plots/', mhd_field_plots, name='mhd_field_plots'),
]

# Debug print all registered URLs
for url_pattern in urlpatterns:
    if hasattr(url_pattern, 'pattern'):
        logger.info(f"Registered URL: {url_pattern.pattern}")
    else:
        logger.info(f"Registered URL (include): {url_pattern}")
