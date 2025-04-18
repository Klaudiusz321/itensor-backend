from django.contrib import admin
from django.urls import path, include
from calculator.views import calculate_view, health_check
from calculator.numerical_views import numerical_calculate_view
from calculator.symbolic_views import symbolic_calculation_view
# Import the new views (uncomment when implemented)
# from calculator.differential_operators_views import (
#     symbolic_differential_view,
#     numerical_differential_view, 
#     coordinate_transform_view
# )
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
    path('api/health/', health_check, name='health_check'),
    
    # Include the API app's URLs
    path('api/', include('api.urls')),
    
    # New differential operators endpoints (uncomment when implemented)
    # path('api/differential/symbolic/', symbolic_differential_view, name='symbolic_differential'),
    # path('api/differential/numerical/', numerical_differential_view, name='numerical_differential'),
    # path('api/coordinate/transform/', coordinate_transform_view, name='coordinate_transform'),
]

# Debug print all registered URLs
for url_pattern in urlpatterns:
    if hasattr(url_pattern, 'pattern'):
        logger.info(f"Registered URL: {url_pattern.pattern}")
    else:
        logger.info(f"Registered URL (include): {url_pattern}")
