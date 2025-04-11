from django.contrib import admin
from django.urls import path
from calculator.views import calculate_view, health_check
from calculator.numerical_views import numerical_calculate_view
from calculator.symbolic_views import symbolic_calculation_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/calculate/', calculate_view),
    path('api/calculate', calculate_view),
    path('api/tensors/numerical/', numerical_calculate_view),
    path('api/tensors/numerical', numerical_calculate_view),
    path('api/tensors/symbolic/', symbolic_calculation_view, name='symbolic_calculation'),
    path('api/tensors/symbolic', symbolic_calculation_view, name='symbolic_calculation_no_slash'),
    path('api/health/', health_check, name='health_check'),
]
