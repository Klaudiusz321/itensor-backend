# myproject/urls.py

from django.contrib import admin
from django.urls import path
from rest_framework.routers import DefaultRouter
import logging
from django.urls import path, include
# DRF ViewSet dla /api/tensors/
from calculator.views.tensor_viewset import TensorViewSet

# „Klejone" widoki
from calculator.views.views import (
    health_check,
    differential_operators,
    numerical_calculate_view,
    symbolic_calculate_view
)
from calculator.views.mhd.mhd import (
    mhd_simulation,
    mhd_snapshot,
    mhd_field_plots
)

logger = logging.getLogger(__name__)

# --- DRF router ---
router = DefaultRouter()
router.register(r'tensors', TensorViewSet)

urlpatterns = [
    # panel administracyjny
    path('admin/', admin.site.urls),

    # wszystkie endpointy DRF dla TensorViewSet:
    # GET/POST /api/tensors/  etc.
   
    path('api/', include(router.urls)),
    # proste endpointy „funkcyjne"
    path('api/health/',                health_check,            name='health_check'),
    path('api/tensors/numerical/',     numerical_calculate_view, name='numerical_calculation'),
    path('api/tensors/numeric/',       numerical_calculate_view, name='numerical_calculation_alias'),
    path('api/tensors/symbolic/',      symbolic_calculate_view, name='symbolic_calculation'),
    path('api/differential-operators/', differential_operators,   name='differential_operators'),

    # MHD
    path('api/mhd/simulation/',    mhd_simulation,   name='mhd_simulation'),
    path('api/mhd/snapshot/',      mhd_snapshot,     name='mhd_snapshot'),
    path('api/mhd/field-plots/',   mhd_field_plots,  name='mhd_field_plots'),

    # Add a route for find-similar endpoint
    path('api/tensors/find-similar/', TensorViewSet.as_view({'post': 'find_similar'}), name='find_similar'),
]

# Debug: wypisz wszystkie zarejestrowane ścieżki
for entry in urlpatterns:
    try:
        logger.info(f"URL: {entry.pattern}")
    except Exception:
        logger.info(f"Include/Router entry: {entry}")
