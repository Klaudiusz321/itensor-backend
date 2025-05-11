# myproject/urls.py

from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
import logging

from calculator.views.tensor_viewset import TensorViewSet
from calculator.views.views import health_check
from calculator.views.views import differential_operators  # Import the direct view
from calculator.views.mhd_views import MHDSimulationView

logger = logging.getLogger(__name__)

router = DefaultRouter()
router.register(r'tensors', TensorViewSet, basename='tensor')

urlpatterns = [
    path('admin/', admin.site.urls),

    # wszystkie endpointy DRF dla TensorViewSet wraz z wszystkimi akcjami:
    #   GET  /api/tensors/
    #   POST /api/tensors/              – domyślne create (bez obliczeń)
    #   POST /api/tensors/find-similar/
    #   POST /api/tensors/symbolic/
    #   POST /api/tensors/numerical/
    #   POST /api/tensors/differential-operators/
    path('api/', include(router.urls)),

    path('api/health/', health_check, name='health_check'),
    
    # Add direct path for differential operators (old URL)
    path('api/differential-operators/', differential_operators, name='differential_operators'),
    #MHD
    path('api/mhd/simulation/', MHDSimulationView.as_view(), name='mhd_simulation'),
]

# (opcjonalnie) debug: wypisz wszystkie zarejestrowane ścieżki
for entry in urlpatterns:
    try:
        logger.info(f"URL: {entry.pattern}")
    except Exception:
        logger.info(f"Include/Router entry: {entry}")
