# myproject/urls.py

from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
import logging

# DRF ViewSet dla /api/tensors/
from calculator.views.tensor_viewset import TensorViewSet

# „Klejone” widoki (pozostałe, nie kolidujące)
from calculator.views.views import (
    health_check,
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

    # wszystkie endpointy DRF dla TensorViewSet, w tym:
    #   GET/POST    /api/tensors/
    #   POST        /api/tensors/numerical/
    #   POST        /api/tensors/symbolic/
    #   POST        /api/tensors/differential-operators/
    #   POST        /api/tensors/find-similar/
    path('api/', include(router.urls)),

    # proste endpointy „funkcyjne” (nie kolidują)
    path('api/health/', health_check, name='health_check'),

    # MHD
    path('api/mhd/simulation/',  mhd_simulation,   name='mhd_simulation'),
    path('api/mhd/snapshot/',    mhd_snapshot,     name='mhd_snapshot'),
    path('api/mhd/field-plots/', mhd_field_plots,  name='mhd_field_plots'),
]

# (opcjonalnie) debug: wypisz wszystkie zarejestrowane ścieżki
for entry in urlpatterns:
    try:
        logger.info(f"URL: {entry.pattern}")
    except Exception:
        logger.info(f"Include/Router entry: {entry}")
