from django.contrib import admin
from django.urls import path
from calculator.views import calculate_view, health_check

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/calculate/', calculate_view),
    path('api/calculate', calculate_view),
    path('api/health/', health_check, name='health_check'),
]
