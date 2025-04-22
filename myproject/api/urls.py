from django.urls import path
from api.views import differential_operators, mhd_simulation, mhd_snapshot, mhd_field_plots, mhd_diagnostic, mhd_stress_test

urlpatterns = [
    # ... existing URLs ...
    path('differential-operators/', differential_operators, name='differential_operators'),
    path('mhd-simulation/', mhd_simulation, name='mhd_simulation'),
    path('mhd-snapshot/', mhd_snapshot, name='mhd_snapshot'),
    path('mhd-field-plots/', mhd_field_plots, name='mhd_field_plots'),
    path('mhd-diagnostic/', mhd_diagnostic, name='mhd_diagnostic'),
    path('mhd-stress-test/', mhd_stress_test, name='mhd_stress_test'),
] 