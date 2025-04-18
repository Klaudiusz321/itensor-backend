from django.urls import path
from api.views import differential_operators

urlpatterns = [
    # ... existing URLs ...
    path('differential-operators/', differential_operators, name='differential_operators'),
] 