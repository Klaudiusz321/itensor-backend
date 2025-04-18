from django.urls import path
from . import views

urlpatterns = [
    # ... existing URLs ...
    path('differential-operators/', views.differential_operators, name='differential_operators'),
] 