from django.urls import path
from . import views
from .numerical_views import calculate_schwarzschild_christoffel

urlpatterns = [
    path('calculate', views.calculate_view, name='calculate'),
    path('task_status/<str:task_id>/', views.task_status_view, name='task_status'),
    path('differential-operators/', views.differential_operators, name='differential_operators'),
    path('flrw-metric/', views.calculate_flrw_view, name='flrw_metric'),
    path('schwarzschild-christoffel/', calculate_schwarzschild_christoffel, name='schwarzschild_christoffel'),
]

   
