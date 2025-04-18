from django.urls import path
from . import views

urlpatterns = [
    path('calculate', views.calculate_view, name='calculate'),
    path('task_status/<str:task_id>/', views.task_status_view, name='task_status'),
    path('differential-operators/', views.differential_operators, name='differential_operators'),
]
