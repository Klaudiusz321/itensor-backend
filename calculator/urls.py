from django.urls import path
from . import views

urlpatterns = [
    path('api/calculate', views.calculate_view, name='calculate'),
    path('api/visualize', views.visualize_view, name='visualize')
]
