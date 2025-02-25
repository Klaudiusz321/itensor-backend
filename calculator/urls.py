from django.urls import path
from .views import calculate, visualize_view

urlpatterns = [
    path('calculate/', calculate, name='calculate'),
    path('visualize/', visualize_view, name='visualize'),
]
