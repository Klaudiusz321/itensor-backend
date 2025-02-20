from django.urls import path
from .views import calculate

urlpatterns = [
    path('calculate', calculate, name='calculate'),
]
