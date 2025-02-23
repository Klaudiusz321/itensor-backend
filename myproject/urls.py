from django.contrib import admin
from django.urls import path, include
from calculator.views import calculate

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('calculator.urls')),
    path('api/calculate/', calculate, name='calculate'),
]
