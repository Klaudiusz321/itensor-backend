# myproject/celery.py
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

app = Celery('myproject')

# Wczytaj ustawienia Django (wszystkie z prefiksem CELERY_)
app.config_from_object('django.conf:settings', namespace='CELERY')

# Użyj zmiennych środowiskowych
app.conf.broker_url = os.environ.get('CELERY_BROKER_URL')
app.conf.result_backend = os.environ.get('CELERY_RESULT_BACKEND', app.conf.broker_url)

app.autodiscover_tasks()
