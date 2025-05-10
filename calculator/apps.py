from django.apps import AppConfig


class CalculatorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'calculator'
    verbose_name = 'Calculator'

    def ready(self):
        pass  # Keep it simple for now
