from django.db import models

# Create your models here.
class Tensor(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    components = models.JSONField(default=dict)
    description = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.name
