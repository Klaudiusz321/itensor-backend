from django.db import models
import hashlib
import json

# Create your models here.
class Tensor(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    components = models.JSONField(default=dict)
    description = models.TextField(blank=True, null=True)
    
    # Fields for caching metric calculations
    metric_hash = models.CharField(max_length=64, db_index=True, blank=True, null=True)
    dimension = models.IntegerField(default=4)
    coordinates = models.JSONField(default=list, blank=True, null=True)
    metric_data = models.JSONField(default=list, blank=True, null=True)
    christoffel_symbols = models.JSONField(default=list, blank=True, null=True)
    riemann_tensor = models.JSONField(default=list, blank=True, null=True)
    ricci_tensor = models.JSONField(default=list, blank=True, null=True)
    scalar_curvature = models.CharField(max_length=255, blank=True, null=True)
    einstein_tensor = models.JSONField(default=list, blank=True, null=True)

    def save(self, *args, **kwargs):
        # build a stable JSON from dimension, coordinates and metric_data
        payload = {
            'dimension': self.dimension,
            'coordinates': self.coordinates or [],
            'metric': self.metric_data or []
        }
        # compute hash
        json_string = json.dumps(payload, sort_keys=True)
        self.metric_hash = hashlib.sha256(json_string.encode('utf-8')).hexdigest()
        super().save(*args, **kwargs)
    
    @staticmethod
    def generate_metric_hash(dimension, coordinates, metric_data):
        """
        Generate a hash for a specific metric configuration to use as a cache key
        """
        # Convert all inputs to a stable string representation
        data_to_hash = {
            'dimension': dimension,
            'coordinates': coordinates,
            'metric': metric_data
        }
        
        # Convert to a stable JSON string (sorted keys)
        json_string = json.dumps(data_to_hash, sort_keys=True)
        
        # Generate SHA-256 hash
        return hashlib.sha256(json_string.encode()).hexdigest()
    
    def __str__(self):
        return self.name
