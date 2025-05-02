from django.contrib import admin
from .models import Tensor

# Register your models here.
@admin.register(Tensor)
class TensorAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    search_fields = ('name',)
    list_filter = ('created_at',)
