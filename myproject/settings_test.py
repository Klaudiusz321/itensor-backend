"""
Django settings for myproject project - Testing version.
"""
from pathlib import Path
import os

# Import from base settings instead of original settings with django-heroku
from .settings_base import *

# Ensure we're using SQLite for testing
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Enable debug mode for testing
DEBUG = True

# Logging for testing
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {module} {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'django-test.log'),
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'calculator': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    }
}

# Create logs directory if it doesn't exist
try:
    logs_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
except (IOError, OSError):
    # Failed to create directory, log to console only
    print("Warning: Could not create logs directory. Logging to console only.")
    for logger in LOGGING['loggers'].values():
        if 'file' in logger.get('handlers', []):
            logger['handlers'].remove('file')
    if 'file' in LOGGING['root'].get('handlers', []):
        LOGGING['root']['handlers'].remove('file')

# CORS settings
CORS_ALLOWED_ORIGINS = [
    "https://itensor.online",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
CORS_ALLOW_ALL_ORIGINS = True  # Only use during development
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]
CORS_EXPOSE_HEADERS = ['Content-Type', 'X-CSRFToken']

# CSRF settings
CSRF_TRUSTED_ORIGINS = [
    "https://calculator1-fc4166db17b2.herokuapp.com",
    "https://itensor.online",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Static files (CSS, JavaScript, Images)
BASE_DIR_STR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_ROOT = os.path.join(BASE_DIR_STR, 'staticfiles')
STATIC_URL = '/static/'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Performance settings
VISUALIZE_TIMEOUT = 300
DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB

# Prevent slash appending
APPEND_SLASH = False 