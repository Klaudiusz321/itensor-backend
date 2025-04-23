"""
Production settings for myproject.
"""
import os
from pathlib import Path
from .settings_base import *  # Import from base settings that don't use django-heroku

# SECURITY SETTINGS
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
SECRET_KEY = os.environ.get('SECRET_KEY', 'fallback-key-for-non-production-environment')

# Only use the fallback key for development or testing
if SECRET_KEY == 'fallback-key-for-non-production-environment':
    import warnings
    warnings.warn('Using fallback SECRET_KEY. Set SECRET_KEY environment variable in production.')

# ALLOWED HOSTS
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost').split(',')

# DATABASE CONFIGURATION - Using SQLite
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# CORS SETTINGS
CORS_ALLOWED_ORIGINS = [
    os.environ.get('FRONTEND_URL', 'https://itensor.online'),
]

# If set, allow all origins - use with caution
if os.environ.get('CORS_ALLOW_ALL', 'False').lower() == 'true':
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOW_ALL_ORIGINS = False

# CSRF SETTINGS
CSRF_TRUSTED_ORIGINS = [
    os.environ.get('FRONTEND_URL', 'https://itensor.online'),
]

# STATIC FILES
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# LOGGING
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/app/logs/django.log' if os.path.exists('/app/logs') else os.path.join(BASE_DIR, 'logs', 'django.log'),
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
            'level': 'INFO',
            'propagate': False,
        },
    }
}

# Try to create logs directory, but don't fail if we can't (e.g., in read-only filesystem)
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