#!/bin/bash
set -e

# Display settings being used
echo "Using DJANGO_SETTINGS_MODULE: $DJANGO_SETTINGS_MODULE"

# For testing skip database check/connection
echo "Testing mode: Skipping database connection check"

# For testing, initialize SQLite database
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

# Start Gunicorn server
echo "Starting Gunicorn server..."
exec gunicorn myproject.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers ${GUNICORN_WORKERS:-2} \
    --timeout ${GUNICORN_TIMEOUT:-120} \
    --access-logfile - \
    --error-logfile - 