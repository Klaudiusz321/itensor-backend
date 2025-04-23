#!/bin/bash
set -e

# Display settings being used
echo "Using DJANGO_SETTINGS_MODULE: $DJANGO_SETTINGS_MODULE"

# Wait for database to be ready
echo "Checking database connection..."
python -c "
import sys
import time
import psycopg2

max_wait = 60
start = time.time()

while True:
    try:
        conn = psycopg2.connect(
            dbname='itensordb',
            user='itensoruser',
            password='Leskowiec242!',
            host='10.0.0.51',
            port='5432'
        )
        conn.close()
        print('Database connection successful')
        break
    except Exception as e:
        elapsed = time.time() - start
        if elapsed > max_wait:
            print(f'Could not connect to database after {max_wait} seconds: {e}')
            sys.exit(1)
        print(f'Waiting for database... ({int(elapsed)}s)')
        time.sleep(2)
"

# Run database migrations
echo "Running database migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

# Start Gunicorn server
echo "Starting Gunicorn server..."
exec gunicorn myproject.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers ${GUNICORN_WORKERS:-3} \
    --timeout ${GUNICORN_TIMEOUT:-120} \
    --access-logfile - \
    --error-logfile - 