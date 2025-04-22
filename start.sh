#!/bin/bash
# Combined service starter for iTensor app

# Function to cleanup processes on exit
cleanup() {
    echo "Stopping all services..."
    kill $(jobs -p)
    exit
}

# Setup signal trapping for clean exit
trap cleanup SIGINT SIGTERM

echo "Starting iTensor application..."

# Setup backend
cd /app/backend
echo "Setting up Django backend..."
python manage.py collectstatic --noinput
python manage.py migrate --noinput

# Start Django backend
echo "Starting Django backend on port 8000..."
gunicorn myproject.wsgi:application --bind 127.0.0.1:8000 --workers 3 --log-level info &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Start Next.js frontend
cd /app/frontend
echo "Starting Next.js frontend on port 3000..."
node_modules/.bin/next start -p 3000 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Configure and start Nginx
echo "Starting Nginx web server..."
nginx -g "daemon off;" &
NGINX_PID=$!
echo "Nginx started with PID: $NGINX_PID"

echo "All services started successfully!"
echo "The application is available at http://localhost"

# Keep the container running until terminated
wait 