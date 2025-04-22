# iTensor Oracle Cloud Deployment Guide

This guide outlines the steps to deploy the iTensor backend to Oracle Cloud Infrastructure (OCI).

## Prerequisites

- Oracle Cloud account with sufficient privileges
- Docker and Docker Compose installed on your local machine
- Git client installed
- Access to Oracle Cloud Container Registry or other container registry

## Deployment Steps

### 1. Set Up Environment Variables

Create a `.env` file based on the provided `.env.example`:

```bash
cp .env.example .env
# Edit the .env file with your specific configuration
```

Important variables to set:
- `SECRET_KEY`: A secure random string for Django
- `ALLOWED_HOSTS`: Your Oracle Cloud domain name
- `DATABASE_URL`: Your PostgreSQL database connection string
- `FRONTEND_URL`: URL of your frontend application

### 2. Build the Docker Image

```bash
docker build -t itensor-backend:latest .
```

**Important Note**: The application has been optimized specifically for Oracle Cloud deployment:
- Removed Heroku-specific dependencies
- Configured for PostgreSQL database (recommended for production)
- Proper static file handling with whitenoise

### 3. Test Locally (Optional)

```bash
docker run --env-file .env -p 8000:8000 itensor-backend:latest
```

Visit http://localhost:8000/api/health/ to ensure the service is running.

### 4. Push to Container Registry

```bash
# Log in to Oracle Cloud Container Registry
docker login <region-key>.ocir.io

# Tag the image
docker tag itensor-backend:latest <region-key>.ocir.io/<tenancy-namespace>/<repo-name>/itensor-backend:latest

# Push the image
docker push <region-key>.ocir.io/<tenancy-namespace>/<repo-name>/itensor-backend:latest
```

### 5. Deploy to Oracle Container Instance

1. **Navigate to Oracle Cloud Console**
   - Go to Developer Services > Container Instances

2. **Create Container Instance**
   - Name: `itensor-backend`
   - Shape: Choose appropriate shape (e.g., Flex Shape with 1 OCPU, 4GB memory)
   - VCN and Subnet: Select your network configuration

3. **Configure Container**
   - Image URL: `<region-key>.ocir.io/<tenancy-namespace>/<repo-name>/itensor-backend:latest`
   - Environment Variables: Copy from your `.env` file
   - Ports: Add 8000 for HTTP

4. **Configure Storage (Important)**
   - Add a volume mount for persistent logs
   - Mount path: `/app/logs`
   - Volume size: 10GB (or as needed)
   - This ensures logs are preserved across container restarts

5. **Create**
   - Review and create the container instance

### 6. Set Up Database (Optional)

If you're not using an existing database:

1. **Create an Oracle Autonomous Database or PostgreSQL Instance**
   - Configure size based on expected workload
   - Enable TLS/SSL for secure connections

2. **Configure Database Access**
   - Set up network access rules from your container instance
   - Create database user and password

3. **Update Environment Variables**
   - Update `DATABASE_URL` in your container instance config

### 7. Set Up Load Balancer (Optional)

1. **Create Load Balancer**
   - Create a public load balancer
   - Configure backends to point to your container instance
   - Set up health checks using `/api/health/` endpoint

2. **Configure SSL/TLS**
   - Add SSL certificate for secure HTTPS connections
   - Set up HTTP to HTTPS redirection

### 8. Set Up DNS (Optional)

Configure your domain to point to the Oracle Cloud load balancer IP.

## Monitoring and Maintenance

### Viewing Logs

```bash
# Get container ID
oci container-instances list

# View logs
oci container-instances get-container-logs --container-id <container-id>
```

You can also access application logs in the persistent volume you configured:
```bash
# SSH into the instance hosting your container
# Access logs at /app/logs/django.log
```

### Updating the Application

1. Build and push new Docker image
2. Update the container instance with the new image

## Troubleshooting

### Common Issues

- **Database Connection Errors**: Verify database credentials and connection string
- **CORS Issues**: Check CORS settings match your frontend URL
- **Container Not Starting**: Check container logs for errors
- **Missing Logs**: Ensure the volume for /app/logs is properly mounted

### Health Check Endpoint

The application exposes a `/api/health/` endpoint that can be used to verify the service is running correctly.

## Security Considerations

- Keep `.env` files secure and never commit them to version control
- Use a secure `SECRET_KEY` in production
- Limit `ALLOWED_HOSTS` to only the domains you're using
- Use HTTPS for all production traffic
- Review and restrict container permissions 