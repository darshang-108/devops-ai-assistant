Deployment Process

The backend services run inside Docker containers.

Deployment steps:
1. Code is pushed to GitHub
2. CI builds a Docker image
3. The image is deployed to Kubernetes

Redis and PostgreSQL run as separate services.