#!/bin/bash
# rebuild.sh - Rebuild and restart KGChat application

set -e

echo "Rebuilding KGChat application..."

# Stop all containers
docker-compose down

# Remove old images
echo "Removing old images..."
docker image rm app_backend app_frontend 2>/dev/null || true

# Clean up system
docker system prune -f

# Rebuild all images
echo "Building new images..."
docker-compose build --no-cache

# Start databases first
echo "Starting databases..."
docker-compose up -d neo4j qdrant ollama

# Wait for databases
echo "Waiting for databases..."
sleep 60

# Start backend
echo "Starting backend..."
docker-compose up -d backend

# Wait for backend
sleep 30

# Start frontend
echo "Starting frontend..."
docker-compose up -d frontend

# Show final status
echo "Rebuild complete. Application status:"
docker-compose ps

echo ""
echo "Application URLs:"
echo "Frontend: http://localhost:8501"
echo "Backend API: http://localhost:8000"
echo "Neo4j Browser: http://localhost:7475"
echo "Qdrant Dashboard: http://localhost:6333/dashboard"
