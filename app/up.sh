#!/bin/bash
# up.sh - Start KGChat application

set -e

echo "Starting KGChat application..."

# Stop any existing containers
docker-compose down

# Start databases first
echo "Starting databases..."
docker-compose up -d neo4j qdrant ollama

# Wait for databases to be ready
echo "Waiting for databases to initialize..."
sleep 60

# Check database health
echo "Checking database connectivity..."
curl -f http://localhost:6333/health || echo "Qdrant not ready yet"
curl -f http://localhost:11434/api/version || echo "Ollama not ready yet"

# Start backend
echo "Starting backend..."
docker-compose up -d backend

# Wait for backend
sleep 30

# Start frontend
echo "Starting frontend..."
docker-compose up -d frontend

# Show status
echo "Application status:"
docker-compose ps

echo ""
echo "Application URLs:"
echo "Frontend: http://localhost:8501"
echo "Backend API: http://localhost:8000"
echo "Neo4j Browser: http://localhost:7475 (user: neo4j, pass: password)"
echo "Qdrant Dashboard: http://localhost:6333/dashboard"


