#!/bin/bash
# down.sh - Stop KGChat application

set -e

echo "Stopping KGChat application..."

# Stop all containers
docker-compose down

# Clean up
echo "Cleaning up..."
docker container prune -f

echo "Application stopped successfully."