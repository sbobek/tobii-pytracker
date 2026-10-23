#!/bin/bash
# Script to stop the Tobii PyTracker Docker container

set -e

echo "Stopping Tobii PyTracker..."
docker-compose down

echo "Container stopped"
