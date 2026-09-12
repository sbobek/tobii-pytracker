#!/bin/bash
# Development script for hot reload during development

set -e

echo "Tobii PyTracker Docker Setup (Development Mode)"
echo "==================================================="

docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

echo ""
echo "Running in development mode with hot reload"
echo "Connect to VNC server at: localhost:5900"
