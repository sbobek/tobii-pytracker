#!/bin/bash
set -e

HOST_UID=$(id -u)
HOST_GID=$(id -g)
echo "Using UID=$HOST_UID, GID=$HOST_GID for file permissions"
echo ""
export HOST_UID HOST_GID

echo "Creating output directory and adding group permission to read and write"
mkdir -p output
chown -R $HOST_UID:$HOST_GID output
chmod u+rwx output
chmod g+rwx output
chmod g+s output


if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "Docker Compose is not installed. Please install Docker Compose first."
  exit 1
fi

echo "Tobii PyTracker Docker Setup"
echo "================================"

if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

echo "Docker and Docker Compose are installed"
echo ""

HOST_UID=$(id -u)
HOST_GID=$(id -g)
echo "Using UID=$HOST_UID, GID=$HOST_GID for file permissions"
echo ""
export HOST_UID HOST_GID

echo "Starting Tobii PyTracker..."
$COMPOSE_CMD up --build

echo ""
echo "Application stopped."
echo "Connect to VNC server at: localhost:5900"
