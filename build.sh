#!/bin/bash
set -e

API_DOCKERFILE=docker/api.Dockerfile
WORKER_DOCKERFILE=docker/worker.Dockerfile
API_IMAGE=closetx-api:latest
WORKER_IMAGE=closetx-worker:latest

echo "Building API image..."
docker build -f $API_DOCKERFILE -t $API_IMAGE .

echo "Building Worker image..."
docker build -f $WORKER_DOCKERFILE -t $WORKER_IMAGE .

echo "Docker images built successfully:"
docker images | grep closetx