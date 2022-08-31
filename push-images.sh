#!/usr/bin/env bash
set -euo pipefail

# docker build -f Dockerfile_export_data.api -t go2joy_export_data:latest .

DOCKER_USERNAME='anhlbt'
DOCKER_PASSWORD='anhtuan3.'
IMAGE_NAME=$1
TAG_NAME='latest'
IMAGE_ORG='anhlbt'

echo "${DOCKER_PASSWORD}" | docker login -u="${DOCKER_USERNAME}" --password-stdin

docker tag "${IMAGE_NAME}:${TAG_NAME}" "${IMAGE_ORG}/${IMAGE_NAME}:${TAG_NAME}"
docker push "${IMAGE_ORG}/${IMAGE_NAME}:${TAG_NAME}"


