#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-aalansari/vllm-turboquant}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
VLLM_TAG="${VLLM_TAG:-v0.18.1}"
FULL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building ${FULL_TAG} (base: vllm/vllm-openai:${VLLM_TAG})..."

# If running on macOS, build for linux/amd64 (Unraid target)
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS - building for linux/amd64"
    docker buildx build --platform linux/amd64 --build-arg VLLM_TAG="${VLLM_TAG}" -t "${FULL_TAG}" --load "${SCRIPT_DIR}"
else
    docker build --build-arg VLLM_TAG="${VLLM_TAG}" -t "${FULL_TAG}" "${SCRIPT_DIR}"
fi

echo ""
echo "Build complete: ${FULL_TAG}"
echo ""
echo "To push to Docker Hub:"
echo "  docker push ${FULL_TAG}"
echo ""
echo "To run locally (requires NVIDIA GPU):"
echo "  docker run --runtime nvidia --gpus all --ipc=host -p 8000:8000 ${FULL_TAG}"
echo ""
echo "To test the API:"
echo "  curl http://localhost:8000/v1/models"
