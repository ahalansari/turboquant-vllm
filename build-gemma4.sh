#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-aalansari/vllm-turboquant}"
IMAGE_TAG="${IMAGE_TAG:-gemma4}"
VLLM_TAG="${VLLM_TAG:-v0.19.0}"
FULL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building ${FULL_TAG} (base: vllm/vllm-openai:${VLLM_TAG})..."

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
echo "  docker run --runtime nvidia --gpus all --ipc=host -p 8000:8000 \\"
echo "    -e MODEL=cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \\"
echo "    -e QUANTIZATION=auto \\"
echo "    -e MAX_MODEL_LEN=32768 \\"
echo "    -e TENSOR_PARALLEL=2 \\"
echo "    -e GPU_MEMORY_UTIL=0.90 \\"
echo "    -e DTYPE=half \\"
echo "    -e TURBOQUANT=true \\"
echo "    ${FULL_TAG}"
echo ""
echo "To test the API:"
echo "  curl http://localhost:8000/v1/models"
