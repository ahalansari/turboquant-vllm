#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Find a compose command (Unraid: install the "Docker Compose Manager" plugin)
if docker compose version &>/dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE="docker-compose"
else
    echo "ERROR: docker compose not found."
    echo "On Unraid, install the 'Docker Compose Manager' plugin from Community Apps,"
    echo "or build without compose:  docker build -t aalansari/vllm-turboquant:gemma4 .."
    exit 1
fi

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example - edit it to change model/settings."
fi

echo "Building aalansari/vllm-turboquant:gemma4 (native $(uname -m))..."
$COMPOSE build

echo ""
echo "Build complete. Check the build output above for the two guard lines:"
echo "  transformers 5.5.0 gemma4 OK"
echo "  vllm 0.19.x OK"
echo ""
echo "Start the server:   $COMPOSE up -d"
echo "Follow logs:        $COMPOSE logs -f"
echo "Test the API:       curl http://localhost:\${HOST_PORT:-8000}/v1/models"
