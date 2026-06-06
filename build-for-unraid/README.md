# Build on Unraid

Builds the `aalansari/vllm-turboquant:gemma4` image natively on the Unraid box (linux/amd64, no emulation) and runs it via docker compose. The base image unpacks to ~30GB — make sure the Docker vDisk has room (Settings → Docker).

## Prerequisites

- Nvidia Driver plugin (Community Apps)
- Docker Compose Manager plugin (Community Apps) — or any `docker compose`/`docker-compose` on PATH

## Steps

```bash
git clone https://github.com/ahalansari/turboquant-vllm.git
cd turboquant-vllm/build-for-unraid
./build-unraid.sh        # creates .env from .env.example on first run, then builds
docker compose up -d
docker compose logs -f   # watch model download + startup
curl http://localhost:8000/v1/models
```

## Configuration

Edit `.env` (created from `.env.example` on first build). Defaults match the working Gemma 4 deployment: `lcu0312/gemma-4-26B-A4B-it-AWQ-4bit`, AWQ, TP=2, TurboQuant 4/4-bit, Gemma 4 tool-calling enabled via `EXTRA_ARGS`.

Models cache to `/mnt/user/appdata/vllm-turboquant/hf-cache` (override with `HF_CACHE_DIR`).

## Updating

```bash
cd turboquant-vllm && git pull
cd build-for-unraid && ./build-unraid.sh && docker compose up -d
```

A `TRANSFORMERS_VERSION` or `VLLM_TAG` bump in `.env` automatically invalidates the right Docker layer — no `--no-cache` needed.
