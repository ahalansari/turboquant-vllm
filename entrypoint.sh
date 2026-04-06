#!/bin/bash
set -e

# === Defaults optimized for RTX 3060 12GB ===
MODEL="${MODEL:-QuantTrio/Qwen3.5-9B-AWQ}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
QUANTIZATION="${QUANTIZATION:-awq}"
DTYPE="${DTYPE:-half}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# TurboQuant KV cache compression via plugin (--attention-backend CUSTOM)
TURBOQUANT="${TURBOQUANT:-true}"
export TQ4_K_BITS="${TQ4_K_BITS:-4}"
export TQ4_V_BITS="${TQ4_V_BITS:-4}"

echo "=== vLLM TurboQuant ==="
echo "Model:            $MODEL"
echo "TurboQuant:       $TURBOQUANT (K=${TQ4_K_BITS}bit V=${TQ4_V_BITS}bit)"
echo "Max Context:      $MAX_MODEL_LEN"
echo "GPU Mem Util:     $GPU_MEMORY_UTIL"
echo "Quantization:     $QUANTIZATION"
echo "Tensor Parallel:  $TENSOR_PARALLEL"
echo "Thinking:         $ENABLE_THINKING"
echo "Port:             $PORT"
echo "======================="

# Build argument list
QUANT_ARGS=""
if [[ "$QUANTIZATION" != "auto" ]]; then
    QUANT_ARGS="--quantization $QUANTIZATION"
fi

TQ_ARGS=""
if [[ "$TURBOQUANT" == "true" ]]; then
    TQ_ARGS="--attention-backend CUSTOM"
fi

exec vllm serve "$MODEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --default-chat-template-kwargs "{\"enable_thinking\": $ENABLE_THINKING}" \
    $QUANT_ARGS \
    $TQ_ARGS \
    --dtype "$DTYPE" \
    --port "$PORT" \
    --enforce-eager \
    --trust-remote-code \
    $EXTRA_ARGS
