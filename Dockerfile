ARG VLLM_TAG=v0.19.0
FROM vllm/vllm-openai:${VLLM_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "turboquant-vllm[vllm] @ git+https://github.com/ahalansari/turboquant-vllm.git"
RUN pip install --no-cache-dir --upgrade huggingface_hub
RUN pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/huggingface/transformers.git

# Patch vLLM's kv_cache_utils.py to handle non-divisible page sizes
# (Gemma 4 + TQ4 produces heterogeneous page sizes that aren't divisible)
COPY patch_kv_cache_utils.py /tmp/patch_kv_cache_utils.py
RUN python3 /tmp/patch_kv_cache_utils.py && rm /tmp/patch_kv_cache_utils.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
