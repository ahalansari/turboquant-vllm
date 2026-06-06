ARG VLLM_TAG=v0.19.0
FROM vllm/vllm-openai:${VLLM_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "turboquant-vllm[vllm] @ git+https://github.com/ahalansari/turboquant-vllm.git"
RUN pip install --no-cache-dir --upgrade huggingface_hub

# Gemma 4 (model_type: gemma4) needs transformers >= 5.5.0, but vLLM 0.19.0
# pins transformers<5 — install the pinned release over it (expected pip
# conflict warning). Pinning also busts Docker's layer cache on bumps.
ARG TRANSFORMERS_VERSION=5.5.0
RUN pip install --no-cache-dir --upgrade "transformers==${TRANSFORMERS_VERSION}"

# Guards: gemma4 arch must be present, and the plugin install must not have
# replaced the base image's vLLM.
RUN python3 -c "import transformers; from transformers.models.gemma4 import Gemma4Config; print('transformers', transformers.__version__, 'gemma4 OK')"
RUN python3 -c "import vllm; assert vllm.__version__.startswith('0.19'), vllm.__version__; print('vllm', vllm.__version__, 'OK')"

# Patch vLLM's kv_cache_utils.py to handle non-divisible page sizes
# (Gemma 4 + TQ4 produces heterogeneous page sizes that aren't divisible)
COPY patch_kv_cache_utils.py /tmp/patch_kv_cache_utils.py
RUN python3 /tmp/patch_kv_cache_utils.py && rm /tmp/patch_kv_cache_utils.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
