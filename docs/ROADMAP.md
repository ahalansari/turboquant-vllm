# TurboQuant Consumer — Roadmap

Implementation status and path forward for TurboQuant KV cache compression
on consumer GPUs (RTX 4090, 24 GB VRAM) with Molmo2 vision-language models.

**Paper:** arXiv 2504.19874 — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)

---

## Completed

### Layer 1: Core Quantization Algorithm

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Lloyd-Max codebook solver | Done | 8 | Gaussian approx for d >= 64, `@lru_cache`d |
| `TurboQuantMSE` (Stage 1) | Done | 6 | Rotation + scalar quantize, ~95% cosine sim at 3-bit |
| `TurboQuantProd` (Stage 2) | Done | 5 | MSE + QJL correction, unbiased inner products |

### Layer 2a: Production Compressors

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `CompressedKeys` / `CompressedValues` | Done | — | Dataclass containers |
| `TurboQuantCompressorV2` (keys) | Done | 6 | Includes `asymmetric_attention_scores()` |
| `TurboQuantCompressorMSE` (values) | Done | 2 | MSE-only for value reconstruction |

### Layer 2b: Compressed KV Cache (real VRAM savings)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `TurboQuantKVCache` (accuracy-only) | Done | 7 | Compress-decompress round-trip, no VRAM savings |
| `CompressedDynamicCache` | Done | 13 | uint8 indices + fp32 norms, 1.94x compression |
| Benchmark harness (`--compressed`) | Done | — | A/B testing with Molmo2, VRAM + quality metrics |

### Experiments

| # | Date | Result | Key Finding |
|---|------|--------|-------------|
| 001 | 2026-03-25 | Failed | TurboQuantProd (2-bit MSE + 1-bit QJL) garbled output — QJL wasted in drop-in mode |
| 002 | 2026-03-25 | Passed | MSE-only fix: identical text output, coherent video (1.3x overhead) |
| 003 | 2026-03-25 | Passed | CompressedDynamicCache: coherent output, 1.94x compression, fp16 norms bug found and fixed |

---

### P1: Long-sequence regression test

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Multi-layer precision test | Done | 4 | 36 layers, 1024 prefill + 32 gen steps, >0.999 cosine sim |
| TQ4 regression at scale | Done | 1 | Same scale test for 4-bit nibble-packed path |

### P2: TQ4 nibble packing (3.76x compression)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `_nibble_pack` / `_nibble_unpack` | Done | 1 | Bit-shift pack/unpack, exact round-trip verified |
| `CompressedDynamicCache` bits=4 | Done | 7 | Auto-enabled at bits=4, transparent to callers |
| `_CompressedLayer.packed` flag | Done | — | Tracks packing format through cat/stats |

### Compression Summary

| Mode | Bytes/block | Compression | Quality | Status |
|------|-------------|-------------|---------|--------|
| FP16 baseline | 256 | 1.0x | — | — |
| TQ3 uint8 | 132 | 1.94x | ~95% cosine | Done |
| **TQ4 nibble** | **68** | **3.76x** | **~97% cosine** | **Done** |
| TQ3 bit-packed | 52 | 4.92x | ~95% cosine | Deferred (P5) |

**Projected VRAM for Molmo2-4B (36 layers, 8 KV heads, 11K tokens):**

| Mode | KV Cache Size | Savings vs FP16 |
|------|--------------|-----------------|
| FP16 baseline | 1,639 MiB | — |
| TQ3 uint8 | 845 MiB | 794 MiB (1.94x) |
| **TQ4 nibble** | **436 MiB** | **1,203 MiB (3.76x)** |

---

## Next Up

### P3: Incremental dequantization (eliminate decode overhead)

**Goal:** Eliminate the 3.36x decode overhead by dequantizing only NEW tokens instead of the entire cache at every layer at every step.

**Why:** The current `CompressedDynamicCache` dequantizes ALL 11K+ cached tokens at every layer at every decode step via a 128x128 rotation matmul. The overhead is dominated by memory churn: `indices.long()` allocates 88 MB temporary tensors per layer (36 layers = 3.2 GB of allocations per decode step).

**Approach:** Maintain a running decompressed key buffer per layer. On each decode step:
1. Dequantize ONLY the 1 new token (microseconds)
2. `torch.cat` onto the running buffer
3. Free the previous layer's buffer (same one-layer-at-a-time pattern)
4. SDPA gets the full decompressed K without re-dequantizing 11K tokens

**Expected result:** Decode overhead drops from 3.36x to ~1.0x (near baseline). SDPA precision maintained across all 36 layers (proven by unfused TQ4 path). 3.76x VRAM compression preserved.

**Effort:** ~30 minutes. Changes only `CompressedDynamicCache._compressed_update()`.

### P3b: Fused Triton Q@K^T kernel (validated, needs Flash Attention fusion)

| Component | Status | Notes |
|-----------|--------|-------|
| Fused Q@K^T Triton kernel | **Done** | Nibble unpacking + pre-rotation trick, 17.8x speedup |
| Micro-benchmark (11K tokens) | **Done** | 1.0 cosine similarity vs unfused reference |
| Single-layer Molmo2-4B integration | **Done** | Correct output with fused kernel |
| Multi-layer integration | **Blocked** | Needs Flash Attention-style fusion (see below) |

**Key finding:** A fused Q@K^T-only kernel does not match SDPA precision when composed across all 36 transformer layers. The fp32 kernel scores differ from bf16 SDPA scores by 0.023 cosine per layer, which compounds into degenerate output. Full Flash Attention-style fusion (Q@K^T + online softmax + V matmul in one kernel) is required for multi-layer correctness. This is a 1-2 week project using the Triton Flash Attention tutorial as scaffold.

---

## Future Work

### P4: Molmo2-8B validation

**Goal:** Confirm CompressedDynamicCache works with the larger model. The 8B model recognizes character names (e.g., "Elaine", "Kramer") which 4B cannot.

**Approach:** Run benchmark with `--model allenai/Molmo2-8B --compressed`. May need `bitsandbytes` 4-bit weight quantization to fit model + compressed cache in 24 GB.

**When:** After P3 (incremental dequant) eliminates the decode overhead.

### P5: Flash Attention-style fused kernel (research)

**Goal:** Fuse the full attention computation (Q@K^T + online softmax + V matmul) into a single Triton kernel that reads nibble-packed indices directly.

**Why:** The P3b Q@K^T-only kernel achieves 17.8x on the micro-benchmark but can't maintain SDPA precision across 36 layers. A full Flash Attention-style kernel would match SDPA's online softmax behavior while reading compressed keys.

**Complexity:** 1-2 weeks. Use the Triton Flash Attention tutorial as scaffold, inject centroid gather + nibble unpack into the inner tile loop.

### P6: TQ3 bit-packing (research, nice-to-have)

**Goal:** Pack 3-bit indices at the theoretical optimum (48 bytes per 128 indices, 4.92x compression).

**Why deferred:** 3-bit indices cross byte boundaries, making parallel pack/unpack non-trivial. No PyTorch/Triton implementation exists — only C/CUDA (ik_llama.cpp). The 30% improvement over TQ4 nibble (4.92x vs 3.76x) doesn't justify the complexity until the easier wins are shipped.

### P6: vLLM native integration

**Goal:** Run TurboQuant as a vLLM KV cache backend, enabling compressed caching in production serving.

**Status:** vLLM currently supports FP8 KV cache only. No integer sub-byte support. The KV Offloading Connector API (Jan 2026) handles offloading tiers, not quantization. Integrating TurboQuant would require attention backend changes, not just connector API.

**Our path:** Monitor upstream. When vLLM ships TurboQuant support (expected Q2-Q3 2026), we adopt on day one with confidence from our validation work. Our codebase could also serve as a reference implementation for a vLLM contribution.

---

## Hardware Context

| Component | Spec | Relevance |
|-----------|------|-----------|
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X) | All benchmarks run here |
| CPU | AMD 7800X3D | Codebook solving, data loading |
| RAM | 128 GB DDR5 | Model offloading when needed |
| Target model | Molmo2-4B (experiments) / 8B (future) | Vision-language model for video analysis |
| Target workload | Seinfeld clip analysis | 11K+ visual tokens at 2fps |
| Production stack | vLLM in Podman (CDI GPU) | Currently FP8 KV cache |

---

## Key Lessons

1. **FP16 norms are a trap.** At 10K+ token sequences across 36 layers, fp16 norm precision loss compounds and flips low-confidence logits. Always use fp32 for norms.

2. **QJL is invisible in drop-in mode.** Standard attention does `Q @ K.T` on decompressed keys. QJL correction only helps with `estimate_inner_product()` (custom kernel). Using QJL in drop-in mode wastes 1 bit of MSE resolution for nothing.

3. **Peak VRAM != KV cache size.** On Molmo2-4B with 11K tokens, forward-pass activations dominate peak VRAM (~90%). KV cache compression savings are real but invisible to `max_memory_allocated()`. They matter for max_model_len budgeting, not peak measurement.

4. **PyTorch treats uint8 as boolean masks.** Fancy indexing with uint8 tensors triggers boolean masking, not integer indexing. Always cast to `.long()` before centroid lookup.

5. **Don't fight byte alignment.** TQ4 nibble packing (2 values per byte) is trivial and gives 3.76x compression. TQ3 bit-packing (3-bit byte-crossing) is hard and only 30% better. Work with the hardware, not against it.

6. **No PyTorch sub-byte ecosystem.** `torch.uint3` etc. are placeholders with no ops. TorchAO packing is weight-quant-specific. Every KV cache implementation rolls its own Triton kernels. Plan accordingly.
