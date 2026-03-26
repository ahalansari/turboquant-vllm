## Experiment: 005 — Incremental dequantization (Molmo2-4B + video)

**Date:** 2026-03-26
**Hardware:** RTX 4090 (24 GB), AMD 7800X3D, 128 GB DDR5
**Model:** allenai/Molmo2-4B (bfloat16, device_map="auto")
**Baseline Config:** Standard DynamicCache, no compression
**Experimental Config:** CompressedDynamicCache with TQ4 nibble packing + incremental dequantization

### Hypothesis

Incremental dequantization — decompressing only the 1 new token per decode
step instead of the entire 11K+ cache — should dramatically reduce the 3.36x
overhead from Experiment 004 while maintaining the same 3.76x compression
and output quality.

### Setup

- Video: Seinfeld clip01.mp4 (~11K visual tokens at 2fps)
- Prompt: "Describe what happens in this video scene in detail..."
- Generation: 256 tokens, greedy decoding (do_sample=False)
- Change from Exp 004: `_compressed_update()` now maintains a running
  decompressed buffer per layer, catting only new tokens instead of
  re-dequantizing the full cache

### Results

| Metric | Baseline | TQ4 Incremental | Delta |
|--------|----------|-----------------|-------|
| Input tokens | 11,397 | 11,397 | Same |
| Output tokens | 256 | 256 | Same |
| Tokens/sec | 30.0 | 16.9 | 1.78x slower |
| VRAM peak | 18,058 MiB | 18,057 MiB | ~0 |
| Output quality | Detailed scene description | Near-identical | Minor phrasing |

**Comparison with Experiment 004 (full-cache dequant):**

| Metric | Exp 004 (full dequant) | Exp 005 (incremental) | Improvement |
|--------|----------------------|----------------------|-------------|
| Tokens/sec | 8.9 | **16.9** | 1.9x faster |
| Overhead | 3.36x | **1.78x** | 47% less |
| Compression | 3.76x | 3.76x | Same |
| Quality | Near-identical | Near-identical | Same |

### Observations

1. **Overhead nearly halved.** 3.36x → 1.78x. The bottleneck was
   re-dequantizing all 11K+ tokens (88 MB int64 allocation + 128x128
   rotation matmul) at every layer at every decode step. Incremental
   dequant reduces this to 1 token per layer per step.

2. **Output quality unchanged.** Both experiments produce near-identical
   descriptions of the Seinfeld scene. The first 100+ tokens match
   word-for-word between Exp 005 and baseline.

3. **Remaining 1.78x overhead** comes from:
   - Prefill: all tokens still dequantized once (unavoidable)
   - Per-step: compress 1 new token (quantize + nibble pack)
   - Per-step: dequantize 1 new token (centroid lookup + rotation)
   - Per-step: torch.cat to grow the decompressed buffer

4. **VRAM trade-off:** Decompressed buffers persist across layers
   (not freed between layers like Exp 003/004). Total VRAM is
   compressed storage + one full decompressed cache. The compressed
   storage provides savings for future Flash Attention kernel work.

### Next Steps

1. **Molmo2-8B validation** — with 1.78x overhead, 8B benchmarks are
   now practical.

2. **Flash Attention-style fused kernel** — would eliminate the
   decompressed buffer entirely, reading compressed data directly
   in the attention kernel.
