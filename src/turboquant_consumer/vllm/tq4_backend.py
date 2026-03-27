"""TQ4 compressed KV cache attention backend for vLLM.

Phase 3b: Pure PyTorch compress/decompress in vLLM's attention path.
Compresses K/V into nibble-packed TQ4 format in the paged cache,
decompresses to FP16 for Flash Attention on each forward call.

The naive PyTorch implementation validates correctness. Performance
optimization (Triton kernels) is deferred to Phase 3c.

Implementation phases:
    3a (done): Passthrough skeleton — validated plugin wiring.
    3b (this): Pure PyTorch compress + decompress — validates correctness.
    3c: Triton kernel optimization (if profiling shows bottleneck).
    3d: Production benchmark against vLLM baseline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)

from turboquant_consumer.quantizer import TurboQuantMSE

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionImplBase, AttentionMetadataBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TQ4 constants
# ---------------------------------------------------------------------------

TQ4_BITS = 4
TQ4_SEED = 42

# Per-token per-head storage: head_dim/2 bytes (nibble-packed) + 4 bytes (fp32 norm)
# For head_dim=128: 64 + 4 = 68 bytes vs 256 bytes FP16 = 3.76x compression
TQ4_NORM_BYTES = 4  # fp32


def _tq4_bytes_per_token(head_dim: int) -> int:
    """Packed byte count for one token, one KV head, one of K or V.

    Returns:
        Byte count: ``head_dim // 2`` (nibble-packed indices) + 4 (fp32 norm).
    """
    return head_dim // 2 + TQ4_NORM_BYTES


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TQ4AttentionBackend(FlashAttentionBackend):
    """TQ4 compressed KV cache attention backend.

    Phase 3b uses the standard Flash Attention cache shape (5D) and
    validates TQ4 compression quality via compress→decompress round-trip
    in ``forward()``. The cache stores decompressed (lossy) FP16 data.

    Phase 3c will override ``get_kv_cache_shape()`` and
    ``KVCacheSpec.page_size_bytes`` for the packed TQ4 layout with
    real VRAM savings. This requires deeper integration with vLLM's
    buffer allocation (``page_size_bytes`` controls allocation size,
    not ``get_kv_cache_shape()``).
    """

    forward_includes_kv_cache_update = True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """Required for VLMs like Molmo2 with bidirectional visual tokens."""
        return True

    @staticmethod
    def get_name() -> str:
        """Must return ``"CUSTOM"`` to match ``AttentionBackendEnum.CUSTOM``."""
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase]:
        """Return :class:`TQ4AttentionImpl`."""
        return TQ4AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        """Return :class:`FlashAttentionMetadataBuilder` — reused."""
        return FlashAttentionMetadataBuilder

    # get_kv_cache_shape() and get_kv_cache_stride_order() inherited from
    # FlashAttentionBackend — standard (2, NB, BS, H, D) shape.
    # Phase 3c will override with TQ4 packed layout once buffer allocation
    # (KVCacheSpec.page_size_bytes) is also overridden.


# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------


class TQ4AttentionImpl(FlashAttentionImpl):
    """TQ4 attention: compress → store → decompress → Flash Attention.

    Overrides FlashAttentionImpl to:

    1. Initialize TQ4 compression primitives (rotation matrix, codebook).
    2. Compress incoming K/V tokens and write to the packed TQ4 cache.
    3. Decompress the entire cache to standard FP16 format.
    4. Delegate attention computation to Flash Attention via ``super()``.

    This is the "naive-correct" Phase 3b implementation using pure PyTorch.
    Performance optimization (Triton kernels) is deferred to Phase 3c.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize TQ4 attention with compression primitives."""
        super().__init__(*args, **kwargs)

        # Use attributes set by super().__init__()
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads

        # TQ4 compression primitives (deterministic from seed, shared across layers)
        quantizer = TurboQuantMSE(head_size, TQ4_BITS, seed=TQ4_SEED)
        self._tq4_rotation = quantizer.rotation  # (D, D) fp32
        self._tq4_centroids = quantizer.codebook.centroids  # (16,) fp32
        self._tq4_boundaries = quantizer.codebook.boundaries  # (15,) fp32
        self._tq4_on_device = False

        # Byte layout offsets (per token in uint8 view)
        half_D = head_size // 2
        self._half_D = half_D
        self._k_idx_end = num_kv_heads * half_D
        self._k_norm_end = self._k_idx_end + num_kv_heads * TQ4_NORM_BYTES
        self._v_idx_end = self._k_norm_end + num_kv_heads * half_D
        self._total_bytes = self._v_idx_end + num_kv_heads * TQ4_NORM_BYTES

        logger.info(
            "TQ4AttentionImpl: %d KV heads, head_size=%d, "
            "%d bytes/token (%.2fx compression vs FP16)",
            num_kv_heads,
            head_size,
            self._total_bytes,
            (2 * num_kv_heads * head_size * 2) / self._total_bytes,
        )

    # ----- device management -----

    def _ensure_device(self, device: torch.device) -> None:
        """Move compression primitives to GPU on first use."""
        if not self._tq4_on_device:
            self._tq4_rotation = self._tq4_rotation.to(device)
            self._tq4_centroids = self._tq4_centroids.to(device)
            self._tq4_boundaries = self._tq4_boundaries.to(device)
            self._tq4_on_device = True

    # ----- compression / decompression -----

    def _compress(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress ``(N, H, D)`` → nibble-packed indices + fp32 norms.

        Returns:
            packed: ``(N, H, D//2)`` uint8 — two 4-bit centroid indices per byte.
            norms: ``(N, H, 1)`` fp32 — vector norms.
        """
        N, H, D = x.shape
        flat = x.reshape(N * H, D).float()

        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ self._tq4_rotation.T

        indices = torch.bucketize(rotated, self._tq4_boundaries)
        indices = indices.clamp(0, (1 << TQ4_BITS) - 1)

        idx_u8 = indices.to(torch.uint8)
        packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]

        return packed.reshape(N, H, D // 2), norms.reshape(N, H, 1)

    def _decompress(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decompress nibble-packed indices + norms → ``(N, H, D)``.

        Args:
            packed: ``(N, H, D//2)`` uint8.
            norms: ``(N, H, 1)`` fp32.
            dtype: Output dtype (e.g., ``torch.bfloat16``).

        Returns:
            Reconstructed tensor ``(N, H, D)`` in ``dtype``.
        """
        N, H, half_D = packed.shape
        D = half_D * 2

        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(N * H, D)

        flat_norms = norms.reshape(N * H, 1)
        reconstructed = self._tq4_centroids[indices]
        unrotated = reconstructed @ self._tq4_rotation
        result = unrotated * flat_norms

        return result.reshape(N, H, D).to(dtype)

    # ----- compress → decompress round-trip -----

    def _compress_decompress(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compress then decompress: validates TQ4 quality through vLLM.

        Input and output have the same shape and dtype. The returned tensor
        has lossy reconstruction error from the TQ4 round-trip (rotation →
        quantize → centroid lookup → inverse rotation → rescale).
        """
        packed, norms = self._compress(x)
        return self._decompress(packed, norms, x.dtype)

    # ----- forward -----

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        """TQ4 attention: compress→decompress K/V, then Flash Attention.

        Phase 3b: Validates TQ4 compression quality through vLLM's serving
        pipeline. The standard 5D KV cache stores decompressed (lossy) data.
        No VRAM savings yet — that requires overriding KVCacheSpec.page_size_bytes
        (Phase 3c).

        1. Compress new K/V via TQ4 round-trip (lossy).
        2. Write lossy K/V to standard cache via ``do_kv_cache_update()``.
        3. Run Flash Attention on the cache.
        """
        assert output is not None

        # Profiling mode
        if attn_metadata is None:
            output.zero_()
            return output

        # TQ4 round-trip on new K/V tokens
        if kv_cache is not None and key is not None and value is not None:
            self._ensure_device(query.device)
            key = self._compress_decompress(key)
            value = self._compress_decompress(value)

            # Write lossy K/V to standard cache
            self.do_kv_cache_update(
                layer,
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
            )

        # Delegate attention to Flash Attention
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_tq4_backend() -> None:
    """Register TQ4 as the CUSTOM attention backend.

    Called automatically by the ``vllm.general_plugins`` entry point,
    or manually before starting vLLM::

        from turboquant_consumer.vllm import register_tq4_backend

        register_tq4_backend()
        # then start vLLM with --attention-backend CUSTOM
    """
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "turboquant_consumer.vllm.tq4_backend.TQ4AttentionBackend",
    )
    logger.info("TQ4 attention backend registered as CUSTOM")
