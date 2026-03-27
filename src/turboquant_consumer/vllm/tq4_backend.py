"""TQ4 compressed KV cache attention backend for vLLM.

Phase 3a skeleton: subclasses FlashAttentionBackend with no compression.
The backend registers via ``register_tq4_backend()`` which overrides
``AttentionBackendEnum.CUSTOM``.  Select it with ``--attention-backend CUSTOM``.

Implementation phases:
    3a (this): Passthrough to Flash Attention — validates plugin wiring.
    3b: Override ``do_kv_cache_update()`` with Triton ``reshape_and_cache_tq4``.
    3c: Override ``forward()`` to dequantize TQ4 pages before attention.
    3d: Production benchmark against vLLM baseline and HF-TQ4.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionImplBase, AttentionMetadataBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 3a constants — will be used in Phase 3b/3c
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

    Phase 3a: identical to FlashAttentionBackend.  Overrides only metadata
    (name, impl class) so the plugin wiring can be validated end-to-end.

    Phase 3b-3c will override:
        - ``get_kv_cache_shape()`` for the packed TQ4 page layout
        - ``supported_kv_cache_dtypes`` to include ``"tq4"``
        - ``get_impl_cls()`` to return the compressing/decompressing impl
    """

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """Declare support for multimodal prefix (bidirectional visual tokens).

        Required for VLMs like Molmo2 that use ``is_mm_prefix_lm = True``.
        Without this, vLLM rejects the backend during validation.

        Returns:
            ``True`` — delegates to Flash Attention which handles this correctly.
        """
        return True

    @staticmethod
    def get_name() -> str:
        """Return backend identifier string.

        Must match the ``AttentionBackendEnum`` member name because vLLM
        does ``AttentionBackendEnum[backend.get_name()]`` during model init.

        Returns:
            ``"CUSTOM"`` — matches ``AttentionBackendEnum.CUSTOM``.
        """
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase]:
        """Return the attention implementation class.

        Returns:
            :class:`TQ4AttentionImpl` (passthrough in Phase 3a).
        """
        return TQ4AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        """Return the metadata builder class.

        Returns:
            :class:`FlashAttentionMetadataBuilder` — reused from Flash Attention.
        """
        return FlashAttentionMetadataBuilder

    # Phase 3b will override:
    #
    # @staticmethod
    # def get_kv_cache_shape(
    #     num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto",
    # ) -> tuple[int, ...]:
    #     # One page stores K and V together in packed TQ4 format
    #     bytes_per_token = 2 * num_kv_heads * _tq4_bytes_per_token(head_size)
    #     return (num_blocks, block_size, bytes_per_token)


# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------


class TQ4AttentionImpl(FlashAttentionImpl):
    """TQ4 attention implementation.

    Phase 3a: pure passthrough to ``FlashAttentionImpl``.

    Phase 3b will override ``do_kv_cache_update()`` to compress K/V into
    TQ4 format before writing to paged cache blocks.

    Phase 3c will override ``forward()`` to dequantize TQ4 pages into
    temporary FP16 buffers before calling Flash Attention.
    """

    # Phase 3b will override:
    #
    # def do_kv_cache_update(
    #     self,
    #     key: torch.Tensor,
    #     value: torch.Tensor,
    #     kv_cache: torch.Tensor,
    #     attn_metadata: FlashAttentionMetadata,
    #     layer: torch.nn.Module,
    # ) -> None:
    #     reshape_and_cache_tq4(key, value, kv_cache, attn_metadata.slot_mapping,
    #                           self._rotation, self._codebook)

    # Phase 3c will override:
    #
    # def forward(
    #     self,
    #     layer: torch.nn.Module,
    #     query: torch.Tensor,
    #     key: torch.Tensor,
    #     value: torch.Tensor,
    #     kv_cache: torch.Tensor,
    #     attn_metadata: ...,
    #     output: torch.Tensor,
    #     ...
    # ) -> torch.Tensor:
    #     # 1. Compress and store new K/V
    #     # 2. Dequantize relevant cache pages to FP16
    #     # 3. Call Flash Attention on decompressed data


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
