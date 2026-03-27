"""Tests for TQ4 vLLM attention backend (Phase 3a skeleton).

These tests validate the plugin wiring: registration, class hierarchy,
and interface compliance.  They require vLLM to be installed.
"""

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402

from turboquant_consumer.vllm.tq4_backend import (  # noqa: E402
    TQ4AttentionBackend,
    TQ4AttentionImpl,
    _tq4_bytes_per_token,
    register_tq4_backend,
)


class TestTQ4Registration:
    """Backend registration and discovery."""

    def test_register_overrides_custom_enum(self):
        register_tq4_backend()
        cls = AttentionBackendEnum.CUSTOM.get_class()
        assert cls is TQ4AttentionBackend

    def test_register_is_idempotent(self):
        register_tq4_backend()
        register_tq4_backend()
        cls = AttentionBackendEnum.CUSTOM.get_class()
        assert cls is TQ4AttentionBackend


class TestTQ4AttentionBackend:
    """Backend class interface compliance."""

    def test_name_matches_enum(self):
        assert TQ4AttentionBackend.get_name() == "CUSTOM"

    def test_impl_cls(self):
        assert TQ4AttentionBackend.get_impl_cls() is TQ4AttentionImpl

    def test_builder_cls(self):
        assert TQ4AttentionBackend.get_builder_cls() is FlashAttentionMetadataBuilder

    def test_subclasses_flash_attention(self):
        assert issubclass(TQ4AttentionBackend, FlashAttentionBackend)

    def test_kv_cache_shape_matches_flash(self):
        shape = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
        )
        expected = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
        )
        assert shape == expected

    def test_supported_dtypes(self):
        assert torch.float16 in TQ4AttentionBackend.supported_dtypes
        assert torch.bfloat16 in TQ4AttentionBackend.supported_dtypes

    def test_supports_mm_prefix(self):
        assert TQ4AttentionBackend.supports_mm_prefix() is True


class TestTQ4AttentionImpl:
    """Impl class hierarchy."""

    def test_subclasses_flash_impl(self):
        assert issubclass(TQ4AttentionImpl, FlashAttentionImpl)


class TestTQ4ByteCalculation:
    """TQ4 page layout math."""

    def test_bytes_per_token_head_128(self):
        assert _tq4_bytes_per_token(128) == 68

    def test_bytes_per_token_head_64(self):
        assert _tq4_bytes_per_token(64) == 36

    def test_bytes_per_token_head_256(self):
        assert _tq4_bytes_per_token(256) == 132
