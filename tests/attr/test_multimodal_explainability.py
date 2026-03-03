#!/usr/bin/env python3

# pyre-strict

"""
Unified tests for the multimodal explainability module.

Covers:
  - Low-level math utilities (avg_heads, apply_self_attention_rules, etc.)
  - AttentionHookManager
  - LayerID parsing
  - TransformerArchConfig registry and resolution
  - ActivationAccessor
  - MultiModalModelWrapper (layer access + relevance generation)
  - MultiModalAttribution (layer attribution methods)
  - ExplainabilityMode enum
  - Readout utility
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from captum._utils.transformer.accessor import ActivationAccessor
from captum._utils.transformer.arch_config import (
    ARCH_CONFIGS,
    get_arch_config,
    TransformerArchConfig,
)
from captum._utils.transformer.layer_id import LayerID
from captum.attr._core.layer.layer_activation import LayerActivation
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_integrated_gradients import (
    LayerIntegratedGradients,
)
from captum.attr._core.multimodal_explainability import (
    apply_mm_attention_rules,
    apply_self_attention_rules,
    AttentionHookManager,
    avg_heads,
    ExplainabilityMode,
    MultiModalAttribution,
    MultiModalModelWrapper,
    normalize_self_relevance,
)
from captum.testing.helpers import BaseTest
from parameterized import parameterized, parameterized_class
from torch import nn, Tensor


# ═══════════════════════════════════════════════════════════════════════
# Dummy models — HuggingFace-style (return attention weights in tuple)
# ═══════════════════════════════════════════════════════════════════════


class DummyAttentionLayer(nn.Module):
    """
    A transformer self-attention layer that always returns
    (output, attention_weights) as a tuple — mimicking HuggingFace
    ``output_attentions=True`` behavior.
    """

    def __init__(self, d_model: int = 16, nhead: int = 2) -> None:
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, S, D = x.shape
        # Compute Q, K, V
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_weights = (q @ k.transpose(-2, -1)) / scale
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (B, H, S, S)

        # Attention output
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        attn_output = self.out_proj(attn_output)

        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))

        # Return (output, attention_weights) like HuggingFace
        return x, attn_weights


class DummyCrossAttentionLayer(nn.Module):
    """
    A cross-attention layer that takes (query_hidden, context_hidden)
    and returns (output, cross_attention_weights).
    """

    def __init__(self, d_model: int = 16, nhead: int = 2) -> None:
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, query: Tensor, context: Tensor
    ) -> Tuple[Tensor, Tensor]:
        B, Sq, D = query.shape
        Sc = context.shape[1]

        q = self.q_proj(query).view(B, Sq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, Sc, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, Sc, self.nhead, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        attn_weights = (q @ k.transpose(-2, -1)) / scale
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (B, H, Sq, Sc)

        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, Sq, D)
        attn_output = self.out_proj(attn_output)

        output = self.norm(query + attn_output)
        return output, attn_weights


class DummySingleStreamModel(nn.Module):
    """Single-stream transformer for testing Template A."""

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [DummyAttentionLayer(d_model, nhead) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x, _attn = layer(x)
        # Classification head: use first token (CLS)
        return self.head(x[:, 0, :])  # (B, vocab_size)


class DummyCoAttentionModel(nn.Module):
    """Co-attention bi-modal model for testing Template B."""

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 16,
        nhead: int = 2,
        text_layers: int = 2,
        vision_layers: int = 2,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.image_proj = nn.Linear(3 * 4 * 4, d_model)

        # Self-attention stacks
        self.text_self_layers = nn.ModuleList(
            [DummyAttentionLayer(d_model, nhead) for _ in range(text_layers)]
        )
        self.vision_self_layers = nn.ModuleList(
            [DummyAttentionLayer(d_model, nhead) for _ in range(vision_layers)]
        )

        # Cross-attention (1 co-attention block)
        self.text_cross = DummyCrossAttentionLayer(d_model, nhead)
        self.image_cross = DummyCrossAttentionLayer(d_model, nhead)

        self.classifier = nn.Linear(d_model * 2, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
    ) -> Tensor:
        # Text stream
        text_h = self.text_embedding(input_ids)
        for layer in self.text_self_layers:
            text_h, _ = layer(text_h)

        # Image stream
        B = pixel_values.shape[0]
        patches = pixel_values.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.contiguous().view(B, -1, 3 * 4 * 4)
        image_h = self.image_proj(patches)
        for layer in self.vision_self_layers:
            image_h, _ = layer(image_h)

        # Cross-attention
        text_h, _ = self.text_cross(text_h, image_h)
        image_h, _ = self.image_cross(image_h, text_h)

        # CLS classification: combine both streams so gradients flow
        # through both cross-attention modules
        text_cls = text_h[:, 0, :]
        image_cls = image_h.mean(dim=1)
        combined = torch.cat([text_cls, image_cls], dim=-1)
        return self.classifier(combined)


class DummyEncoderDecoderModel(nn.Module):
    """Encoder-decoder model for testing Template C."""

    def __init__(
        self,
        d_model: int = 16,
        nhead: int = 2,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        num_queries: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.image_proj = nn.Linear(3 * 4 * 4, d_model)

        # Encoder (image self-attention)
        self.encoder_layers = nn.ModuleList(
            [DummyAttentionLayer(d_model, nhead) for _ in range(encoder_layers)]
        )

        # Decoder queries (learnable)
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))

        # Decoder (self-attention + cross-attention)
        self.decoder_self_layers = nn.ModuleList(
            [DummyAttentionLayer(d_model, nhead) for _ in range(decoder_layers)]
        )
        self.decoder_cross_layers = nn.ModuleList(
            [DummyCrossAttentionLayer(d_model, nhead) for _ in range(decoder_layers)]
        )

        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, pixel_values: Tensor) -> Tensor:
        B = pixel_values.shape[0]
        patches = pixel_values.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.contiguous().view(B, -1, 3 * 4 * 4)
        encoder_h = self.image_proj(patches)

        # Encoder self-attention
        for layer in self.encoder_layers:
            encoder_h, _ = layer(encoder_h)

        # Decoder
        decoder_h = self.query_embed.expand(B, -1, -1)
        for self_layer, cross_layer in zip(
            self.decoder_self_layers, self.decoder_cross_layers
        ):
            decoder_h, _ = self_layer(decoder_h)
            decoder_h, _ = cross_layer(decoder_h, encoder_h)

        # Classification per query
        return self.class_head(decoder_h)  # (B, num_queries, num_classes)


# ═══════════════════════════════════════════════════════════════════════
# Dummy models — standard nn.MultiheadAttention (for accessor / attr)
# ═══════════════════════════════════════════════════════════════════════


class DummyTransformerLayer(nn.Module):
    """A minimal transformer layer with named sub-modules for testing."""

    def __init__(self, d_model: int = 16, nhead: int = 2) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.layer_norm(x)
        x = x + self.mlp(x)
        return x


class DummyTextEncoder(nn.Module):
    """A minimal text encoder (like a small LLM) with named layers."""

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 16,
        num_layers: int = 3,
        nhead: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [DummyTransformerLayer(d_model, nhead) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self, input_ids: Tensor, extra_features: Optional[Tensor] = None
    ) -> Tensor:
        x = self.embedding(input_ids)
        if extra_features is not None:
            x = x + extra_features
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class DummyVisionEncoder(nn.Module):
    """A minimal vision encoder (like a small ViT) with named layers."""

    def __init__(
        self,
        d_model: int = 16,
        num_layers: int = 2,
        nhead: int = 2,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(3 * 4 * 4, d_model)  # 4x4 patches, 3 channels
        self.layers = nn.ModuleList(
            [DummyTransformerLayer(d_model, nhead) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values: (batch, channels, height, width)
        batch = pixel_values.shape[0]
        # Simple patch embedding: reshape into patches and project
        patches = pixel_values.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.contiguous().view(batch, -1, 3 * 4 * 4)
        x = self.patch_embed(patches)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DummyLVLM(nn.Module):
    """
    A minimal Large Vision-Language Model combining a vision encoder
    and a text encoder, mimicking the structure of LLaVA-like models.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 16,
        vision_layers: int = 2,
        text_layers: int = 3,
        nhead: int = 2,
    ) -> None:
        super().__init__()
        self.vision_encoder = DummyVisionEncoder(d_model, vision_layers, nhead)
        self.text_encoder = DummyTextEncoder(vocab_size, d_model, text_layers, nhead)
        self.projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
    ) -> Tensor:
        extra_features = None
        if pixel_values is not None:
            vis_out = self.vision_encoder(pixel_values)
            vis_projected = self.projection(vis_out)
            # Mean-pool vision features to inject into text encoder
            extra_features = vis_projected.mean(dim=1, keepdim=True)
        if input_ids is not None:
            if extra_features is not None:
                extra_features = extra_features.expand(
                    -1, input_ids.shape[1], -1
                )
            return self.text_encoder(input_ids, extra_features=extra_features)
        # Vision-only mode: return projected vision features
        assert extra_features is not None
        return extra_features


# ═══════════════════════════════════════════════════════════════════════
# Architecture configs
# ═══════════════════════════════════════════════════════════════════════

# -- Configs for HF-style dummy models (relevance generation) --

SINGLE_STREAM_CONFIG = TransformerArchConfig(
    text_encoder_prefix="layers",
    attn_module_name="self_attn",  # not used, will be overridden
    num_attention_heads=2,
)

CO_ATTN_TEXT_SELF_CONFIG = TransformerArchConfig(
    text_encoder_prefix="text_self_layers",
    vision_encoder_prefix="vision_self_layers",
    attn_module_name="self_attn",  # placeholder
    num_attention_heads=2,
)

SINGLE_STREAM_TEST_CONFIG = TransformerArchConfig(
    text_encoder_prefix="layers",
    attn_module_name="self_attn",  # unused in direct hook approach
    mlp_module_name="mlp",
    output_module_name="norm2",
    num_attention_heads=2,
)

# -- Configs for nn.MHA-style dummy models (accessor / attribution) --

DUMMY_TEXT_CONFIG = TransformerArchConfig(
    text_encoder_prefix="layers",
    attn_module_name="self_attn",
    mlp_module_name="mlp",
    output_module_name="layer_norm",
    num_attention_heads=2,
)

DUMMY_VISION_CONFIG = TransformerArchConfig(
    vision_encoder_prefix="layers",
    default_encoder_prefix="layers",
    attn_module_name="self_attn",
    mlp_module_name="mlp",
    output_module_name="layer_norm",
    num_attention_heads=2,
)

DUMMY_LVLM_CONFIG = TransformerArchConfig(
    vision_encoder_prefix="vision_encoder.layers",
    text_encoder_prefix="text_encoder.layers",
    attn_module_name="self_attn",
    mlp_module_name="mlp",
    output_module_name="layer_norm",
    num_attention_heads=2,
)


# ═══════════════════════════════════════════════════════════════════════
# Tests — Low-level math utilities
# ═══════════════════════════════════════════════════════════════════════


class TestAvgHeads(BaseTest):
    """Test gradient-weighted head averaging (Eq. 5)."""

    def test_basic_shape_3d(self) -> None:
        """3D input: (heads, q, k)."""
        attn = torch.rand(4, 6, 6)
        grad = torch.randn(4, 6, 6)
        result = avg_heads(attn, grad)
        self.assertEqual(result.shape, (6, 6))

    def test_basic_shape_4d(self) -> None:
        """4D input: (batch, heads, q, k)."""
        attn = torch.rand(2, 4, 6, 6)
        grad = torch.randn(2, 4, 6, 6)
        result = avg_heads(attn, grad)
        self.assertEqual(result.shape, (2, 6, 6))

    def test_non_negative(self) -> None:
        """Result should be non-negative (clamp at 0)."""
        attn = torch.rand(4, 6, 6)
        grad = torch.randn(4, 6, 6)
        result = avg_heads(attn, grad)
        self.assertTrue((result >= 0).all())

    def test_zero_gradient(self) -> None:
        """Zero gradient should yield zero result."""
        attn = torch.rand(4, 6, 6)
        grad = torch.zeros(4, 6, 6)
        result = avg_heads(attn, grad)
        self.assertTrue(torch.allclose(result, torch.zeros(6, 6)))

    def test_positive_gradient_preserves_attention(self) -> None:
        """If gradient is all positive, result = mean(grad * attn)."""
        attn = torch.rand(2, 5, 5)
        grad = torch.abs(torch.randn(2, 5, 5))  # all positive
        result = avg_heads(attn, grad)
        expected = (grad * attn).mean(dim=0)
        self.assertTrue(torch.allclose(result, expected))


class TestApplySelfAttentionRules(BaseTest):
    """Test self-attention relevance update (Eq. 6/7)."""

    def test_identity_update(self) -> None:
        """With zero A_bar, R_ss should remain unchanged."""
        R_ss = torch.eye(4)
        A_bar = torch.zeros(4, 4)
        R_ss_new, _ = apply_self_attention_rules(R_ss, None, A_bar)
        self.assertTrue(torch.allclose(R_ss_new, R_ss))

    def test_eq6_basic(self) -> None:
        """R_ss updates correctly: R_ss + A_bar @ R_ss."""
        R_ss = torch.eye(4)
        A_bar = 0.1 * torch.ones(4, 4)
        R_ss_new, _ = apply_self_attention_rules(R_ss, None, A_bar)
        expected = R_ss + A_bar @ R_ss
        self.assertTrue(torch.allclose(R_ss_new, expected))

    def test_eq7_propagation(self) -> None:
        """R_sq should also be updated when provided."""
        R_ss = torch.eye(4)
        R_sq = torch.randn(4, 3)
        A_bar = 0.1 * torch.ones(4, 4)
        R_ss_new, R_sq_new = apply_self_attention_rules(R_ss, R_sq, A_bar)
        expected_sq = R_sq + A_bar @ R_sq
        self.assertIsNotNone(R_sq_new)
        self.assertTrue(torch.allclose(R_sq_new, expected_sq))

    def test_none_rsq(self) -> None:
        """R_sq=None should return None."""
        R_ss = torch.eye(4)
        A_bar = torch.rand(4, 4)
        _, R_sq_new = apply_self_attention_rules(R_ss, None, A_bar)
        self.assertIsNone(R_sq_new)


class TestNormalizeSelfRelevance(BaseTest):
    """Test residual normalization (Eq. 8-9)."""

    def test_identity_input(self) -> None:
        """Normalizing identity should return identity (R_hat=0, row_norm=0)."""
        R = torch.eye(4)
        R_bar = normalize_self_relevance(R)
        # R_hat = R - I = 0; row_norm(0) = 0; result = 0 + I = I
        self.assertTrue(torch.allclose(R_bar, R, atol=1e-6))

    def test_shape_preserved(self) -> None:
        R = torch.rand(5, 5) + torch.eye(5)
        R_bar = normalize_self_relevance(R)
        self.assertEqual(R_bar.shape, (5, 5))

    def test_batch_shape(self) -> None:
        R = torch.rand(2, 5, 5) + torch.eye(5).unsqueeze(0)
        R_bar = normalize_self_relevance(R)
        self.assertEqual(R_bar.shape, (2, 5, 5))

    def test_finite(self) -> None:
        R = torch.rand(5, 5) + torch.eye(5)
        R_bar = normalize_self_relevance(R)
        self.assertTrue(torch.isfinite(R_bar).all())


class TestApplyMMAttentionRules(BaseTest):
    """Test cross-attention relevance update (Eq. 10/11)."""

    def test_eq10_basic(self) -> None:
        """Cross-attention update should increase R_sq magnitude."""
        t, i_count = 4, 3
        R_ss = torch.eye(t)
        R_qq = torch.eye(i_count)
        R_sq = torch.zeros(t, i_count)
        R_qs = torch.zeros(i_count, t)
        A_sq_bar = 0.1 * torch.ones(t, i_count)

        R_ss_new, R_sq_new = apply_mm_attention_rules(
            R_ss, R_qq, R_sq, R_qs, A_sq_bar, use_eq11=False
        )
        # R_sq should no longer be zero
        self.assertTrue(R_sq_new.abs().sum() > 0)

    def test_eq11_updates_rss(self) -> None:
        """Eq. 11 should update R_ss when R_qs is provided."""
        t, i_count = 4, 3
        R_ss = torch.eye(t)
        R_qq = torch.eye(i_count)
        R_sq = torch.zeros(t, i_count)
        R_qs = 0.1 * torch.ones(i_count, t)
        A_sq_bar = 0.1 * torch.ones(t, i_count)

        R_ss_new, _ = apply_mm_attention_rules(
            R_ss, R_qq, R_sq, R_qs, A_sq_bar, use_eq11=True
        )
        # R_ss should differ from identity (was updated by Eq. 11)
        diff = (R_ss_new - torch.eye(t)).abs().sum()
        self.assertTrue(diff > 0)

    def test_no_eq11_when_disabled(self) -> None:
        """With use_eq11=False, R_ss should not be updated by cross-attn."""
        t, i_count = 4, 3
        R_ss_orig = torch.eye(t)
        R_qq = torch.eye(i_count)
        R_sq = torch.zeros(t, i_count)
        R_qs = 0.5 * torch.ones(i_count, t)
        A_sq_bar = 0.1 * torch.ones(t, i_count)

        R_ss_new, _ = apply_mm_attention_rules(
            R_ss_orig.clone(), R_qq, R_sq, R_qs, A_sq_bar, use_eq11=False
        )
        self.assertTrue(torch.allclose(R_ss_new, R_ss_orig))

    def test_no_eq11_when_rqs_none(self) -> None:
        """With R_qs=None, Eq. 11 is skipped even if use_eq11=True."""
        t, i_count = 4, 3
        R_ss_orig = torch.eye(t)
        R_qq = torch.eye(i_count)
        R_sq = torch.zeros(t, i_count)
        A_sq_bar = 0.1 * torch.ones(t, i_count)

        R_ss_new, _ = apply_mm_attention_rules(
            R_ss_orig.clone(), R_qq, R_sq, None, A_sq_bar, use_eq11=True
        )
        self.assertTrue(torch.allclose(R_ss_new, R_ss_orig))

    def test_shapes(self) -> None:
        """Output shapes should match input shapes."""
        t, i_count = 5, 7
        R_ss = torch.eye(t)
        R_qq = torch.eye(i_count)
        R_sq = torch.zeros(t, i_count)
        A_sq_bar = torch.rand(t, i_count)

        R_ss_new, R_sq_new = apply_mm_attention_rules(
            R_ss, R_qq, R_sq, None, A_sq_bar, use_eq11=False
        )
        self.assertEqual(R_ss_new.shape, (t, t))
        self.assertEqual(R_sq_new.shape, (t, i_count))


# ═══════════════════════════════════════════════════════════════════════
# Tests — AttentionHookManager
# ═══════════════════════════════════════════════════════════════════════


class TestAttentionHookManager(BaseTest):
    """Test hook-based attention capture."""

    def test_capture_attention_weights(self) -> None:
        """Hook should capture attention weights from forward pass."""
        layer = DummyAttentionLayer(d_model=16, nhead=2)
        hook_mgr = AttentionHookManager(layer, [layer])

        x = torch.randn(1, 5, 16)
        out, _ = layer(x)

        attn = hook_mgr.get_attention(0)
        self.assertEqual(attn.shape, (1, 2, 5, 5))
        hook_mgr.remove_hooks()

    def test_capture_attention_gradients(self) -> None:
        """Hook should capture gradients after backward."""
        layer = DummyAttentionLayer(d_model=16, nhead=2)
        hook_mgr = AttentionHookManager(layer, [layer])

        x = torch.randn(1, 5, 16, requires_grad=True)
        out, _ = layer(x)
        out.sum().backward()

        grad = hook_mgr.get_gradient(0)
        self.assertEqual(grad.shape, (1, 2, 5, 5))
        hook_mgr.remove_hooks()

    def test_multiple_modules(self) -> None:
        """Multiple modules should be captured independently."""
        layer1 = DummyAttentionLayer(d_model=16, nhead=2)
        layer2 = DummyAttentionLayer(d_model=16, nhead=2)

        # Create a simple model
        class TwoLayerModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = layer1
                self.l2 = layer2

            def forward(self, x: Tensor) -> Tensor:
                x, _ = self.l1(x)
                x, _ = self.l2(x)
                return x

        model = TwoLayerModel()
        hook_mgr = AttentionHookManager(model, [layer1, layer2])

        x = torch.randn(1, 5, 16, requires_grad=True)
        out = model(x)
        out.sum().backward()

        attn0 = hook_mgr.get_attention(0)
        attn1 = hook_mgr.get_attention(1)
        self.assertEqual(attn0.shape, (1, 2, 5, 5))
        self.assertEqual(attn1.shape, (1, 2, 5, 5))
        # Two different layers should produce different attention maps
        self.assertFalse(torch.allclose(attn0, attn1))

        hook_mgr.remove_hooks()

    def test_remove_hooks(self) -> None:
        """After remove_hooks, accessing data should raise."""
        layer = DummyAttentionLayer(d_model=16, nhead=2)
        hook_mgr = AttentionHookManager(layer, [layer])

        x = torch.randn(1, 5, 16)
        layer(x)
        hook_mgr.remove_hooks()

        with self.assertRaises(RuntimeError):
            hook_mgr.get_attention(0)

    def test_clear(self) -> None:
        """clear() should remove cached data but keep hooks."""
        layer = DummyAttentionLayer(d_model=16, nhead=2)
        hook_mgr = AttentionHookManager(layer, [layer])

        x = torch.randn(1, 5, 16)
        layer(x)
        self.assertIsNotNone(hook_mgr.get_attention(0))

        hook_mgr.clear()
        with self.assertRaises(RuntimeError):
            hook_mgr.get_attention(0)

        # But hooks are still active — run forward again
        layer(x)
        attn = hook_mgr.get_attention(0)
        self.assertIsNotNone(attn)

        hook_mgr.remove_hooks()


# ═══════════════════════════════════════════════════════════════════════
# Tests — LayerID
# ═══════════════════════════════════════════════════════════════════════


class TestLayerID(BaseTest):
    """Test the LayerID parser for compact layer identifier notation."""

    @parameterized.expand(
        [
            ("L0", None, 0, None, None),
            ("L5", None, 5, None, None),
            ("L12", None, 12, None, None),
            ("V.L1", "V", 1, None, None),
            ("T.L3", "T", 3, None, None),
            ("L1.H5", None, 1, 5, None),
            ("V.L1.H5", "V", 1, 5, None),
            ("T.L2.H0", "T", 2, 0, None),
            ("L0.attn", None, 0, None, "attn"),
            ("V.L1.attn", "V", 1, None, "attn"),
            ("T.L3.mlp", "T", 3, None, "mlp"),
            ("V.L0.output", "V", 0, None, "output"),
            ("L0.mlp.0", None, 0, None, "mlp.0"),
            ("L0.mlp.1", None, 0, None, "mlp.1"),
            ("V.L1.mlp.0", "V", 1, None, "mlp.0"),
            ("T.L2.attn.q_proj", "T", 2, None, "attn.q_proj"),
        ]
    )
    def test_parse_valid(
        self,
        layer_str: str,
        expected_encoder: Optional[str],
        expected_layer: int,
        expected_head: Optional[int],
        expected_component: Optional[str],
    ) -> None:
        lid = LayerID.parse(layer_str)
        self.assertEqual(lid.encoder_type, expected_encoder)
        self.assertEqual(lid.layer_num, expected_layer)
        self.assertEqual(lid.head_num, expected_head)
        self.assertEqual(lid.component, expected_component)

    @parameterized.expand(
        [
            ("",),
            ("invalid",),
            ("X.L1",),  # X is not V or T
            ("V.L",),  # missing layer number
            ("V.L1.H5.extra",),  # H5 is head notation, extra part is invalid
        ]
    )
    def test_parse_invalid(self, layer_str: str) -> None:
        with self.assertRaises(ValueError):
            LayerID.parse(layer_str)

    def test_str_roundtrip(self) -> None:
        """Test that str(LayerID.parse(s)) produces a valid re-parseable string."""
        test_cases = ["L0", "L5", "V.L1", "T.L3", "L1.H5", "V.L1.H5", "L0.attn",
                      "L0.mlp.0", "V.L1.mlp.1"]
        for s in test_cases:
            lid = LayerID.parse(s)
            result_str = str(lid)
            self.assertEqual(result_str, s, f"str() mismatch for '{s}'")
            reparsed = LayerID.parse(result_str)
            self.assertEqual(lid, reparsed, f"Roundtrip failed for '{s}'")

    def test_frozen_dataclass(self) -> None:
        lid = LayerID.parse("V.L1.H5")
        with self.assertRaises(AttributeError):
            lid.layer_num = 2  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            lid.encoder_type = "T"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            lid.head_num = 0  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            lid.component = "mlp"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# Tests — TransformerArchConfig
# ═══════════════════════════════════════════════════════════════════════


class TestTransformerArchConfig(BaseTest):
    """Test architecture configuration and the built-in registry."""

    def test_get_known_configs(self) -> None:
        for name in ["gpt2", "llama", "vit", "clip_vision", "llava", "qwen2_vl"]:
            config = get_arch_config(name)
            self.assertIsInstance(config, TransformerArchConfig)

    def test_get_unknown_config_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_arch_config("nonexistent_model")

    def test_get_config_returns_copy(self) -> None:
        c1 = get_arch_config("llama")
        c2 = get_arch_config("llama")
        self.assertEqual(c1.text_encoder_prefix, c2.text_encoder_prefix)
        c1.text_encoder_prefix = "modified"
        self.assertNotEqual(c1.text_encoder_prefix, c2.text_encoder_prefix)

    def test_component_name_resolution(self) -> None:
        config = TransformerArchConfig(
            text_encoder_prefix="layers",
            attn_module_name="attention",
            mlp_module_name="feed_forward",
        )
        self.assertEqual(config.get_component_name("attn"), "attention")
        self.assertEqual(config.get_component_name("mlp"), "feed_forward")
        self.assertEqual(config.get_component_name(None), None)
        # Unknown component returned as-is
        self.assertEqual(config.get_component_name("custom_module"), "custom_module")

    def test_encoder_prefix_resolution(self) -> None:
        config = TransformerArchConfig(
            vision_encoder_prefix="v_encoder.layers",
            text_encoder_prefix="t_encoder.layers",
        )
        self.assertEqual(config.get_encoder_prefix("V"), "v_encoder.layers")
        self.assertEqual(config.get_encoder_prefix("T"), "t_encoder.layers")
        # Default should be text encoder
        self.assertEqual(config.get_encoder_prefix(None), "t_encoder.layers")

    def test_encoder_prefix_missing_raises(self) -> None:
        config = TransformerArchConfig(text_encoder_prefix="layers")
        with self.assertRaises(ValueError):
            config.get_encoder_prefix("V")  # No vision encoder defined

    def test_custom_component_map(self) -> None:
        config = TransformerArchConfig(
            text_encoder_prefix="layers",
            component_map={"qkv": "self_attn.qkv_proj"},
        )
        self.assertEqual(config.get_component_name("qkv"), "self_attn.qkv_proj")

    def test_dotted_component_name_resolution(self) -> None:
        """Test that dotted sub-paths resolve the first segment correctly."""
        config = TransformerArchConfig(
            text_encoder_prefix="layers",
            attn_module_name="self_attn",
            mlp_module_name="feed_forward",
        )
        # "mlp.0" -> resolve "mlp" to "feed_forward", append ".0"
        self.assertEqual(config.get_component_name("mlp.0"), "feed_forward.0")
        self.assertEqual(config.get_component_name("mlp.1"), "feed_forward.1")
        # "attn.q_proj" -> resolve "attn" to "self_attn", append ".q_proj"
        self.assertEqual(
            config.get_component_name("attn.q_proj"), "self_attn.q_proj"
        )
        # Unknown prefix passed through as-is
        self.assertEqual(config.get_component_name("custom.sub"), "custom.sub")

    def test_add_custom_to_registry(self) -> None:
        """Test adding a custom architecture to the global registry."""
        ARCH_CONFIGS["test_custom"] = TransformerArchConfig(
            text_encoder_prefix="custom.layers"
        )
        config = get_arch_config("test_custom")
        self.assertEqual(config.text_encoder_prefix, "custom.layers")
        # Cleanup
        del ARCH_CONFIGS["test_custom"]


# ═══════════════════════════════════════════════════════════════════════
# Tests — ActivationAccessor
# ═══════════════════════════════════════════════════════════════════════


@parameterized_class(
    ("device",),
    [("cpu",), ("cuda",)] if torch.cuda.is_available() else [("cpu",)],
)
class TestActivationAccessor(BaseTest):
    device: str = "cpu"

    def test_resolve_text_layer(self) -> None:
        model = DummyTextEncoder().to(self.device)
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG)
        module = accessor.resolve_module("L0")
        self.assertIsInstance(module, DummyTransformerLayer)
        self.assertIs(module, model.layers[0])

    def test_resolve_text_layer_attn(self) -> None:
        model = DummyTextEncoder().to(self.device)
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG)
        module = accessor.resolve_module("L1.attn")
        self.assertIsInstance(module, nn.MultiheadAttention)
        self.assertIs(module, model.layers[1].self_attn)

    def test_resolve_text_layer_mlp(self) -> None:
        model = DummyTextEncoder().to(self.device)
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG)
        module = accessor.resolve_module("L2.mlp")
        self.assertIs(module, model.layers[2].mlp)

    def test_resolve_vision_layer(self) -> None:
        model = DummyVisionEncoder().to(self.device)
        accessor = ActivationAccessor(model, DUMMY_VISION_CONFIG)
        module = accessor.resolve_module("L0")
        self.assertIs(module, model.layers[0])

    def test_resolve_lvlm_vision_and_text(self) -> None:
        model = DummyLVLM().to(self.device)
        accessor = ActivationAccessor(model, DUMMY_LVLM_CONFIG)

        v_layer = accessor.resolve_module("V.L0")
        self.assertIs(v_layer, model.vision_encoder.layers[0])

        t_layer = accessor.resolve_module("T.L2")
        self.assertIs(t_layer, model.text_encoder.layers[2])

    def test_resolve_lvlm_attn_component(self) -> None:
        model = DummyLVLM().to(self.device)
        accessor = ActivationAccessor(model, DUMMY_LVLM_CONFIG)

        attn = accessor.resolve_module("V.L1.attn")
        self.assertIs(attn, model.vision_encoder.layers[1].self_attn)

    def test_get_activation_text(self) -> None:
        model = DummyTextEncoder().to(self.device)
        model.eval()
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        act = accessor.get_activation("L0", input_ids)
        self.assertIsInstance(act, Tensor)
        self.assertEqual(act.device.type, self.device)
        # Activation should have shape (batch, seq_len, d_model)
        self.assertEqual(act.shape[0], 1)
        self.assertEqual(act.shape[1], 5)

    def test_get_activation_input(self) -> None:
        model = DummyTextEncoder().to(self.device)
        model.eval()
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        act = accessor.get_activation(
            "L0", input_ids, attribute_to_layer_input=True
        )
        self.assertIsInstance(act, Tensor)

    def test_get_activation_vision(self) -> None:
        model = DummyVisionEncoder().to(self.device)
        model.eval()
        accessor = ActivationAccessor(model, DUMMY_VISION_CONFIG, device=self.device)
        pixel_values = torch.randn(1, 3, 8, 8, device=self.device)
        act = accessor.get_activation("L0", pixel_values)
        self.assertIsInstance(act, Tensor)
        self.assertEqual(act.device.type, self.device)

    def test_get_multi_layer_activations(self) -> None:
        model = DummyTextEncoder().to(self.device)
        model.eval()
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        acts = accessor.get_multi_layer_activations(
            ["L0", "L1", "L2"], input_ids
        )
        self.assertEqual(len(acts), 3)
        for key in ["L0", "L1", "L2"]:
            self.assertIn(key, acts)
            self.assertIsInstance(acts[key], Tensor)

    def test_resolve_ffn_sub_modules(self) -> None:
        """Test resolving FFN sub-modules using dotted component paths."""
        model = DummyTextEncoder().to(self.device)
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG)
        # mlp is nn.Sequential(Linear, ReLU, Linear)
        module_0 = accessor.resolve_module("L0.mlp.0")
        self.assertIsInstance(module_0, nn.Linear)
        self.assertIs(module_0, model.layers[0].mlp[0])

        module_1 = accessor.resolve_module("L0.mlp.1")
        self.assertIsInstance(module_1, nn.ReLU)
        self.assertIs(module_1, model.layers[0].mlp[1])

        module_2 = accessor.resolve_module("L0.mlp.2")
        self.assertIsInstance(module_2, nn.Linear)
        self.assertIs(module_2, model.layers[0].mlp[2])

    def test_get_activation_ffn_sub_modules(self) -> None:
        """Test extracting activations from FFN sub-modules."""
        model = DummyTextEncoder().to(self.device)
        model.eval()
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)

        # Get activation of the first Linear in the MLP (input to ReLU)
        act_linear1 = accessor.get_activation("L0.mlp.0", input_ids)
        self.assertIsInstance(act_linear1, Tensor)

        # Get activation of the ReLU (output of activation function)
        act_relu = accessor.get_activation("L0.mlp.1", input_ids)
        self.assertIsInstance(act_relu, Tensor)

        # ReLU output should have no negative values
        self.assertTrue((act_relu >= 0).all())

        # Get activation of the second Linear
        act_linear2 = accessor.get_activation("L0.mlp.2", input_ids)
        self.assertIsInstance(act_linear2, Tensor)

    def test_get_multi_layer_activations_with_sub_modules(self) -> None:
        """Test extracting multiple activations including sub-modules."""
        model = DummyTextEncoder().to(self.device)
        model.eval()
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        acts = accessor.get_multi_layer_activations(
            ["L0", "L0.mlp", "L0.mlp.0", "L0.mlp.1"], input_ids
        )
        self.assertEqual(len(acts), 4)
        for key in ["L0", "L0.mlp", "L0.mlp.0", "L0.mlp.1"]:
            self.assertIn(key, acts)
            self.assertIsInstance(acts[key], Tensor)

    def test_get_attention_weights_all_heads(self) -> None:
        """Test extracting attention weights from a layer."""
        model = DummyTextEncoder().to(self.device)
        model.eval()
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)

        weights = accessor.get_attention_weights("L0", input_ids)
        self.assertIsInstance(weights, Tensor)
        # nn.MultiheadAttention with default average_attn_weights=True
        # returns (batch, seq_len, seq_len)
        self.assertEqual(weights.dim(), 3)
        self.assertEqual(weights.shape[0], 1)
        self.assertEqual(weights.shape[1], 5)
        self.assertEqual(weights.shape[2], 5)

    def test_get_attention_weights_single_head(self) -> None:
        """Test extracting attention weights for a specific head."""
        model = DummyTextEncoder().to(self.device)
        model.eval()
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)

        # With averaged weights (3D), head=0 selects index 0 along dim 1
        weights = accessor.get_attention_weights("L0", input_ids, head=0)
        self.assertIsInstance(weights, Tensor)

    def test_resolve_invalid_layer_raises(self) -> None:
        model = DummyTextEncoder(num_layers=2).to(self.device)
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG)
        with self.assertRaises((IndexError, AttributeError)):
            accessor.resolve_module("L5")  # Only 2 layers

    def test_visualize_attention_heads(self) -> None:
        """Test visualize_attention_heads with a 4D tensor."""
        import matplotlib
        matplotlib.use("Agg")
        from captum._utils.transformer.visualization import (
            visualize_attention_heads,
        )

        # Simulate (batch, num_heads, seq_len, seq_len)
        weights = torch.rand(1, 4, 6, 6)
        tokens = [f"t{i}" for i in range(6)]

        # All heads
        fig = visualize_attention_heads(
            weights, tokens=tokens, use_pyplot=False,
        )
        self.assertEqual(len(fig.axes), 4 + 4)  # 4 plots + 4 colorbars

        # Single head
        fig2 = visualize_attention_heads(
            weights, head=2, tokens=tokens, use_pyplot=False,
        )
        self.assertIsNotNone(fig2)

        # Multiple heads
        fig3 = visualize_attention_heads(
            weights, head=[0, 3], tokens=tokens, use_pyplot=False,
        )
        self.assertIsNotNone(fig3)

        # 2D input (single head, no batch)
        fig4 = visualize_attention_heads(
            weights[0, 0], tokens=tokens, use_pyplot=False,
        )
        self.assertIsNotNone(fig4)


# ═══════════════════════════════════════════════════════════════════════
# Tests — MultiModalModelWrapper (layer access)
# ═══════════════════════════════════════════════════════════════════════


@parameterized_class(
    ("device",),
    [("cpu",), ("cuda",)] if torch.cuda.is_available() else [("cpu",)],
)
class TestMultiModalModelWrapperAccess(BaseTest):
    """Test layer resolution and activation access via the wrapper."""
    device: str = "cpu"

    def test_wrapper_text_model(self) -> None:
        model = DummyTextEncoder().to(self.device)
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)
        layer = wrapper.resolve_layer("L0")
        self.assertIs(layer, model.layers[0])

    def test_wrapper_lvlm(self) -> None:
        model = DummyLVLM().to(self.device)
        wrapper = MultiModalModelWrapper(model, DUMMY_LVLM_CONFIG, device=self.device)

        v_layer = wrapper.resolve_layer("V.L0")
        self.assertIs(v_layer, model.vision_encoder.layers[0])

        t_layer = wrapper.resolve_layer("T.L1")
        self.assertIs(t_layer, model.text_encoder.layers[1])

    def test_wrapper_resolve_layers(self) -> None:
        model = DummyLVLM().to(self.device)
        wrapper = MultiModalModelWrapper(model, DUMMY_LVLM_CONFIG, device=self.device)
        layers = wrapper.resolve_layers(["V.L0", "T.L0", "T.L1"])
        self.assertEqual(len(layers), 3)
        self.assertIs(layers[0], model.vision_encoder.layers[0])
        self.assertIs(layers[1], model.text_encoder.layers[0])
        self.assertIs(layers[2], model.text_encoder.layers[1])

    def test_wrapper_get_activation(self) -> None:
        model = DummyTextEncoder().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        act = wrapper.get_activation("L0", input_ids)
        self.assertIsInstance(act, Tensor)

    def test_wrapper_get_multi_layer_activations(self) -> None:
        model = DummyTextEncoder().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        acts = wrapper.get_multi_layer_activations(["L0", "L1"], input_ids)
        self.assertEqual(len(acts), 2)

    def test_wrapper_with_custom_config(self) -> None:
        custom_config = TransformerArchConfig(
            text_encoder_prefix="layers",
            attn_module_name="self_attn",
        )
        model = DummyTextEncoder().to(self.device)
        wrapper = MultiModalModelWrapper(model, custom_config)
        layer = wrapper.resolve_layer("L0")
        self.assertIs(layer, model.layers[0])


# ═══════════════════════════════════════════════════════════════════════
# Tests — MultiModalModelWrapper (single-stream relevance)
# ═══════════════════════════════════════════════════════════════════════


class TestMultiModalModelWrapperSingleStream(BaseTest):
    """Test Template A (single-stream) explainability."""

    def _make_model_and_explainer(
        self,
    ) -> Tuple[DummySingleStreamModel, MultiModalModelWrapper]:
        model = DummySingleStreamModel(
            vocab_size=32, d_model=16, nhead=2, num_layers=3
        )
        model.eval()

        config = TransformerArchConfig(
            text_encoder_prefix="layers",
            attn_module_name="dummy",  # our layers ARE the attn modules
            num_attention_heads=2,
        )
        explainer = MultiModalModelWrapper(
            model, config, mode="single_stream"
        )
        return model, explainer

    def test_generate_single_stream_shape(self) -> None:
        """Relevance matrix should be (n, n) where n = seq_len."""
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 6))

        # We need to hook the attn modules directly — the DummyAttentionLayer
        # itself returns (output, attn_weights). So we override _resolve_attn_modules.
        # Patching: make the explainer use the correct modules.
        original_resolve = explainer._resolve_attn_modules

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return list(model.layers)

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        R = explainer.generate_single_stream(
            input_ids, target_index=0, num_layers=3
        )
        self.assertEqual(R.shape, (6, 6))

    def test_relevance_is_finite(self) -> None:
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 6))

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return list(model.layers)

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        R = explainer.generate_single_stream(
            input_ids, target_index=0, num_layers=3
        )
        self.assertTrue(torch.isfinite(R).all())

    def test_relevance_non_trivial(self) -> None:
        """After propagation, R should differ from identity."""
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 6))

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return list(model.layers)

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        R = explainer.generate_single_stream(
            input_ids, target_index=0, num_layers=3
        )
        eye = torch.eye(6)
        diff = (R - eye).abs().sum()
        self.assertTrue(diff > 0, "Relevance should differ from identity.")

    def test_cls_readout(self) -> None:
        """readout(R, 0) should give a 1D vector of length n."""
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 6))

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return list(model.layers)

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        R = explainer.generate_single_stream(
            input_ids, target_index=0, num_layers=3
        )
        relevance = MultiModalModelWrapper.readout(R, readout_index=0)
        self.assertEqual(relevance.shape, (6,))

    def test_target_fn(self) -> None:
        """Custom target_fn should be usable instead of target_index."""
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 6))

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return list(model.layers)

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        def my_target(outputs: Tensor) -> Tensor:
            return outputs[0, 5]  # arbitrary target

        R = explainer.generate_single_stream(
            input_ids, target_fn=my_target, num_layers=3
        )
        self.assertEqual(R.shape, (6, 6))
        self.assertTrue(torch.isfinite(R).all())


# ═══════════════════════════════════════════════════════════════════════
# Tests — MultiModalModelWrapper (co-attention relevance)
# ═══════════════════════════════════════════════════════════════════════


class TestMultiModalModelWrapperCoAttention(BaseTest):
    """Test Template B (co-attention) explainability."""

    def _make_model_and_explainer(
        self,
    ) -> Tuple[DummyCoAttentionModel, MultiModalModelWrapper]:
        model = DummyCoAttentionModel(
            vocab_size=32, d_model=16, nhead=2,
            text_layers=2, vision_layers=2,
        )
        model.eval()

        config = TransformerArchConfig(
            text_encoder_prefix="text_self_layers",
            vision_encoder_prefix="vision_self_layers",
            attn_module_name="dummy",
            num_attention_heads=2,
        )
        explainer = MultiModalModelWrapper(
            model, config, mode="co_attention"
        )
        return model, explainer

    def test_generate_co_attention_output_keys(self) -> None:
        """Output should contain R_tt, R_ii, R_ti, R_it."""
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 5))
        pixel_values = torch.randn(1, 3, 8, 8)

        # Patch module resolution
        text_self = list(model.text_self_layers)
        vis_self = list(model.vision_self_layers)
        cross_mods = [model.text_cross, model.image_cross]

        all_mods = text_self + vis_self + cross_mods

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_co_attention(
            input_ids, pixel_values,
            target_index=0,
            text_self_layer_ids=["T.L0", "T.L1"],
            image_self_layer_ids=["V.L0", "V.L1"],
            cross_layer_ids=[("T.L0", "V.L0")],  # 1 co-attn block
        )

        self.assertIn("R_tt", result)
        self.assertIn("R_ii", result)
        self.assertIn("R_ti", result)
        self.assertIn("R_it", result)

    def test_co_attention_shapes(self) -> None:
        """R_tt should be (t,t), R_ii (i,i), R_ti (t,i), R_it (i,t)."""
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 5))
        pixel_values = torch.randn(1, 3, 8, 8)

        text_self = list(model.text_self_layers)
        vis_self = list(model.vision_self_layers)
        cross_mods = [model.text_cross, model.image_cross]
        all_mods = text_self + vis_self + cross_mods

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_co_attention(
            input_ids, pixel_values,
            target_index=0,
            text_self_layer_ids=["T.L0", "T.L1"],
            image_self_layer_ids=["V.L0", "V.L1"],
            cross_layer_ids=[("T.L0", "V.L0")],
        )

        t = 5  # text tokens
        # Image: 8x8 with 4x4 patches = 4 patches
        i_count = 4

        self.assertEqual(result["R_tt"].shape, (t, t))
        self.assertEqual(result["R_ii"].shape, (i_count, i_count))
        self.assertEqual(result["R_ti"].shape, (t, i_count))
        self.assertEqual(result["R_it"].shape, (i_count, t))

    def test_co_attention_cross_relevance_nontrivial(self) -> None:
        """R_ti should be non-zero after cross-attention propagation."""
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 5))
        pixel_values = torch.randn(1, 3, 8, 8)

        text_self = list(model.text_self_layers)
        vis_self = list(model.vision_self_layers)
        cross_mods = [model.text_cross, model.image_cross]
        all_mods = text_self + vis_self + cross_mods

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_co_attention(
            input_ids, pixel_values,
            target_index=0,
            text_self_layer_ids=["T.L0", "T.L1"],
            image_self_layer_ids=["V.L0", "V.L1"],
            cross_layer_ids=[("T.L0", "V.L0")],
        )
        self.assertTrue(
            result["R_ti"].abs().sum() > 0,
            "R_ti should be non-zero after cross-attention."
        )

    def test_co_attention_finite(self) -> None:
        """All relevance matrices should be finite."""
        model, explainer = self._make_model_and_explainer()
        input_ids = torch.randint(0, 32, (1, 5))
        pixel_values = torch.randn(1, 3, 8, 8)

        text_self = list(model.text_self_layers)
        vis_self = list(model.vision_self_layers)
        cross_mods = [model.text_cross, model.image_cross]
        all_mods = text_self + vis_self + cross_mods

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_co_attention(
            input_ids, pixel_values,
            target_index=0,
            text_self_layer_ids=["T.L0", "T.L1"],
            image_self_layer_ids=["V.L0", "V.L1"],
            cross_layer_ids=[("T.L0", "V.L0")],
        )
        for key, mat in result.items():
            self.assertTrue(
                torch.isfinite(mat).all(),
                f"{key} contains non-finite values."
            )


# ═══════════════════════════════════════════════════════════════════════
# Tests — MultiModalModelWrapper (encoder-decoder relevance)
# ═══════════════════════════════════════════════════════════════════════


class TestMultiModalModelWrapperEncoderDecoder(BaseTest):
    """Test Template C (encoder-decoder) explainability."""

    def _make_model_and_explainer(
        self,
    ) -> Tuple[DummyEncoderDecoderModel, MultiModalModelWrapper]:
        model = DummyEncoderDecoderModel(
            d_model=16, nhead=2,
            encoder_layers=2, decoder_layers=2,
            num_queries=4, num_classes=10,
        )
        model.eval()

        config = TransformerArchConfig(
            vision_encoder_prefix="encoder_layers",
            text_encoder_prefix="decoder_self_layers",
            attn_module_name="dummy",
            num_attention_heads=2,
        )
        explainer = MultiModalModelWrapper(
            model, config, mode="encoder_decoder"
        )
        return model, explainer

    def test_generate_enc_dec_output_keys(self) -> None:
        model, explainer = self._make_model_and_explainer()
        pixel_values = torch.randn(1, 3, 8, 8)

        enc_self = list(model.encoder_layers)
        dec_self = list(model.decoder_self_layers)
        dec_cross = list(model.decoder_cross_layers)
        all_mods = enc_self + dec_self + dec_cross

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_encoder_decoder(
            pixel_values,
            target_fn=lambda out: out[0, 0, 0],  # query 0, class 0
            encoder_self_layer_ids=["V.L0", "V.L1"],
            decoder_self_layer_ids=["T.L0", "T.L1"],
            decoder_cross_layer_ids=["T.L0", "T.L1"],
        )

        self.assertIn("R_ee", result)
        self.assertIn("R_qq", result)
        self.assertIn("R_qe", result)

    def test_enc_dec_shapes(self) -> None:
        model, explainer = self._make_model_and_explainer()
        pixel_values = torch.randn(1, 3, 8, 8)

        enc_self = list(model.encoder_layers)
        dec_self = list(model.decoder_self_layers)
        dec_cross = list(model.decoder_cross_layers)
        all_mods = enc_self + dec_self + dec_cross

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_encoder_decoder(
            pixel_values,
            target_fn=lambda out: out[0, 0, 0],
            encoder_self_layer_ids=["V.L0", "V.L1"],
            decoder_self_layer_ids=["T.L0", "T.L1"],
            decoder_cross_layer_ids=["T.L0", "T.L1"],
        )

        e = 4  # 8x8 with 4x4 patches = 4
        q = 4  # num_queries

        self.assertEqual(result["R_ee"].shape, (e, e))
        self.assertEqual(result["R_qq"].shape, (q, q))
        self.assertEqual(result["R_qe"].shape, (q, e))

    def test_enc_dec_cross_relevance_nontrivial(self) -> None:
        model, explainer = self._make_model_and_explainer()
        pixel_values = torch.randn(1, 3, 8, 8)

        enc_self = list(model.encoder_layers)
        dec_self = list(model.decoder_self_layers)
        dec_cross = list(model.decoder_cross_layers)
        all_mods = enc_self + dec_self + dec_cross

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_encoder_decoder(
            pixel_values,
            target_fn=lambda out: out[0, 0, 0],
            encoder_self_layer_ids=["V.L0", "V.L1"],
            decoder_self_layer_ids=["T.L0", "T.L1"],
            decoder_cross_layer_ids=["T.L0", "T.L1"],
        )

        self.assertTrue(
            result["R_qe"].abs().sum() > 0,
            "R_qe should be non-zero."
        )

    def test_enc_dec_finite(self) -> None:
        model, explainer = self._make_model_and_explainer()
        pixel_values = torch.randn(1, 3, 8, 8)

        enc_self = list(model.encoder_layers)
        dec_self = list(model.decoder_self_layers)
        dec_cross = list(model.decoder_cross_layers)
        all_mods = enc_self + dec_self + dec_cross

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_encoder_decoder(
            pixel_values,
            target_fn=lambda out: out[0, 0, 0],
            encoder_self_layer_ids=["V.L0", "V.L1"],
            decoder_self_layer_ids=["T.L0", "T.L1"],
            decoder_cross_layer_ids=["T.L0", "T.L1"],
        )
        for key, mat in result.items():
            self.assertTrue(
                torch.isfinite(mat).all(),
                f"{key} contains non-finite values."
            )

    def test_enc_dec_query_readout(self) -> None:
        """readout for query j should give encoder-token relevance."""
        model, explainer = self._make_model_and_explainer()
        pixel_values = torch.randn(1, 3, 8, 8)

        enc_self = list(model.encoder_layers)
        dec_self = list(model.decoder_self_layers)
        dec_cross = list(model.decoder_cross_layers)
        all_mods = enc_self + dec_self + dec_cross

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return all_mods

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        result = explainer.generate_encoder_decoder(
            pixel_values,
            target_fn=lambda out: out[0, 0, 0],
            encoder_self_layer_ids=["V.L0", "V.L1"],
            decoder_self_layer_ids=["T.L0", "T.L1"],
            decoder_cross_layer_ids=["T.L0", "T.L1"],
        )

        readout = MultiModalModelWrapper.readout(
            result, readout_index=0, modality="query"
        )
        self.assertIn("encoder", readout)
        self.assertEqual(readout["encoder"].shape[0], 4)  # encoder tokens


# ═══════════════════════════════════════════════════════════════════════
# Tests — generate_relevance (unified entry point)
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateRelevance(BaseTest):
    """Test the unified generate_relevance dispatcher."""

    def test_dispatch_single_stream(self) -> None:
        model = DummySingleStreamModel(vocab_size=32, d_model=16, nhead=2, num_layers=2)
        model.eval()
        config = TransformerArchConfig(
            text_encoder_prefix="layers",
            attn_module_name="dummy",
            num_attention_heads=2,
        )
        explainer = MultiModalModelWrapper(model, config, mode="single_stream")

        def patched_resolve(layer_ids: List[str]) -> List[nn.Module]:
            return list(model.layers)

        explainer._resolve_attn_modules = patched_resolve  # type: ignore[assignment]

        input_ids = torch.randint(0, 32, (1, 5))
        R = explainer.generate_relevance(
            input_ids, target_index=0, num_layers=2
        )
        self.assertIsInstance(R, Tensor)
        self.assertEqual(R.shape, (5, 5))

    def test_dispatch_raises_on_bad_mode(self) -> None:
        model = DummySingleStreamModel()
        config = TransformerArchConfig(text_encoder_prefix="layers")
        with self.assertRaises(ValueError):
            MultiModalModelWrapper(model, config, mode="invalid_mode")


# ═══════════════════════════════════════════════════════════════════════
# Tests — ExplainabilityMode enum
# ═══════════════════════════════════════════════════════════════════════


class TestExplainabilityMode(BaseTest):
    def test_values(self) -> None:
        self.assertEqual(ExplainabilityMode.SINGLE_STREAM, "single_stream")
        self.assertEqual(ExplainabilityMode.CO_ATTENTION, "co_attention")
        self.assertEqual(ExplainabilityMode.ENCODER_DECODER, "encoder_decoder")

    def test_from_string(self) -> None:
        mode = ExplainabilityMode("single_stream")
        self.assertEqual(mode, ExplainabilityMode.SINGLE_STREAM)

    def test_invalid_string(self) -> None:
        with self.assertRaises(ValueError):
            ExplainabilityMode("nonexistent")


# ═══════════════════════════════════════════════════════════════════════
# Tests — Readout utility
# ═══════════════════════════════════════════════════════════════════════


class TestReadout(BaseTest):
    def test_tensor_readout(self) -> None:
        R = torch.rand(6, 6)
        row = MultiModalModelWrapper.readout(R, readout_index=2)
        self.assertTrue(torch.allclose(row, R[2]))

    def test_dict_readout_text(self) -> None:
        R = {
            "R_tt": torch.rand(5, 5),
            "R_ti": torch.rand(5, 3),
        }
        result = MultiModalModelWrapper.readout(R, readout_index=0, modality="text")
        self.assertIn("text", result)
        self.assertIn("image", result)
        self.assertEqual(result["text"].shape, (5,))
        self.assertEqual(result["image"].shape, (3,))

    def test_dict_readout_no_modality(self) -> None:
        R = {
            "R_tt": torch.rand(5, 5),
            "R_ii": torch.rand(3, 3),
            "R_ti": torch.rand(5, 3),
        }
        result = MultiModalModelWrapper.readout(R, readout_index=0)
        self.assertEqual(len(result), 3)


# ═══════════════════════════════════════════════════════════════════════
# Tests — MultiModalAttribution
# ═══════════════════════════════════════════════════════════════════════


@parameterized_class(
    ("device",),
    [("cpu",), ("cuda",)] if torch.cuda.is_available() else [("cpu",)],
)
class TestMultiModalAttribution(BaseTest):
    device: str = "cpu"

    def test_layer_activation_text(self) -> None:
        """Test LayerActivation through MultiModalAttribution on text model."""
        model = DummyTextEncoder().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)

        mm_attr = MultiModalAttribution(wrapper, LayerActivation, "L0")
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        result = mm_attr.attribute(input_ids)

        self.assertIsInstance(result, Tensor)
        # LayerActivation returns the layer output
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 5)

    def test_layer_activation_vision(self) -> None:
        """Test LayerActivation through MultiModalAttribution on vision model."""
        model = DummyVisionEncoder().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(
            model, DUMMY_VISION_CONFIG, device=self.device
        )

        mm_attr = MultiModalAttribution(wrapper, LayerActivation, "L0")
        pixel_values = torch.randn(1, 3, 8, 8, device=self.device)
        result = mm_attr.attribute(pixel_values)

        self.assertIsInstance(result, Tensor)

    def test_layer_activation_lvlm_vision(self) -> None:
        """Test attribution on vision encoder layer of LVLM."""
        model = DummyLVLM().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_LVLM_CONFIG, device=self.device)

        mm_attr = MultiModalAttribution(wrapper, LayerActivation, "V.L0")
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        pixel_values = torch.randn(1, 3, 8, 8, device=self.device)
        result = mm_attr.attribute(
            (input_ids, pixel_values),
            additional_forward_args=None,
        )
        self.assertIsInstance(result, Tensor)

    def test_layer_activation_lvlm_text(self) -> None:
        """Test attribution on text encoder layer of LVLM."""
        model = DummyLVLM().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_LVLM_CONFIG, device=self.device)

        mm_attr = MultiModalAttribution(wrapper, LayerActivation, "T.L1")
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        pixel_values = torch.randn(1, 3, 8, 8, device=self.device)
        result = mm_attr.attribute(
            (input_ids, pixel_values),
            additional_forward_args=None,
        )
        self.assertIsInstance(result, Tensor)

    def test_layer_activation_attribute_to_input(self) -> None:
        """Test attribute_to_layer_input flag."""
        model = DummyTextEncoder().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)

        mm_attr = MultiModalAttribution(wrapper, LayerActivation, "L1")
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)

        result_output = mm_attr.attribute(input_ids, attribute_to_layer_input=False)
        result_input = mm_attr.attribute(input_ids, attribute_to_layer_input=True)

        # Both should be tensors, but may differ in values
        self.assertIsInstance(result_output, Tensor)
        self.assertIsInstance(result_input, Tensor)

    def test_layer_integrated_gradients_text(self) -> None:
        """Test LayerIntegratedGradients through MultiModalAttribution."""
        model = DummyTextEncoder().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)

        mm_attr = MultiModalAttribution(
            wrapper, LayerIntegratedGradients, "L0"
        )
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        baselines = torch.zeros_like(input_ids)

        # DummyTextEncoder output: (batch, seq_len, vocab_size)
        # Use tuple target for 3D output: (seq_pos, class_idx)
        result = mm_attr.attribute(
            input_ids,
            baselines=baselines,
            target=(0, 0),
            n_steps=5,
        )
        self.assertIsInstance(result, Tensor)

    def test_layer_conductance_text(self) -> None:
        """Test LayerConductance through MultiModalAttribution."""
        model = DummyTextEncoder().to(self.device)
        model.eval()

        # LayerConductance interpolates between baselines and inputs,
        # which requires float tensors. Use a wrapper that takes
        # pre-embedded float inputs instead of integer token IDs.
        d_model = 16

        class EmbeddedTextModel(nn.Module):
            def __init__(self, encoder: nn.Module) -> None:
                super().__init__()
                self.layers = encoder.layers
                self.head = encoder.head

            def forward(self, x: Tensor) -> Tensor:
                for layer in self.layers:
                    x = layer(x)
                return self.head(x)

        embedded_model = EmbeddedTextModel(model).to(self.device)
        embedded_model.eval()
        embedded_config = TransformerArchConfig(
            text_encoder_prefix="layers",
            attn_module_name="self_attn",
            mlp_module_name="mlp",
            output_module_name="layer_norm",
            num_attention_heads=2,
        )
        wrapper = MultiModalModelWrapper(
            embedded_model, embedded_config, device=self.device
        )

        mm_attr = MultiModalAttribution(wrapper, LayerConductance, "L0")
        inputs = torch.randn(1, 5, d_model, device=self.device)
        baselines = torch.zeros_like(inputs)

        # Use tuple target for 3D output
        result = mm_attr.attribute(
            inputs,
            baselines=baselines,
            target=(0, 0),
            n_steps=5,
        )
        self.assertIsInstance(result, Tensor)

    def test_unsupported_method_raises(self) -> None:
        """Test that unsupported attribution methods raise ValueError."""
        from captum.attr._core.saliency import Saliency

        model = DummyTextEncoder().to(self.device)
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)

        with self.assertRaises(ValueError):
            # Saliency is not a LayerAttribution method
            MultiModalAttribution(wrapper, Saliency, "L0")  # type: ignore[arg-type]

    def test_different_layers_give_different_results(self) -> None:
        """Test that attributions from different layers produce different results."""
        torch.manual_seed(42)
        model = DummyTextEncoder().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)

        input_ids = torch.randint(0, 64, (1, 5), device=self.device)

        mm_attr_l0 = MultiModalAttribution(wrapper, LayerActivation, "L0")
        mm_attr_l1 = MultiModalAttribution(wrapper, LayerActivation, "L1")

        result_l0 = mm_attr_l0.attribute(input_ids)
        result_l1 = mm_attr_l1.attribute(input_ids)

        # Results from different layers should differ
        self.assertFalse(torch.allclose(result_l0, result_l1))

    def test_component_attribution(self) -> None:
        """Test attribution targeting a specific sub-component (attn, mlp)."""
        model = DummyTextEncoder().to(self.device)
        model.eval()
        wrapper = MultiModalModelWrapper(model, DUMMY_TEXT_CONFIG, device=self.device)

        # Attribute to the MLP sub-module
        mm_attr = MultiModalAttribution(wrapper, LayerActivation, "L0.mlp")
        input_ids = torch.randint(0, 64, (1, 5), device=self.device)
        result = mm_attr.attribute(input_ids)
        self.assertIsInstance(result, Tensor)
