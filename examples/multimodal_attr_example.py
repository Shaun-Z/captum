#!/usr/bin/env python3
"""
Multimodal Transformer Attribution — Runnable Example
=====================================================

This script demonstrates the captum multimodal attribution API using
lightweight dummy models. No model downloads or GPUs required — just
copy-paste and run.

It covers three scenarios:
  1. Text-only (LLM-like) model — LayerActivation on a specific layer
  2. Vision-only (ViT-like) model — LayerActivation on a vision layer
  3. Multimodal (LVLM-like) model — attribution on both vision and text
     encoder layers in a single model

Usage:
    pip install captum
    python examples/multimodal_attr_example.py
"""

from typing import Optional

import torch
from torch import nn, Tensor

# ---------------------------------------------------------------------------
# 1. Import captum's multimodal attribution API
# ---------------------------------------------------------------------------
from captum._utils.transformer import (
    ARCH_CONFIGS,
    LayerID,
    TransformerArchConfig,
)
from captum.attr import (
    LayerActivation,
    MultiModalAttribution,
    MultiModalModelWrapper,
)

# ---------------------------------------------------------------------------
# 2. Define lightweight dummy models (no downloads needed)
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """A single transformer block with self-attn + MLP + LayerNorm."""

    def __init__(self, d: int = 32, nhead: int = 4) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.layer_norm(x + attn_out)
        return x + self.mlp(x)


class TinyLLM(nn.Module):
    """A toy text-only transformer (3 layers, vocab 128)."""

    def __init__(self, vocab: int = 128, d: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TransformerLayer(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class TinyViT(nn.Module):
    """A toy vision transformer (2 layers, 4x4 patches)."""

    def __init__(self, d: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(3 * 4 * 4, d)
        self.layers = nn.ModuleList([TransformerLayer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, pixel_values: Tensor) -> Tensor:
        b = pixel_values.shape[0]
        patches = pixel_values.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.contiguous().view(b, -1, 3 * 4 * 4)
        x = self.patch_embed(patches)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyLVLM(nn.Module):
    """A toy vision-language model combining TinyViT + TinyLLM."""

    def __init__(self, vocab: int = 128, d: int = 32) -> None:
        super().__init__()
        self.vision_encoder = TinyViT(d, n_layers=2)
        self.text_encoder = TinyLLM(vocab, d, n_layers=3)
        self.projection = nn.Linear(d, d)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
    ) -> Tensor:
        extra = None
        if pixel_values is not None:
            v = self.vision_encoder(pixel_values)
            extra = self.projection(v).mean(dim=1, keepdim=True)
        if input_ids is not None:
            x = self.text_encoder.embedding(input_ids)
            if extra is not None:
                x = x + extra.expand(-1, x.shape[1], -1)
            for layer in self.text_encoder.layers:
                x = layer(x)
            return self.text_encoder.head(x)
        assert extra is not None
        return extra


# ---------------------------------------------------------------------------
# 3. Register custom architecture configs for our dummy models
# ---------------------------------------------------------------------------

# Text-only model: layers live at "layers.<i>"
text_config = TransformerArchConfig(
    text_encoder_prefix="layers",
    attn_module_name="self_attn",
    mlp_module_name="mlp",
    output_module_name="layer_norm",
    num_attention_heads=4,
)

# Vision-only model: layers also at "layers.<i>"
vision_config = TransformerArchConfig(
    vision_encoder_prefix="layers",
    default_encoder_prefix="layers",
    attn_module_name="self_attn",
    mlp_module_name="mlp",
    output_module_name="layer_norm",
    num_attention_heads=4,
)

# Multimodal model: vision at "vision_encoder.layers", text at "text_encoder.layers"
lvlm_config = TransformerArchConfig(
    vision_encoder_prefix="vision_encoder.layers",
    text_encoder_prefix="text_encoder.layers",
    attn_module_name="self_attn",
    mlp_module_name="mlp",
    output_module_name="layer_norm",
    num_attention_heads=4,
)


def main() -> None:
    torch.manual_seed(42)

    # -----------------------------------------------------------------------
    # Example 1: Text-only LLM — get layer activations
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Text-only LLM — LayerActivation")
    print("=" * 60)

    llm = TinyLLM()
    llm.eval()

    wrapper = MultiModalModelWrapper(llm, text_config)
    mm_attr = MultiModalAttribution(wrapper, LayerActivation, "L0")

    input_ids = torch.randint(0, 128, (1, 10))  # batch=1, seq_len=10
    activation = mm_attr.attribute(input_ids)

    print(f"  Input shape:      {input_ids.shape}")
    print(f"  Layer L0 output:  {activation.shape}")
    print(f"  Activation mean:  {activation.mean().item():.4f}")
    print()

    # You can also resolve modules directly
    layer1_attn = wrapper.resolve_layer("L1.attn")
    print(f"  Resolved L1.attn: {layer1_attn.__class__.__name__}")

    # Or extract raw activations with a forward pass
    act = wrapper.get_activation("L2", input_ids)
    print(f"  L2 activation:    {act.shape}")
    print()

    # -----------------------------------------------------------------------
    # Example 2: Vision-only ViT — get layer activations
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Example 2: Vision-only ViT — LayerActivation")
    print("=" * 60)

    vit = TinyViT()
    vit.eval()

    wrapper_v = MultiModalModelWrapper(vit, vision_config)
    mm_attr_v = MultiModalAttribution(wrapper_v, LayerActivation, "L0")

    image = torch.randn(1, 3, 8, 8)  # batch=1, 3 channels, 8x8 image
    activation_v = mm_attr_v.attribute(image)

    print(f"  Image shape:      {image.shape}")
    print(f"  Layer L0 output:  {activation_v.shape}")
    print(f"  Activation mean:  {activation_v.mean().item():.4f}")
    print()

    # -----------------------------------------------------------------------
    # Example 3: Multimodal LVLM — attribution on both encoders
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Example 3: Multimodal LVLM — Vision + Text attribution")
    print("=" * 60)

    lvlm = TinyLVLM()
    lvlm.eval()

    wrapper_m = MultiModalModelWrapper(lvlm, lvlm_config)

    # Resolve vision and text layers using compact notation
    v_layer = wrapper_m.resolve_layer("V.L0")
    t_layer = wrapper_m.resolve_layer("T.L2")
    print(f"  V.L0 module:  {v_layer.__class__.__name__}")
    print(f"  T.L2 module:  {t_layer.__class__.__name__}")
    print()

    # Get activations from vision encoder layer 1
    image = torch.randn(1, 3, 8, 8)
    input_ids = torch.randint(0, 128, (1, 6))

    # Attribution on vision encoder layer
    mm_attr_vis = MultiModalAttribution(wrapper_m, LayerActivation, "V.L1")
    vis_result = mm_attr_vis.attribute((input_ids, image))
    print(f"  V.L1 activation shape: {vis_result.shape}")

    # Attribution on text encoder layer
    mm_attr_txt = MultiModalAttribution(wrapper_m, LayerActivation, "T.L0")
    txt_result = mm_attr_txt.attribute((input_ids, image))
    print(f"  T.L0 activation shape: {txt_result.shape}")
    print()

    # Multi-layer activation extraction in one forward pass
    acts = wrapper_m.get_multi_layer_activations(
        ["V.L0", "V.L1", "T.L0", "T.L1", "T.L2"],
        input_ids,
        pixel_values=image,
    )
    print("  Multi-layer activations (single forward pass):")
    for layer_name, act_tensor in acts.items():
        print(f"    {layer_name}: {act_tensor.shape}")
    print()

    # -----------------------------------------------------------------------
    # Example 4: LayerID parsing demo
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Example 4: LayerID parsing")
    print("=" * 60)

    for s in ["L0", "V.L1.H5", "T.L3.attn", "L2.mlp", "V.L0.output"]:
        lid = LayerID.parse(s)
        print(
            f"  '{s}' -> encoder={lid.encoder_type}, "
            f"layer={lid.layer_num}, "
            f"head={lid.head_num}, "
            f"component={lid.component}"
        )
    print()

    # -----------------------------------------------------------------------
    # Example 5: Available pre-registered architectures
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Example 5: Pre-registered architecture configs")
    print("=" * 60)

    for name, cfg in ARCH_CONFIGS.items():
        print(
            f"  {name:15s}  V={cfg.vision_encoder_prefix or '-':>45s}  "
            f"T={cfg.text_encoder_prefix or '-'}"
        )
    print()
    print("Done!")


if __name__ == "__main__":
    main()
