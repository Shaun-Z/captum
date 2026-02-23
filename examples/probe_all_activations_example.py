#!/usr/bin/env python3
"""
Probe All Intermediate Activations — Example
=============================================

This script demonstrates how to probe **every** intermediate activation
inside the first transformer block using captum's ``ActivationAccessor``
and compact layer-ID notation.

In a standard transformer layer::

    TransformerLayer
    ├── self_attn   (MultiheadAttention)
    ├── mlp         (Sequential)
    │   ├── 0       (Linear — FFN input projection)
    │   ├── 1       (GELU  — non-linear activation)
    │   └── 2       (Linear — FFN output projection)
    └── layer_norm  (LayerNorm)

The layer-ID notation lets you target any of these::

    "L0"            → full layer output
    "L0.attn"       → attention sub-module output
    "L0.mlp"        → full MLP / FFN output
    "L0.mlp.0"      → first Linear output   (activation fn input)
    "L0.mlp.1"      → GELU activation output (activation fn output)
    "L0.mlp.2"      → second Linear output   (FFN projection)
    "L0.output"     → LayerNorm output

For **multimodal** models, prefix with ``V.`` or ``T.``::

    "V.L0.mlp.1"    → vision encoder, layer 0, activation fn output
    "T.L2.mlp.0"    → text encoder, layer 2, FFN input projection

Usage:
    pip install captum
    python examples/probe_all_activations_example.py
"""

import torch
from torch import nn, Tensor

from captum._utils.transformer import TransformerArchConfig
from captum._utils.transformer.accessor import ActivationAccessor


# ---------------------------------------------------------------------------
# 1. Dummy transformer model (same structure as real models)
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """A single transformer block with self-attn + MLP + LayerNorm."""

    def __init__(self, d: int = 32, nhead: int = 4) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4),   # index 0: FFN input projection
            nn.GELU(),             # index 1: non-linear activation
            nn.Linear(d * 4, d),   # index 2: FFN output projection
        )
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
        self.layers = nn.ModuleList(
            [TransformerLayer(d) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# 2. Architecture config
# ---------------------------------------------------------------------------

config = TransformerArchConfig(
    text_encoder_prefix="layers",
    attn_module_name="self_attn",
    mlp_module_name="mlp",
    output_module_name="layer_norm",
    num_attention_heads=4,
)


def main() -> None:
    torch.manual_seed(42)

    model = TinyLLM()
    model.eval()
    accessor = ActivationAccessor(model, config)

    input_ids = torch.randint(0, 128, (1, 8))  # batch=1, seq_len=8
    print(f"Input shape: {input_ids.shape}\n")

    # -------------------------------------------------------------------
    # All probe-able locations in the first transformer block (L0)
    # -------------------------------------------------------------------
    probe_ids = [
        # Layer-level
        ("L0",          "Full layer output"),
        # Attention sub-module
        ("L0.attn",     "Self-attention output"),
        # Full MLP / FFN
        ("L0.mlp",      "Full MLP/FFN output"),
        # FFN internals  ← NEW: dotted sub-path notation
        ("L0.mlp.0",    "FFN Linear-1 output  (activation fn INPUT)"),
        ("L0.mlp.1",    "FFN GELU output      (activation fn OUTPUT)"),
        ("L0.mlp.2",    "FFN Linear-2 output  (FFN projection)"),
        # LayerNorm
        ("L0.output",   "LayerNorm output"),
    ]

    print("=" * 72)
    print("Probing all intermediate activations in the first transformer block")
    print("=" * 72)

    # --- Method 1: one-by-one extraction ---
    print("\n--- Method 1: Single-layer extraction (one forward pass each) ---\n")
    for layer_id, description in probe_ids:
        module = accessor.resolve_module(layer_id)
        act = accessor.get_activation(layer_id, input_ids)
        print(
            f"  {layer_id:15s}  shape={str(act.shape):20s}  "
            f"module={module.__class__.__name__:25s}  # {description}"
        )

    # --- Method 2: batch extraction (single forward pass) ---
    print("\n--- Method 2: Multi-layer extraction (ONE forward pass) ---\n")
    all_ids = [pid for pid, _ in probe_ids]
    activations = accessor.get_multi_layer_activations(all_ids, input_ids)
    for layer_id, description in probe_ids:
        act = activations[layer_id]
        print(
            f"  {layer_id:15s}  shape={str(act.shape):20s}  # {description}"
        )

    # --- Verify FFN activation function properties ---
    print("\n--- Verification: FFN activation function ---\n")
    gelu_input = activations["L0.mlp.0"]
    gelu_output = activations["L0.mlp.1"]
    expected_output = torch.nn.functional.gelu(gelu_input)
    match = torch.allclose(gelu_output, expected_output, atol=1e-6)
    print(f"  GELU(Linear-1 output) == mlp.1 output? {match}")
    print(f"  Linear-1 output (activation fn input)  min={gelu_input.min():.4f}  "
          f"max={gelu_input.max():.4f}")
    print(f"  GELU output (activation fn output)     min={gelu_output.min():.4f}  "
          f"max={gelu_output.max():.4f}")

    # --- Probe layer input (instead of output) ---
    print("\n--- Bonus: Capture layer INPUT instead of output ---\n")
    act_input = accessor.get_activation(
        "L0.mlp.1", input_ids, attribute_to_layer_input=True
    )
    print(f"  L0.mlp.1 input  shape={act_input.shape}  "
          f"(should match Linear-1 output)")
    print(f"  Matches mlp.0 output? "
          f"{torch.allclose(act_input, gelu_input, atol=1e-6)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
