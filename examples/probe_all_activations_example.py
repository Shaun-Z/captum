#!/usr/bin/env python3
"""
Probe All Intermediate Activations — Pretrained GPT-2 Example
==============================================================

This script demonstrates how to probe **every** intermediate activation
inside the first transformer block of a pretrained GPT-2 model using
captum's ``ActivationAccessor`` and compact layer-ID notation.

GPT-2's transformer block has the following structure::

    GPT2Block
    ├── ln_1          (LayerNorm — pre-attention norm)
    ├── attn          (GPT2Attention)
    │   ├── c_attn    (Conv1D — combined Q/K/V projection)
    │   └── c_proj    (Conv1D — output projection)
    ├── ln_2          (LayerNorm — pre-FFN norm)
    └── mlp           (GPT2MLP)
        ├── c_fc      (Conv1D — FFN input projection, d→4d)
        ├── act       (NewGELUActivation — non-linear activation)
        └── c_proj    (Conv1D — FFN output projection, 4d→d)

The layer-ID notation lets you target any of these::

    "L0"              → full block output (GPT2Block)
    "L0.attn"         → attention sub-module (GPT2Attention)
    "L0.attn.c_attn"  → combined Q/K/V projection
    "L0.attn.c_proj"  → attention output projection
    "L0.mlp"          → full MLP / FFN output (GPT2MLP)
    "L0.mlp.c_fc"     → FFN input projection  (activation fn INPUT)
    "L0.mlp.act"      → GELU activation       (activation fn OUTPUT)
    "L0.mlp.c_proj"   → FFN output projection
    "L0.output"       → pre-FFN LayerNorm (ln_2)

For **multimodal** models, prefix with ``V.`` or ``T.``::

    "V.L0.mlp.act"   → vision encoder, layer 0, activation fn output
    "T.L2.mlp.c_fc"  → text encoder, layer 2, FFN input projection

Requirements:
    pip install captum transformers

Usage:
    python examples/probe_all_activations_example.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from captum._utils.transformer.accessor import ActivationAccessor


def main() -> None:
    torch.manual_seed(42)

    # -------------------------------------------------------------------
    # 1. Load pretrained GPT-2 model and tokenizer
    # -------------------------------------------------------------------
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print(
        f"Model loaded: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )
    print()

    # -------------------------------------------------------------------
    # 2. Create ActivationAccessor with built-in "gpt2" architecture config
    #    This maps:
    #      L<i>       → transformer.h.<i>
    #      L<i>.attn  → transformer.h.<i>.attn
    #      L<i>.mlp   → transformer.h.<i>.mlp
    #      L<i>.output→ transformer.h.<i>.ln_2
    # -------------------------------------------------------------------
    accessor = ActivationAccessor(model, "gpt2")

    # -------------------------------------------------------------------
    # 3. Tokenize input
    # -------------------------------------------------------------------
    text = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    print(f"Input text:  '{text}'")
    print(f"Input shape: {input_ids.shape}")
    print()

    # -------------------------------------------------------------------
    # 4. All probe-able locations in the first transformer block (L0)
    # -------------------------------------------------------------------
    probe_ids = [
        # Full block
        ("L0",              "Full block output (GPT2Block)"),
        # Attention sub-module and its internals
        ("L0.attn",         "Self-attention output (GPT2Attention)"),
        ("L0.attn.c_attn",  "Combined Q/K/V projection (Conv1D)"),
        ("L0.attn.c_proj",  "Attention output projection (Conv1D)"),
        # Full MLP / FFN
        ("L0.mlp",          "Full MLP output (GPT2MLP)"),
        # FFN internals — dotted sub-path notation
        ("L0.mlp.c_fc",     "FFN input projection (activation fn INPUT)"),
        ("L0.mlp.act",      "GELU activation (activation fn OUTPUT)"),
        ("L0.mlp.c_proj",   "FFN output projection"),
        # Pre-FFN LayerNorm
        ("L0.output",       "Pre-FFN LayerNorm (ln_2)"),
    ]

    print("=" * 76)
    print("Probing all intermediate activations in GPT-2 block 0")
    print("=" * 76)

    # --- Method 1: one-by-one extraction ---
    print("\n--- Method 1: Single-layer extraction (one forward pass each) ---\n")
    for layer_id, description in probe_ids:
        module = accessor.resolve_module(layer_id)
        act = accessor.get_activation(layer_id, input_ids)
        print(
            f"  {layer_id:20s}  shape={str(act.shape):25s}  "
            f"module={module.__class__.__name__:25s}  # {description}"
        )

    # --- Method 2: batch extraction (single forward pass) ---
    print("\n--- Method 2: Multi-layer extraction (ONE forward pass) ---\n")
    all_ids = [pid for pid, _ in probe_ids]
    activations = accessor.get_multi_layer_activations(all_ids, input_ids)
    for layer_id, description in probe_ids:
        act = activations[layer_id]
        print(
            f"  {layer_id:20s}  shape={str(act.shape):25s}  # {description}"
        )

    # --- Verify FFN activation function properties ---
    print("\n--- Verification: FFN activation function ---\n")
    fc_output = activations["L0.mlp.c_fc"]
    act_output = activations["L0.mlp.act"]
    print(
        f"  c_fc output  (activation fn input)   "
        f"min={fc_output.min():.4f}  max={fc_output.max():.4f}"
    )
    print(
        f"  act output   (activation fn output)  "
        f"min={act_output.min():.4f}  max={act_output.max():.4f}"
    )

    # --- Probe layer input (instead of output) ---
    print("\n--- Bonus: Capture layer INPUT instead of output ---\n")
    act_input = accessor.get_activation(
        "L0.mlp.act", input_ids, attribute_to_layer_input=True
    )
    print(
        f"  L0.mlp.act input  shape={act_input.shape}  "
        f"(should match c_fc output)"
    )
    print(
        f"  Matches c_fc output? "
        f"{torch.allclose(act_input, fc_output, atol=1e-6)}"
    )

    # --- Multi-layer extraction across all 12 blocks ---
    print("\n--- Extracting MLP activations from all 12 blocks ---\n")
    mlp_act_ids = [f"L{i}.mlp.act" for i in range(12)]
    mlp_acts = accessor.get_multi_layer_activations(mlp_act_ids, input_ids)
    for name, act in mlp_acts.items():
        print(
            f"  {name:15s}  shape={str(act.shape):25s}  "
            f"mean={act.mean().item():.6f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
