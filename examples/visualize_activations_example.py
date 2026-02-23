#!/usr/bin/env python3
"""
Visualize Probed Activations — Pretrained GPT-2 Example
========================================================

This script demonstrates how to **visualize** intermediate activations
probed from a pretrained GPT-2 model. It uses the visualization utilities
in ``captum._utils.transformer.visualization`` to produce:

  1. **Heatmaps** of per-token, per-feature activation values
  2. **Summary statistics** (mean ± std per token) across the feature
     dimension
  3. **Distribution plots** comparing activation value distributions
     across different sub-modules

Prerequisites:
    pip install captum transformers matplotlib

Usage:
    python examples/visualize_activations_example.py

The script saves the figures to PNG files in the current directory
and also displays them if running interactively.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from captum._utils.transformer.accessor import ActivationAccessor
from captum._utils.transformer.visualization import (
    visualize_activation_distribution,
    visualize_activation_stats,
    visualize_activations,
)


def main() -> None:
    torch.manual_seed(42)

    # -------------------------------------------------------------------
    # 1. Load pretrained GPT-2
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

    # -------------------------------------------------------------------
    # 2. Tokenize input
    # -------------------------------------------------------------------
    text = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"Input: '{text}'")
    print(f"Tokens: {tokens}")
    print()

    # -------------------------------------------------------------------
    # 3. Create accessor and extract activations
    # -------------------------------------------------------------------
    accessor = ActivationAccessor(model, "gpt2")

    # Probe FFN internals in block 0
    ffn_ids = ["L0.mlp.c_fc", "L0.mlp.act", "L0.mlp.c_proj"]
    ffn_acts = accessor.get_multi_layer_activations(ffn_ids, input_ids)

    # Probe attention internals in block 0
    attn_ids = ["L0.attn.c_attn", "L0.attn.c_proj"]
    attn_acts = accessor.get_multi_layer_activations(attn_ids, input_ids)

    # Probe GELU activation across all 12 blocks
    all_gelu_ids = [f"L{i}.mlp.act" for i in range(12)]
    all_gelu_acts = accessor.get_multi_layer_activations(
        all_gelu_ids, input_ids
    )

    # -------------------------------------------------------------------
    # 4. Visualization 1: FFN internals heatmap
    # -------------------------------------------------------------------
    print("Generating FFN activation heatmaps...")
    fig1 = visualize_activations(
        ffn_acts,
        tokens=tokens,
        title="GPT-2 Block 0 — FFN Intermediate Activations",
        use_pyplot=False,
    )
    fig1.savefig("ffn_activations_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved: ffn_activations_heatmap.png")

    # -------------------------------------------------------------------
    # 5. Visualization 2: Attention internals heatmap
    # -------------------------------------------------------------------
    print("Generating attention activation heatmaps...")
    fig2 = visualize_activations(
        attn_acts,
        tokens=tokens,
        title="GPT-2 Block 0 — Attention Intermediate Activations",
        use_pyplot=False,
    )
    fig2.savefig("attn_activations_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved: attn_activations_heatmap.png")

    # -------------------------------------------------------------------
    # 6. Visualization 3: Per-token summary statistics
    # -------------------------------------------------------------------
    print("Generating per-token statistics chart...")
    fig3 = visualize_activation_stats(
        ffn_acts,
        tokens=tokens,
        title="GPT-2 Block 0 — FFN Per-Token Statistics (mean ± std)",
        use_pyplot=False,
    )
    fig3.savefig("ffn_activation_stats.png", dpi=150, bbox_inches="tight")
    print("  Saved: ffn_activation_stats.png")

    # -------------------------------------------------------------------
    # 7. Visualization 4: Activation distribution comparison
    # -------------------------------------------------------------------
    print("Generating activation distribution plot...")
    # Compare GELU output distribution across early, mid, and late blocks
    dist_acts = {
        "L0.mlp.act": all_gelu_acts["L0.mlp.act"],
        "L5.mlp.act": all_gelu_acts["L5.mlp.act"],
        "L11.mlp.act": all_gelu_acts["L11.mlp.act"],
    }
    fig4 = visualize_activation_distribution(
        dist_acts,
        title="GELU Activation Distribution: Early vs Mid vs Late Blocks",
        use_pyplot=False,
    )
    fig4.savefig(
        "gelu_activation_distribution.png", dpi=150, bbox_inches="tight"
    )
    print("  Saved: gelu_activation_distribution.png")

    # -------------------------------------------------------------------
    # 8. Visualization 5: All GELU activations across 12 blocks
    # -------------------------------------------------------------------
    print("Generating all-blocks GELU statistics...")
    fig5 = visualize_activation_stats(
        all_gelu_acts,
        tokens=tokens,
        title="GPT-2 — GELU Activation Statistics Across All 12 Blocks",
        use_pyplot=False,
    )
    fig5.savefig(
        "all_blocks_gelu_stats.png", dpi=150, bbox_inches="tight"
    )
    print("  Saved: all_blocks_gelu_stats.png")

    print("\nDone! All visualizations saved.")


if __name__ == "__main__":
    main()
