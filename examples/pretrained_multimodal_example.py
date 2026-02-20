#!/usr/bin/env python3
"""
Pretrained Model Attribution — LLaVA / GPT-2 Example
=====================================================

This script demonstrates how to use captum's multimodal attribution API
with real pretrained models from HuggingFace.

Two examples are provided:
  1. GPT-2 (text-only, ~124M params, runs on CPU)
  2. LLaVA-1.5 (multimodal vision-language, ~7B params, requires GPU)

Requirements:
    pip install captum transformers accelerate pillow requests

Usage:
    # Run just the GPT-2 example (works on CPU):
    python examples/pretrained_multimodal_example.py --gpt2

    # Run just the LLaVA example (requires GPU with ≥16GB VRAM):
    python examples/pretrained_multimodal_example.py --llava

    # Run both:
    python examples/pretrained_multimodal_example.py --gpt2 --llava
"""

import argparse
import sys

import torch


def example_gpt2() -> None:
    """
    Example 1: GPT-2 (text-only pretrained LLM)
    =============================================
    Uses the built-in "gpt2" architecture config.
    Runs on CPU, no GPU required (~124M parameters).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from captum._utils.transformer import LayerID
    from captum.attr import (
        LayerActivation,
        MultiModalAttribution,
        MultiModalModelWrapper,
    )

    print("=" * 60)
    print("Example 1: GPT-2 — Pretrained Text LLM")
    print("=" * 60)

    # 1. Load model and tokenizer
    model_name = "gpt2"
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # 2. Wrap with captum's MultiModalModelWrapper
    #    "gpt2" is a pre-registered architecture config
    wrapper = MultiModalModelWrapper(model, "gpt2")

    # 3. Resolve specific layers using compact notation
    #    GPT-2 layers are at: transformer.h.<i>
    #    L0 = transformer.h.0, L11 = transformer.h.11
    layer_0 = wrapper.resolve_layer("L0")
    layer_11 = wrapper.resolve_layer("L11")
    attn_5 = wrapper.resolve_layer("L5.attn")
    mlp_5 = wrapper.resolve_layer("L5.mlp")
    print(f"  L0:      {layer_0.__class__.__name__}")
    print(f"  L11:     {layer_11.__class__.__name__}")
    print(f"  L5.attn: {attn_5.__class__.__name__}")
    print(f"  L5.mlp:  {mlp_5.__class__.__name__}")
    print()

    # 4. Tokenize input
    text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"  Input text:   '{text}'")
    print(f"  Token IDs:    {input_ids.shape}")
    print()

    # 5. Get layer activations using MultiModalAttribution + LayerActivation
    mm_attr = MultiModalAttribution(wrapper, LayerActivation, "L0")
    activation = mm_attr.attribute(input_ids)
    print(f"  L0  activation shape: {activation.shape}")
    print(f"  L0  activation mean:  {activation.mean().item():.6f}")

    mm_attr_11 = MultiModalAttribution(wrapper, LayerActivation, "L11")
    activation_11 = mm_attr_11.attribute(input_ids)
    print(f"  L11 activation shape: {activation_11.shape}")
    print(f"  L11 activation mean:  {activation_11.mean().item():.6f}")
    print()

    # 6. Extract activations from multiple layers in one forward pass
    layer_ids = [f"L{i}" for i in range(12)]
    acts = wrapper.get_multi_layer_activations(layer_ids, input_ids)
    print("  All 12 layer activations (single forward pass):")
    for name, act in acts.items():
        print(f"    {name}: shape={act.shape}, mean={act.mean().item():.6f}")
    print()

    # 7. Parse LayerID for inspection
    lid = LayerID.parse("L5.attn")
    print(f"  LayerID 'L5.attn' -> encoder={lid.encoder_type}, "
          f"layer={lid.layer_num}, component={lid.component}")
    print()


def example_llava() -> None:
    """
    Example 2: LLaVA-1.5 (multimodal vision-language model)
    ========================================================
    Uses the built-in "llava" architecture config.
    Requires GPU with ≥16GB VRAM (~7B parameters).
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image
    import requests

    from captum._utils.transformer import LayerID
    from captum.attr import (
        LayerActivation,
        MultiModalAttribution,
        MultiModalModelWrapper,
    )

    print("=" * 60)
    print("Example 2: LLaVA-1.5 — Pretrained Vision-Language Model")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available. LLaVA requires a GPU with ≥16GB VRAM.")
        print("  Skipping this example.")
        print()
        return

    # 1. Load model and processor
    model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"  Loading {model_name} (this may take a few minutes)...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    # 2. Wrap with captum's MultiModalModelWrapper
    #    "llava" is a pre-registered architecture config that maps:
    #      V.L<i> -> model.vision_tower.vision_model.encoder.layers.<i>
    #      T.L<i> -> model.language_model.layers.<i>
    wrapper = MultiModalModelWrapper(model, "llava")

    # 3. Resolve layers from both vision and text encoders
    v_layer_0 = wrapper.resolve_layer("V.L0")
    v_layer_23 = wrapper.resolve_layer("V.L23")
    t_layer_0 = wrapper.resolve_layer("T.L0")
    t_layer_31 = wrapper.resolve_layer("T.L31")
    t_attn_15 = wrapper.resolve_layer("T.L15.attn")
    t_mlp_15 = wrapper.resolve_layer("T.L15.mlp")

    print(f"  V.L0:       {v_layer_0.__class__.__name__}")
    print(f"  V.L23:      {v_layer_23.__class__.__name__}")
    print(f"  T.L0:       {t_layer_0.__class__.__name__}")
    print(f"  T.L31:      {t_layer_31.__class__.__name__}")
    print(f"  T.L15.attn: {t_attn_15.__class__.__name__}")
    print(f"  T.L15.mlp:  {t_mlp_15.__class__.__name__}")
    print()

    # 4. Prepare multimodal input (image + text)
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    print(f"  Downloading image from: {image_url}")
    image = Image.open(requests.get(image_url, stream=True).raw)

    prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"  Prompt:            '{prompt}'")
    print(f"  input_ids shape:   {inputs['input_ids'].shape}")
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    print()

    # 5. Extract vision encoder activations
    print("  Extracting vision encoder activations...")
    v_acts = wrapper.get_multi_layer_activations(
        ["V.L0", "V.L12", "V.L23"],
        **inputs,
    )
    for name, act in v_acts.items():
        print(f"    {name}: shape={act.shape}, mean={act.float().mean().item():.6f}")
    print()

    # 6. Extract text decoder activations
    print("  Extracting text decoder activations...")
    t_acts = wrapper.get_multi_layer_activations(
        ["T.L0", "T.L15", "T.L31"],
        **inputs,
    )
    for name, act in t_acts.items():
        print(f"    {name}: shape={act.shape}, mean={act.float().mean().item():.6f}")
    print()

    # 7. Use LayerActivation via MultiModalAttribution
    print("  Computing LayerActivation for T.L0...")
    mm_attr = MultiModalAttribution(wrapper, LayerActivation, "T.L0")
    result = mm_attr.attribute(inputs["input_ids"], additional_forward_args=None)
    print(f"    T.L0 attribution shape: {result.shape}")
    print()

    # 8. LayerID parsing examples for LLaVA
    print("  LayerID examples for LLaVA:")
    for s in ["V.L0", "V.L23.attn", "T.L0", "T.L31.mlp", "T.L15.H0"]:
        lid = LayerID.parse(s)
        print(f"    '{s}' -> encoder={lid.encoder_type}, "
              f"layer={lid.layer_num}, head={lid.head_num}, "
              f"component={lid.component}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrained model attribution examples using captum"
    )
    parser.add_argument(
        "--gpt2", action="store_true",
        help="Run GPT-2 example (text-only, runs on CPU)"
    )
    parser.add_argument(
        "--llava", action="store_true",
        help="Run LLaVA example (vision-language, requires GPU)"
    )
    args = parser.parse_args()

    # Default: run GPT-2 if no flags specified
    if not args.gpt2 and not args.llava:
        args.gpt2 = True

    if args.gpt2:
        example_gpt2()

    if args.llava:
        example_llava()

    print("Done!")


if __name__ == "__main__":
    main()
