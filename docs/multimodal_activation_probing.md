---
id: multimodal_activation_probing
title: Multimodal Activation Probing
---

# Multimodal Activation Probing

Captum provides a unified interface for probing **any intermediate activation**
inside transformer models — including multimodal vision-language models — using
a compact string notation. This document explains the notation, the supported
architectures, and how to use the API.

## Layer-ID Notation

Every activation target is specified with a **Layer-ID string** that follows
the pattern:

```
[V|T.]L<layer_num>[.H<head_num>|.<component>[.<sub_path>]]
```

| Segment | Meaning | Example |
|---------|---------|---------|
| `V.` / `T.` | Encoder prefix — **V**ision or **T**ext. Omit for single-encoder models. | `V.L0`, `T.L3` |
| `L<num>` | Layer index (zero-based). | `L0`, `L11` |
| `.H<num>` | Attention head index. | `L5.H3` |
| `.<component>` | Named sub-module: `attn`, `mlp`, `output`. | `L0.attn`, `L0.mlp` |
| `.<sub_path>` | Dotted path into nested sub-modules. | `L0.mlp.c_fc`, `L0.attn.c_proj` |

### Examples

| Layer-ID | Resolves to |
|----------|-------------|
| `L0` | Full transformer block 0 |
| `L5.H3` | Attention head 3 in block 5 |
| `V.L1.attn` | Vision encoder, block 1, attention module |
| `T.L3.mlp` | Text encoder, block 3, MLP / FFN module |
| `L0.mlp.c_fc` | FFN input projection (e.g., `nn.Linear`) |
| `L0.mlp.act` | FFN activation function output (e.g., GELU) |
| `L0.mlp.c_proj` | FFN output projection |
| `L0.mlp.0` | First sub-module of a `nn.Sequential` MLP |

## Supported Architectures

Pre-registered configs are available for the following model families:

| Config Name | Vision Prefix | Text Prefix | Attn | MLP | Output |
|-------------|---------------|-------------|------|-----|--------|
| `gpt2` | — | `transformer.h` | `attn` | `mlp` | `ln_2` |
| `llama` | — | `model.layers` | `self_attn` | `mlp` | `post_attention_layernorm` |
| `vit` | `encoder.layers` | — | `self_attention` | `mlp` | `ln_2` |
| `clip_vision` | `vision_model.encoder.layers` | — | `self_attn` | `mlp` | `layer_norm2` |
| `clip_text` | — | `text_model.encoder.layers` | `self_attn` | `mlp` | `layer_norm2` |
| `llava` | `model.vision_tower.vision_model.encoder.layers` | `model.language_model.layers` | `self_attn` | `mlp` | `layer_norm2` |
| `qwen2_vl` | `visual.blocks` | `model.layers` | `attn` | `mlp` | `norm2` |

You can register custom architectures:

```python
from captum._utils.transformer import ARCH_CONFIGS, TransformerArchConfig

ARCH_CONFIGS["my_model"] = TransformerArchConfig(
    text_encoder_prefix="backbone.layers",
    attn_module_name="attention",
    mlp_module_name="feed_forward",
    output_module_name="norm",
    num_attention_heads=16,
)
```

## Core API

### ActivationAccessor

The low-level API for resolving modules and extracting activations.

```python
from captum._utils.transformer import ActivationAccessor

accessor = ActivationAccessor(model, "gpt2")

# Resolve a layer-ID to its torch.nn.Module
module = accessor.resolve_module("L0.mlp.act")

# Extract activation (output) from a single layer
act = accessor.get_activation("L0.mlp.act", input_ids)

# Extract activation input instead of output
act_in = accessor.get_activation("L0.mlp.act", input_ids,
                                  attribute_to_layer_input=True)

# Extract activations from multiple layers in ONE forward pass
acts = accessor.get_multi_layer_activations(
    ["L0", "L0.attn", "L0.mlp", "L0.mlp.c_fc", "L0.mlp.act"],
    input_ids,
)
```

### MultiModalModelWrapper

A higher-level wrapper that adds layer resolution to any model.

```python
from captum.attr import MultiModalModelWrapper

wrapper = MultiModalModelWrapper(model, "llava")

# Resolve layers
v_layer = wrapper.resolve_layer("V.L0")
t_layer = wrapper.resolve_layer("T.L15.attn")

# Extract activations
act = wrapper.get_activation("V.L0.mlp.act", **inputs)
```

### MultiModalAttribution

Combines the wrapper with any `LayerAttribution` method from Captum.

```python
from captum.attr import MultiModalAttribution, LayerActivation

mm_attr = MultiModalAttribution(wrapper, LayerActivation, "T.L0")
result = mm_attr.attribute(input_ids)
```

## Probing FFN Internals

A standard transformer FFN / MLP block typically has:

```
MLP
├── Linear (input projection, d → 4d)
├── Activation (GELU / ReLU / SiLU)
└── Linear (output projection, 4d → d)
```

The dotted sub-path notation lets you target **any** of these:

```python
# GPT-2 MLP: mlp.c_fc → mlp.act → mlp.c_proj
accessor.get_activation("L0.mlp.c_fc", input_ids)    # Linear output (act fn input)
accessor.get_activation("L0.mlp.act", input_ids)      # GELU output (act fn output)
accessor.get_activation("L0.mlp.c_proj", input_ids)   # Projection output

# For Sequential-style MLP (e.g., nn.Sequential(Linear, GELU, Linear)):
accessor.get_activation("L0.mlp.0", input_ids)   # Linear output
accessor.get_activation("L0.mlp.1", input_ids)   # GELU output
accessor.get_activation("L0.mlp.2", input_ids)   # Projection output
```

## Probing Attention Internals

Similarly, you can probe inside the attention sub-module:

```python
# GPT-2 Attention: attn.c_attn (Q/K/V) → attn.c_proj (output projection)
accessor.get_activation("L0.attn.c_attn", input_ids)   # Combined Q/K/V projection
accessor.get_activation("L0.attn.c_proj", input_ids)    # Output projection

# LLaMA Attention: self_attn.q_proj, self_attn.k_proj, ...
accessor.get_activation("L0.attn.q_proj", input_ids)    # Query projection
accessor.get_activation("L0.attn.v_proj", input_ids)    # Value projection
```

## Multimodal Models

For multimodal models (e.g., LLaVA, CLIP), prefix with `V.` or `T.`:

```python
wrapper = MultiModalModelWrapper(model, "llava")

# Vision encoder activations
wrapper.get_activation("V.L0.mlp", **inputs)
wrapper.get_activation("V.L12.attn", **inputs)

# Text decoder activations
wrapper.get_activation("T.L0.mlp.act", **inputs)
wrapper.get_activation("T.L31.attn", **inputs)

# Extract from both encoders in one forward pass
acts = wrapper.get_multi_layer_activations(
    ["V.L0", "V.L23", "T.L0", "T.L15", "T.L31"],
    **inputs,
)
```

## Visualization

The `visualize_activations` utility provides heatmap-based visualization of
probed activations, similar to the existing `visualize_image_attr` in Captum.

```python
from captum._utils.transformer.visualization import visualize_activations

# Extract activations
acts = accessor.get_multi_layer_activations(
    ["L0.mlp.c_fc", "L0.mlp.act", "L0.mlp.c_proj"],
    input_ids,
)

# Visualize as heatmaps
fig = visualize_activations(
    acts,
    tokens=tokenizer.convert_ids_to_tokens(input_ids[0]),
    figsize=(14, 8),
)
fig.savefig("activations.png")
```

See `examples/probe_all_activations_example.py` and
`examples/visualize_activations_example.py` for runnable examples.

## Complete Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum._utils.transformer import ActivationAccessor

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

accessor = ActivationAccessor(model, "gpt2")
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

# Probe every intermediate activation in block 0
probe_ids = [
    "L0",             # Full block
    "L0.attn",        # Attention
    "L0.attn.c_attn", # Q/K/V projection
    "L0.attn.c_proj", # Attention output projection
    "L0.mlp",         # Full MLP
    "L0.mlp.c_fc",    # FFN input projection
    "L0.mlp.act",     # GELU activation
    "L0.mlp.c_proj",  # FFN output projection
    "L0.output",      # LayerNorm
]

activations = accessor.get_multi_layer_activations(probe_ids, input_ids)

for name, act in activations.items():
    print(f"{name:20s}  shape={act.shape}  mean={act.mean():.6f}")
```

---

## Bi-Modal / Encoder-Decoder Relevance Propagation

Beyond activation probing, Captum provides **gradient-weighted attention
relevance propagation** for explaining predictions made by multimodal and
encoder-decoder transformers.  The implementation follows the algorithm from:

> Chefer, H., Gur, S., & Wolf, L. (2021). *Generic Attention-model
> Explainability for Interpreting Bi-Modal and Encoder-Decoder
> Transformers*. ICCV 2021.

### How it works

The method maintains **relevance matrices** that track how every token in
each modality contributes to every other token.  Starting from an identity
matrix (each token only relevant to itself), it propagates relevance
through gradient-weighted attention at each transformer layer.

The key equations are:

| Equation | Name | Purpose |
|----------|------|---------|
| Eq. 5 | `avg_heads` | Gradient-weighted, positive-only head averaging |
| Eq. 6 | `apply_self_attention_rules` (self) | Self-attention update for self-relevance |
| Eq. 7 | `apply_self_attention_rules` (cross) | Self-attention update for cross-relevance |
| Eq. 8–9 | `normalize_self_relevance` | Residual normalization |
| Eq. 10 | `apply_mm_attention_rules` | Cross-attention relevance propagation |
| Eq. 11 | `apply_mm_attention_rules` (Eq. 11) | Feedback from reciprocal cross-memory |

### Supported architecture patterns

| Template | Architecture | Relevance matrices | Example models |
|----------|-------------|-------------------|----------------|
| **A** — Single-stream | ViT, GPT-2 | `R` (n×n) | VisualBERT (concat) |
| **B** — Co-attention | LXMERT, ViLBERT | `R_tt, R_ii, R_ti, R_it` | LXMERT |
| **C** — Encoder-decoder | DETR | `R_ee, R_qq, R_qe` | DETR |

### Relevance Propagation API

```python
from captum.attr import MultiModalModelWrapper
```

#### Template A — Single-stream (ViT / GPT-2)

```python
explainer = MultiModalModelWrapper(model, "gpt2", mode="single_stream")

# Generate per-token relevance matrix
R = explainer.generate_relevance(input_ids, target_index=class_idx)

# Extract CLS-token relevance (row 0)
cls_relevance = MultiModalModelWrapper.readout(R, readout_index=0)
# cls_relevance[i] = how much token i contributes to the CLS decision
```

#### Template B — Co-attention (LXMERT / ViLBERT)

```python
explainer = MultiModalModelWrapper(model, arch_config, mode="co_attention")

result = explainer.generate_co_attention(
    input_ids, pixel_values,
    target_index=answer_idx,
    text_self_layer_ids=["T.L0", "T.L1", ...],
    image_self_layer_ids=["V.L0", "V.L1", ...],
    cross_layer_ids=[("T.L5", "V.L5"), ...],  # (text←image, image←text)
)

# result["R_tt"] — text-to-text relevance
# result["R_ti"] — text-to-image relevance  (most useful for VQA)
# result["R_ii"] — image-to-image relevance
# result["R_it"] — image-to-text relevance

# Readout for CLS token
readout = MultiModalModelWrapper.readout(result, readout_index=0, modality="text")
text_relevance  = readout["text"]   # contribution of each text token
image_relevance = readout["image"]  # contribution of each image token
```

#### Template C — Encoder-decoder (DETR)

```python
explainer = MultiModalModelWrapper(model, arch_config, mode="encoder_decoder")

result = explainer.generate_encoder_decoder(
    pixel_values,
    target_fn=lambda out: out[0, query_idx, class_idx],
    encoder_self_layer_ids=["V.L0", "V.L1", ...],
    decoder_self_layer_ids=["T.L0", "T.L1", ...],
    decoder_cross_layer_ids=["T.L0", "T.L1", ...],
)

# result["R_qe"][j, :] — relevance of each encoder token for query j
readout = MultiModalModelWrapper.readout(
    result, readout_index=query_idx, modality="query"
)
encoder_relevance = readout["encoder"]
```

### Low-level utilities

The building-block functions are also available for custom pipelines:

```python
from captum.attr import (
    avg_heads,
    apply_self_attention_rules,
    normalize_self_relevance,
    apply_mm_attention_rules,
)

# Eq. 5: gradient-weighted head averaging
A_bar = avg_heads(attn_weights, attn_grads)  # (q, k)

# Eq. 6-7: self-attention update
R_ss, R_sq = apply_self_attention_rules(R_ss, R_sq, A_bar)

# Eq. 8-9: residual normalization
R_bar = normalize_self_relevance(R_ss)

# Eq. 10-11: cross-attention update
R_ss, R_sq = apply_mm_attention_rules(
    R_ss, R_qq, R_sq, R_qs, A_sq_bar, use_eq11=True
)
```

### AttentionHookManager

To capture attention weights and their gradients during forward/backward
passes, use `AttentionHookManager`:

```python
from captum.attr import AttentionHookManager

# Hook into attention modules
manager = AttentionHookManager(model, [model.layer[i].attn for i in range(12)])

# Forward + backward
output = model(input_ids)
loss = output[0, target_class]
loss.backward()

# Retrieve captured data
for i in range(12):
    attn, grad = manager.get_attention_and_gradient(i)
    A_bar = avg_heads(attn[0], grad[0])
    # ... propagate relevance ...

manager.remove_hooks()
```

### Complete Example

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import MultiModalModelWrapper

model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    attn_implementation="eager",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

from captum._utils.transformer import TransformerArchConfig

bert_config = TransformerArchConfig(
    text_encoder_prefix="bert.encoder.layer",
    attn_module_name="attention",
    mlp_module_name="intermediate",
    output_module_name="output",
    num_attention_heads=12,
)

explainer = MultiModalModelWrapper(model, bert_config, mode="single_stream")

text = "This movie was absolutely wonderful"
inputs = tokenizer(text, return_tensors="pt")

R = explainer.generate_single_stream(
    **inputs,
    target_index=1,        # positive sentiment class
    num_layers=12,
)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
cls_relevance = MultiModalModelWrapper.readout(R, readout_index=0)

for tok, rel in zip(tokens, cls_relevance):
    print(f"{tok:15s}  relevance={rel:.4f}")
```