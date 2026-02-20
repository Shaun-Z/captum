#!/usr/bin/env python3

# pyre-strict

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TransformerArchConfig:
    r"""
    Architecture configuration that maps a standardized layer naming scheme
    to model-specific module paths. This enables the ``ActivationAccessor``
    and ``MultiModalAttribution`` classes to resolve layer identifiers such
    as ``"V.L1.attn"`` to concrete ``torch.nn.Module`` references within
    any supported transformer architecture.

    By defining a configuration for each model family, users only need to
    specify a model name and layer identifiers (e.g., ``"V.L1.H5"``),
    and the framework handles the underlying path resolution.

    Args:
        vision_encoder_prefix (str or None): Dot-separated module path to
                    the vision encoder's layer container. For example,
                    ``"vision_tower.vision_model.encoder.layers"`` for
                    LLaVA-style models. ``None`` if the model has no
                    vision encoder (e.g., a text-only LLM).
        text_encoder_prefix (str or None): Dot-separated module path to the
                    text encoder/decoder's layer container. For example,
                    ``"language_model.model.layers"`` for LLaVA-style or
                    ``"model.layers"`` for standalone LLMs. ``None`` if
                    the model has no text encoder (e.g., a pure ViT).
        default_encoder_prefix (str or None): Dot-separated module path to
                    use when no encoder type (``V`` or ``T``) is specified
                    in the layer identifier. Defaults to the text encoder
                    prefix if not explicitly set.
        attn_module_name (str): Name of the attention sub-module within
                    each transformer layer. Default: ``"self_attn"``.
        mlp_module_name (str): Name of the MLP / feed-forward sub-module
                    within each transformer layer. Default: ``"mlp"``.
        output_module_name (str): Name of the output / layer-norm
                    sub-module within each transformer layer.
                    Default: ``"layer_norm"``.
        num_attention_heads (int or None): Number of attention heads per
                    layer. Used to validate head indices in ``LayerID``.
                    ``None`` means no validation is performed.
        component_map (dict): Additional mapping from component short names
                    to actual sub-module names. For models with non-standard
                    naming, override specific components here.
    """

    vision_encoder_prefix: Optional[str] = None
    text_encoder_prefix: Optional[str] = None
    default_encoder_prefix: Optional[str] = None
    attn_module_name: str = "self_attn"
    mlp_module_name: str = "mlp"
    output_module_name: str = "layer_norm"
    num_attention_heads: Optional[int] = None
    component_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default_encoder_prefix is None:
            self.default_encoder_prefix = self.text_encoder_prefix

    def get_component_name(self, component: Optional[str]) -> Optional[str]:
        r"""
        Resolve a component short name to the actual sub-module name.

        Args:
            component (str or None): Component short name such as
                        ``'attn'``, ``'mlp'``, or ``'output'``.

        Returns:
            str or None: The resolved sub-module name, or ``None`` if
            ``component`` is ``None``.
        """
        if component is None:
            return None

        # Check custom component map first, then standard names
        if component in self.component_map:
            return self.component_map[component]

        standard_map = {
            "attn": self.attn_module_name,
            "mlp": self.mlp_module_name,
            "output": self.output_module_name,
        }
        if component in standard_map:
            return standard_map[component]

        # If the component doesn't match any known name, return it as-is
        # This allows direct sub-module names to be used
        return component

    def get_encoder_prefix(self, encoder_type: Optional[str]) -> Optional[str]:
        r"""
        Get the module path prefix for the specified encoder type.

        Args:
            encoder_type (str or None): ``'V'`` for vision, ``'T'`` for
                        text, or ``None`` for the default encoder.

        Returns:
            str or None: The dot-separated module path prefix.

        Raises:
            ValueError: If the requested encoder type is not available
                        in this configuration.
        """
        if encoder_type == "V":
            if self.vision_encoder_prefix is None:
                raise ValueError(
                    "Vision encoder prefix is not defined for this architecture."
                )
            return self.vision_encoder_prefix
        elif encoder_type == "T":
            if self.text_encoder_prefix is None:
                raise ValueError(
                    "Text encoder prefix is not defined for this architecture."
                )
            return self.text_encoder_prefix
        else:
            return self.default_encoder_prefix


def _make_arch_configs() -> Dict[str, TransformerArchConfig]:
    r"""
    Build the registry of known transformer architecture configurations.

    Returns:
        dict: A mapping from architecture name to ``TransformerArchConfig``.
    """
    configs: Dict[str, TransformerArchConfig] = {}

    # GPT-2 style models
    configs["gpt2"] = TransformerArchConfig(
        text_encoder_prefix="transformer.h",
        attn_module_name="attn",
        mlp_module_name="mlp",
        output_module_name="ln_2",
        num_attention_heads=12,
    )

    # LLaMA / Mistral style models
    configs["llama"] = TransformerArchConfig(
        text_encoder_prefix="model.layers",
        attn_module_name="self_attn",
        mlp_module_name="mlp",
        output_module_name="post_attention_layernorm",
        num_attention_heads=32,
    )

    # ViT (standalone vision transformer, e.g. from timm or torchvision)
    configs["vit"] = TransformerArchConfig(
        vision_encoder_prefix="encoder.layers",
        default_encoder_prefix="encoder.layers",
        attn_module_name="self_attention",
        mlp_module_name="mlp",
        output_module_name="ln_2",
        num_attention_heads=12,
    )

    # CLIP vision encoder
    configs["clip_vision"] = TransformerArchConfig(
        vision_encoder_prefix="vision_model.encoder.layers",
        default_encoder_prefix="vision_model.encoder.layers",
        attn_module_name="self_attn",
        mlp_module_name="mlp",
        output_module_name="layer_norm2",
        num_attention_heads=12,
    )

    # CLIP text encoder
    configs["clip_text"] = TransformerArchConfig(
        text_encoder_prefix="text_model.encoder.layers",
        default_encoder_prefix="text_model.encoder.layers",
        attn_module_name="self_attn",
        mlp_module_name="mlp",
        output_module_name="layer_norm2",
        num_attention_heads=12,
    )

    # LLaVA-style multimodal models (vision + language)
    # For HuggingFace LlavaForConditionalGeneration (transformers >= 4.46):
    #   model.vision_tower.vision_model.encoder.layers[i] — CLIP vision encoder
    #   model.language_model.model.layers[i]              — LLaMA text decoder
    configs["llava"] = TransformerArchConfig(
        vision_encoder_prefix=(
            "model.vision_tower.vision_model.encoder.layers"
        ),
        text_encoder_prefix="model.language_model.model.layers",
        attn_module_name="self_attn",
        mlp_module_name="mlp",
        output_module_name="layer_norm2",
        num_attention_heads=32,
    )

    # Qwen2-VL style multimodal models
    configs["qwen2_vl"] = TransformerArchConfig(
        vision_encoder_prefix="visual.blocks",
        text_encoder_prefix="model.layers",
        attn_module_name="attn",
        mlp_module_name="mlp",
        output_module_name="norm2",
        num_attention_heads=16,
    )

    return configs


ARCH_CONFIGS: Dict[str, TransformerArchConfig] = _make_arch_configs()
r"""
Registry of pre-defined architecture configurations for known transformer
model families. Keys are architecture names (e.g., ``"llama"``, ``"vit"``,
``"llava"``), and values are :class:`TransformerArchConfig` instances.

Users can add custom architectures by inserting new entries::

    from captum._utils.transformer import ARCH_CONFIGS, TransformerArchConfig
    ARCH_CONFIGS["my_model"] = TransformerArchConfig(
        text_encoder_prefix="backbone.layers",
        attn_module_name="attention",
    )
"""


def get_arch_config(name: str) -> TransformerArchConfig:
    r"""
    Retrieve an architecture configuration by name, returning a deep copy
    so that modifications do not affect the global registry.

    Args:
        name (str): Architecture name, e.g., ``"llama"``, ``"vit"``,
                    ``"llava"``.

    Returns:
        TransformerArchConfig: A copy of the requested configuration.

    Raises:
        ValueError: If the architecture name is not found in the registry.
    """
    if name not in ARCH_CONFIGS:
        raise ValueError(
            f"Unknown architecture: '{name}'. "
            f"Available architectures: {list(ARCH_CONFIGS.keys())}. "
            "You can register a custom architecture by adding to ARCH_CONFIGS "
            "or by passing a TransformerArchConfig directly."
        )
    return deepcopy(ARCH_CONFIGS[name])
