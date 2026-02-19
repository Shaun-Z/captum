#!/usr/bin/env python3

# pyre-strict

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
from captum.attr._core.multimodal_attr import MultiModalAttribution, MultiModalModelWrapper
from captum.testing.helpers import BaseTest
from parameterized import parameterized, parameterized_class
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Dummy models for testing (no external model downloads required)
# ---------------------------------------------------------------------------


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


# Architecture configs for the dummy models
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


# ---------------------------------------------------------------------------
# Tests for LayerID
# ---------------------------------------------------------------------------


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
            ("V.L1.H5.extra",),  # too many parts
        ]
    )
    def test_parse_invalid(self, layer_str: str) -> None:
        with self.assertRaises(ValueError):
            LayerID.parse(layer_str)

    def test_str_roundtrip(self) -> None:
        """Test that str(LayerID.parse(s)) produces a valid re-parseable string."""
        test_cases = ["L0", "L5", "V.L1", "T.L3", "L1.H5", "V.L1.H5", "L0.attn"]
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


# ---------------------------------------------------------------------------
# Tests for TransformerArchConfig
# ---------------------------------------------------------------------------


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

    def test_add_custom_to_registry(self) -> None:
        """Test adding a custom architecture to the global registry."""
        ARCH_CONFIGS["test_custom"] = TransformerArchConfig(
            text_encoder_prefix="custom.layers"
        )
        config = get_arch_config("test_custom")
        self.assertEqual(config.text_encoder_prefix, "custom.layers")
        # Cleanup
        del ARCH_CONFIGS["test_custom"]


# ---------------------------------------------------------------------------
# Tests for ActivationAccessor
# ---------------------------------------------------------------------------


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

    def test_resolve_invalid_layer_raises(self) -> None:
        model = DummyTextEncoder(num_layers=2).to(self.device)
        accessor = ActivationAccessor(model, DUMMY_TEXT_CONFIG)
        with self.assertRaises((IndexError, AttributeError)):
            accessor.resolve_module("L5")  # Only 2 layers


# ---------------------------------------------------------------------------
# Tests for MultiModalModelWrapper
# ---------------------------------------------------------------------------


@parameterized_class(
    ("device",),
    [("cpu",), ("cuda",)] if torch.cuda.is_available() else [("cpu",)],
)
class TestMultiModalModelWrapper(BaseTest):
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


# ---------------------------------------------------------------------------
# Tests for MultiModalAttribution
# ---------------------------------------------------------------------------


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
