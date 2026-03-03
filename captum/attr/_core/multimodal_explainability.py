#!/usr/bin/env python3

# pyre-strict

r"""
Multimodal Explainability: unified wrapper, attribution, and relevance
propagation for transformer-based models.

This module provides:

1. **MultiModalModelWrapper** — unified wrapper for LLM, ViT, and LVLM
   architectures with layer resolution, activation extraction, and
   gradient-weighted attention relevance propagation (Chefer et al.,
   ICCV 2021).

2. **MultiModalAttribution** — ``Attribution`` subclass that delegates
   to captum's ``LayerAttribution`` methods via compact layer IDs.

3. **Relevance utilities** — low-level math helpers (Eq. 5–11) and an
   ``AttentionHookManager`` for capturing attention weights/gradients.

Reference:
    Chefer, H., Gur, S., & Wolf, L. (2021). *Generic Attention-model
    Explainability for Interpreting Bi-Modal and Encoder-Decoder
    Transformers*. ICCV 2021.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from captum._utils.transformer.accessor import ActivationAccessor
from captum._utils.transformer.arch_config import (
    get_arch_config,
    TransformerArchConfig,
)
from captum._utils.transformer.layer_id import LayerID
from captum.attr._core.layer.layer_activation import LayerActivation
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_gradient_shap import LayerGradientShap
from captum.attr._core.layer.layer_gradient_x_activation import (
    LayerGradientXActivation,
)
from captum.attr._core.layer.layer_integrated_gradients import (
    LayerIntegratedGradients,
)
from captum.attr._utils.attribution import Attribution, LayerAttribution
from captum.log import log_usage
from torch import nn, Tensor


# ────────────────────────────────────────────────────────────────────
# Enum for architecture type selection
# ────────────────────────────────────────────────────────────────────


class ExplainabilityMode(str, Enum):
    """Architecture type for relevance propagation."""

    SINGLE_STREAM = "single_stream"
    CO_ATTENTION = "co_attention"
    ENCODER_DECODER = "encoder_decoder"


# ────────────────────────────────────────────────────────────────────
# Low-level math helpers  (Equations 5-11)
# ────────────────────────────────────────────────────────────────────


def avg_heads(
    attn: Tensor,
    grad: Tensor,
) -> Tensor:
    r"""
    Gradient-weighted head-averaged attention (Eq. 5).

    Computes the positive part of the element-wise product of the
    attention weights and their gradients, then averages across heads.

    .. math::
        \bar A = \mathbb{E}_h\bigl[(\nabla A_h \odot A_h)^+\bigr]

    Args:
        attn (Tensor): Post-softmax attention probabilities with shape
            ``(batch, heads, query_tokens, key_tokens)`` or
            ``(heads, query_tokens, key_tokens)``.
        grad (Tensor): Gradient of the target score with respect to
            ``attn``, same shape as ``attn``.

    Returns:
        Tensor: Head-averaged, gradient-weighted attention with shape
        ``(batch, query_tokens, key_tokens)`` or
        ``(query_tokens, key_tokens)``.
    """
    # Element-wise product, keep only positive contributions
    cam = torch.clamp(grad * attn, min=0.0)
    # Average over the head dimension (dim -3)
    return cam.mean(dim=-3)


def apply_self_attention_rules(
    R_ss: Tensor,
    R_sq: Optional[Tensor],
    A_bar: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Self-attention relevance update (Eq. 6 and Eq. 7).

    For self-attention inside modality *s*:

    .. math::
        R_{ss} &\leftarrow R_{ss} + \bar A_{ss}\, R_{ss}  \quad (\text{Eq. 6})\\
        R_{sq} &\leftarrow R_{sq} + \bar A_{ss}\, R_{sq}  \quad (\text{Eq. 7})

    Args:
        R_ss (Tensor): Self-relevance matrix for modality *s*,
            shape ``(batch, s, s)`` or ``(s, s)``.
        R_sq (Tensor or None): Cross-relevance from *s* to *q*,
            shape ``(batch, s, q)`` or ``(s, q)``. ``None`` if there
            is no cross-modal relevance to propagate.
        A_bar (Tensor): Gradient-weighted attention map from
            :func:`avg_heads`, shape matching ``R_ss``.

    Returns:
        tuple: ``(R_ss, R_sq)`` — updated relevance matrices.
    """
    R_ss = R_ss + A_bar @ R_ss
    if R_sq is not None:
        R_sq = R_sq + A_bar @ R_sq
    return R_ss, R_sq


def normalize_self_relevance(R_xx: Tensor, eps: float = 1e-9) -> Tensor:
    r"""
    Residual normalization of a self-relevance matrix (Eq. 8–9).

    Subtracts the identity, row-normalizes, then adds identity back:

    .. math::
        \hat R_{xx} &= R_{xx} - I  \\
        \bar R_{xx} &= \text{RowNorm}(\hat R_{xx}) + I

    Args:
        R_xx (Tensor): Self-relevance matrix, shape ``(..., n, n)``.
        eps (float): Epsilon for numerical stability.

    Returns:
        Tensor: Normalized self-relevance, same shape.
    """
    n = R_xx.shape[-1]
    device = R_xx.device
    eye = torch.eye(n, device=device, dtype=R_xx.dtype)
    # Broadcast for batch dim
    if R_xx.dim() == 3:
        eye = eye.unsqueeze(0)
    R_hat = R_xx - eye
    # Row-wise normalize (L1 norm per row, keep dim for broadcasting)
    row_sums = R_hat.abs().sum(dim=-1, keepdim=True).clamp(min=eps)
    R_hat = R_hat / row_sums
    return R_hat + eye


def apply_mm_attention_rules(
    R_ss: Tensor,
    R_qq: Tensor,
    R_sq: Tensor,
    R_qs: Optional[Tensor],
    A_sq_bar: Tensor,
    use_eq11: bool = True,
    eps: float = 1e-9,
) -> Tuple[Tensor, Tensor]:
    r"""
    Cross-attention relevance update (Eq. 10 and optional Eq. 11).

    For cross-attention from query modality *s* to context modality *q*:

    .. math::
        R_{sq} &\leftarrow R_{sq}
            + \bar R_{ss}^{\!\top}\,\bar A_{sq}\,\bar R_{qq}
            \quad (\text{Eq. 10})\\
        R_{ss} &\leftarrow R_{ss}
            + \bar A_{sq}\, R_{qs}
            \quad (\text{Eq. 11, optional})

    Eq. 11 is only applied when ``R_qs`` is not ``None`` and
    ``use_eq11=True``.  Skip it for encoder-decoder architectures where
    the encoder is not conditioned on the decoder.

    Args:
        R_ss (Tensor): Self-relevance for modality *s*.
        R_qq (Tensor): Self-relevance for modality *q*.
        R_sq (Tensor): Cross-relevance from *s* to *q*.
        R_qs (Tensor or None): Cross-relevance from *q* to *s*.
        A_sq_bar (Tensor): Gradient-weighted cross-attention,
            shape ``(batch, s_tokens, q_tokens)`` or
            ``(s_tokens, q_tokens)``.
        use_eq11 (bool): Whether to apply Eq. 11.
        eps (float): Epsilon for normalization.

    Returns:
        tuple: ``(R_ss, R_sq)`` — updated relevance matrices.
    """
    R_ss_bar = normalize_self_relevance(R_ss, eps=eps)
    R_qq_bar = normalize_self_relevance(R_qq, eps=eps)

    # Eq. 10: R_sq += R_ss_bar^T @ A_sq_bar @ R_qq_bar
    # Transpose R_ss_bar on last two dims
    R_sq = R_sq + R_ss_bar.transpose(-2, -1) @ A_sq_bar @ R_qq_bar

    # Eq. 11: R_ss += A_sq_bar @ R_qs
    if use_eq11 and R_qs is not None:
        R_ss = R_ss + A_sq_bar @ R_qs

    return R_ss, R_sq


# ────────────────────────────────────────────────────────────────────
# Hook manager for capturing attention weights and gradients
# ────────────────────────────────────────────────────────────────────


class AttentionHookManager:
    r"""
    Registers forward and backward hooks on attention modules to capture
    post-softmax attention probabilities and their gradients.

    Each attention module in a transformer typically returns a tuple
    ``(output, attention_weights)`` when ``output_attentions=True``.
    This manager installs hooks that record both the forward attention
    weights and their backward gradients for use in relevance
    propagation.

    Args:
        model (torch.nn.Module): The model to instrument.
        attn_modules (list of torch.nn.Module): Attention sub-modules
            to hook into.
        attn_output_index (int): Index of the attention-weight tensor
            in the attention module's output tuple. Most HuggingFace
            modules return ``(hidden_states, attn_weights, ...)`` so
            the default is ``1``.

    Example::

        >>> manager = AttentionHookManager(model, [model.layer[0].attn])
        >>> output = model(input_ids)
        >>> loss = output.logits[0, target_class]
        >>> loss.backward()
        >>> attn_probs = manager.get_attention("layer0_attn")
        >>> attn_grads = manager.get_gradient("layer0_attn")
        >>> manager.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        attn_modules: List[nn.Module],
        attn_output_index: int = 1,
    ) -> None:
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._attention_maps: Dict[str, Tensor] = {}
        self._attention_grads: Dict[str, Tensor] = {}

        for idx, module in enumerate(attn_modules):
            key = f"attn_{idx}"
            self._register_hooks(key, module, attn_output_index)

    def _register_hooks(
        self,
        key: str,
        module: nn.Module,
        attn_output_index: int,
    ) -> None:
        """Register forward and backward hooks on an attention module."""

        def forward_hook(
            mod: nn.Module,
            inp: Any,
            output: Any,
        ) -> None:
            if isinstance(output, tuple) and len(output) > attn_output_index:
                attn_weights = output[attn_output_index]
                if isinstance(attn_weights, Tensor):
                    self._attention_maps[key] = attn_weights

                    # Register gradient hook on the attention weights
                    if attn_weights.requires_grad:

                        def grad_hook(grad: Tensor) -> None:
                            self._attention_grads[key] = grad

                        attn_weights.register_hook(grad_hook)

        handle = module.register_forward_hook(forward_hook)
        self._handles.append(handle)

    def get_attention(self, idx: int) -> Tensor:
        """Return captured attention weights for the given module index."""
        key = f"attn_{idx}"
        if key not in self._attention_maps:
            raise RuntimeError(
                f"No attention weights captured for index {idx}. "
                "Ensure the model was called with output_attentions=True "
                "and the attention implementation is 'eager'."
            )
        return self._attention_maps[key]

    def get_gradient(self, idx: int) -> Tensor:
        """Return captured attention gradient for the given module index."""
        key = f"attn_{idx}"
        if key not in self._attention_grads:
            raise RuntimeError(
                f"No attention gradient captured for index {idx}. "
                "Ensure backward() was called before accessing gradients."
            )
        return self._attention_grads[key]

    def get_attention_and_gradient(
        self, idx: int
    ) -> Tuple[Tensor, Tensor]:
        """Return both attention and gradient for the given module index."""
        return self.get_attention(idx), self.get_gradient(idx)

    @property
    def num_modules(self) -> int:
        """Number of hooked attention modules."""
        return len(self._handles)

    def clear(self) -> None:
        """Clear all captured attention maps and gradients."""
        self._attention_maps.clear()
        self._attention_grads.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks and clear cached data."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def __del__(self) -> None:
        self.remove_hooks()


# ────────────────────────────────────────────────────────────────────
# MultiModalModelWrapper
# ────────────────────────────────────────────────────────────────────


class MultiModalModelWrapper:
    r"""
    Unified wrapper for LLM, ViT, and LVLM (Large Vision-Language Model)
    architectures. Provides a consistent interface for resolving layer
    identifiers to ``torch.nn.Module`` objects, extracting activations,
    and generating gradient-weighted attention relevance maps.

    This wrapper supports a compact layer notation:

    - ``"L<num>"`` — layer in the default encoder
    - ``"V.L<num>"`` — layer in the vision encoder
    - ``"T.L<num>"`` — layer in the text encoder
    - Append ``.H<num>`` for a specific attention head (e.g., ``"V.L1.H5"``)
    - Append ``.<component>`` for sub-modules (e.g., ``"T.L3.attn"``,
      ``"T.L3.mlp"``)

    When ``mode`` is specified, the wrapper also provides relevance
    propagation through gradient-weighted attention (Chefer et al.,
    ICCV 2021) via the :meth:`generate_relevance` family of methods.

    Args:
        model (torch.nn.Module): The transformer model to wrap.
        arch_config (str or TransformerArchConfig): Either a registered
                    architecture name (e.g., ``"llama"``, ``"vit"``,
                    ``"llava"``) or a custom ``TransformerArchConfig``
                    instance.
        device (str or torch.device or None): Device for the model and
                    extracted activations. If ``None``, the model's current
                    device is used.
        mode (str, ExplainabilityMode, or None): Architecture type for
                    relevance propagation. One of:

                    - ``"single_stream"`` — ViT / GPT-2 / single-encoder.
                    - ``"co_attention"``  — LXMERT / ViLBERT co-attention.
                    - ``"encoder_decoder"`` — DETR-style encoder-decoder.
                    - ``None`` — relevance methods are not available.

    Examples::

        >>> from captum.attr import MultiModalModelWrapper
        >>> wrapper = MultiModalModelWrapper(model, "llava")
        >>> attn_layer = wrapper.resolve_layer("V.L1.attn")

        >>> # With relevance propagation
        >>> wrapper = MultiModalModelWrapper(
        ...     model, config, mode="single_stream"
        ... )
        >>> R = wrapper.generate_relevance(
        ...     input_ids, target_index=class_idx
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        arch_config: Union[str, TransformerArchConfig],
        device: Union[str, torch.device, None] = None,
        mode: Union[str, ExplainabilityMode, None] = None,
    ) -> None:
        self.model = model
        if isinstance(arch_config, str):
            self.arch_config = get_arch_config(arch_config)
        else:
            self.arch_config = arch_config
        self.device = device
        self.accessor = ActivationAccessor(model, self.arch_config, device)
        self.mode: Optional[ExplainabilityMode] = (
            ExplainabilityMode(mode)
            if isinstance(mode, str)
            else mode
        )

    def resolve_layer(
        self,
        layer_id: Union[str, LayerID],
    ) -> nn.Module:
        r"""
        Resolve a layer identifier to the corresponding ``torch.nn.Module``.

        Args:
            layer_id (str or LayerID): Compact layer identifier string
                        (e.g., ``"V.L1.attn"``) or a ``LayerID`` instance.

        Returns:
            torch.nn.Module: The resolved module within the wrapped model.

        Raises:
            ValueError: If the layer identifier cannot be resolved.
            AttributeError: If the resolved module path does not exist.
        """
        return self.accessor.resolve_module(layer_id)

    def resolve_layers(
        self,
        layer_ids: List[Union[str, LayerID]],
    ) -> List[nn.Module]:
        r"""
        Resolve multiple layer identifiers to their corresponding modules.

        Args:
            layer_ids (list of str or LayerID): Layer identifiers to
                        resolve.

        Returns:
            list of torch.nn.Module: Resolved modules in the same order.
        """
        return self.accessor.resolve_layers(layer_ids)

    def get_activation(
        self,
        layer_id: Union[str, LayerID],
        # pyre-fixme[2]: Parameter must be annotated.
        *model_args: Any,
        attribute_to_layer_input: bool = False,
        # pyre-fixme[2]: Parameter must be annotated.
        **model_kwargs: Any,
    ) -> Tensor:
        r"""
        Extract the activation at a specific layer during a forward pass.

        Args:
            layer_id (str or LayerID): Target layer identifier.
            *model_args: Positional arguments for the model's forward.
            attribute_to_layer_input (bool, optional): If ``True``, captures
                        the layer's input instead of output.
                        Default: ``False``.
            **model_kwargs: Keyword arguments for the model's forward.

        Returns:
            Tensor: The captured activation tensor.
        """
        return self.accessor.get_activation(
            layer_id,
            *model_args,
            attribute_to_layer_input=attribute_to_layer_input,
            **model_kwargs,
        )

    def get_multi_layer_activations(
        self,
        layer_ids: List[Union[str, LayerID]],
        # pyre-fixme[2]: Parameter must be annotated.
        *model_args: Any,
        attribute_to_layer_input: bool = False,
        # pyre-fixme[2]: Parameter must be annotated.
        **model_kwargs: Any,
    ) -> Dict[str, Tensor]:
        r"""
        Extract activations from multiple layers in a single forward pass.

        Args:
            layer_ids (list of str or LayerID): Target layer identifiers.
            *model_args: Positional arguments for the model's forward.
            attribute_to_layer_input (bool, optional): If ``True``, captures
                        each layer's input instead of output.
                        Default: ``False``.
            **model_kwargs: Keyword arguments for the model's forward.

        Returns:
            dict: Mapping from layer identifier strings to tensors.
        """
        return self.accessor.get_multi_layer_activations(
            layer_ids,
            *model_args,
            attribute_to_layer_input=attribute_to_layer_input,
            **model_kwargs,
        )

    # ----------------------------------------------------------------
    # Relevance propagation helpers
    # ----------------------------------------------------------------

    def _resolve_attn_modules(
        self,
        layer_ids: List[str],
    ) -> List[nn.Module]:
        """Resolve attention sub-modules for the given layer IDs."""
        attn_component = self.arch_config.attn_module_name
        modules = []
        for lid_str in layer_ids:
            lid = LayerID.parse(lid_str)
            prefix = self.arch_config.get_encoder_prefix(lid.encoder_type)
            if prefix is None:
                raise ValueError(
                    f"Cannot resolve encoder prefix for layer '{lid_str}'."
                )
            attn_path = f"{prefix}.{lid.layer_num}.{attn_component}"
            module = self.accessor._get_module_by_path(attn_path)
            modules.append(module)
        return modules

    def _count_layers(self, encoder_type: Optional[str] = None) -> int:
        """Count the number of transformer layers for the given encoder."""
        prefix = self.arch_config.get_encoder_prefix(encoder_type)
        if prefix is None:
            raise ValueError(
                f"No encoder prefix for encoder_type='{encoder_type}'."
            )
        container = self.accessor._get_module_by_path(prefix)
        return len(list(container.children()))

    def _build_target_score(
        self,
        outputs: Any,
        target_index: Optional[Union[int, Tuple[int, ...]]] = None,
        target_fn: Optional[Callable[..., Tensor]] = None,
    ) -> Tensor:
        """Extract a scalar target score from model outputs."""
        if target_fn is not None:
            return target_fn(outputs)
        if isinstance(outputs, Tensor):
            logits = outputs
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            raise ValueError(
                "Cannot extract logits from model output. Provide a "
                "'target_fn' callable or ensure model returns a Tensor "
                "or an object with a 'logits' attribute."
            )
        if target_index is not None:
            if isinstance(target_index, int):
                return logits[0, target_index]
            else:
                return logits[(0,) + target_index]
        return logits[0].max()

    # ----------------------------------------------------------------
    # Template A: Single-stream (ViT / GPT-2)
    # ----------------------------------------------------------------

    @log_usage(part_of_slo=True)
    def generate_single_stream(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        *model_args: Any,
        target_index: Optional[Union[int, Tuple[int, ...]]] = None,
        target_fn: Optional[Callable[..., Tensor]] = None,
        num_layers: Optional[int] = None,
        encoder_type: Optional[str] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        **model_kwargs: Any,
    ) -> Tensor:
        r"""
        Generate per-token relevance for a single-stream transformer
        (Template A — ViT / GPT-2 style).

        Maintains a single relevance matrix
        :math:`R \in \mathbb{R}^{n \times n}` and applies the
        self-attention update (Eq. 6) at each layer.

        Args:
            *model_args: Positional arguments for the model's forward.
            target_index (int, tuple of int, or None): Index into the
                model output to select the scalar target score.
            target_fn (callable or None): Custom function that takes model
                outputs and returns a scalar tensor.
            num_layers (int or None): Number of layers to propagate
                through. If ``None``, auto-detect from the model.
            encoder_type (str or None): Encoder type to use.
            **model_kwargs: Keyword arguments for the model's forward.

        Returns:
            Tensor: Relevance matrix of shape ``(n, n)`` where ``n`` is
            the number of tokens.
        """
        if num_layers is None:
            num_layers = self._count_layers(encoder_type)

        prefix = "" if encoder_type is None else f"{encoder_type}."
        layer_ids = [f"{prefix}L{i}" for i in range(num_layers)]
        attn_modules = self._resolve_attn_modules(layer_ids)

        hook_mgr = AttentionHookManager(self.model, attn_modules)

        try:
            self.model.zero_grad()
            outputs = self.model(*model_args, **model_kwargs)
            target_score = self._build_target_score(
                outputs, target_index, target_fn
            )
            target_score.backward(retain_graph=True)

            first_attn = hook_mgr.get_attention(0)
            n_tokens = first_attn.shape[-1]
            device = first_attn.device
            dtype = first_attn.dtype
            R = torch.eye(n_tokens, device=device, dtype=dtype)

            for i in range(num_layers):
                attn, grad = hook_mgr.get_attention_and_gradient(i)
                if attn.dim() == 4:
                    attn = attn[0]
                    grad = grad[0]
                A_bar = avg_heads(attn, grad)
                R = R + A_bar @ R

        finally:
            hook_mgr.remove_hooks()

        if self.device is not None:
            R = R.to(self.device)
        return R

    # ----------------------------------------------------------------
    # Template B: Co-attention (LXMERT / ViLBERT)
    # ----------------------------------------------------------------

    @log_usage(part_of_slo=True)
    def generate_co_attention(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        *model_args: Any,
        target_index: Optional[Union[int, Tuple[int, ...]]] = None,
        target_fn: Optional[Callable[..., Tensor]] = None,
        text_self_layer_ids: Optional[List[str]] = None,
        image_self_layer_ids: Optional[List[str]] = None,
        cross_layer_ids: Optional[List[Tuple[str, str]]] = None,
        text_token_count: Optional[int] = None,
        image_token_count: Optional[int] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        **model_kwargs: Any,
    ) -> Dict[str, Tensor]:
        r"""
        Generate per-token relevance for a co-attention bi-modal
        transformer (Template B — LXMERT / ViLBERT style).

        Maintains four relevance matrices:
        :math:`R_{tt}, R_{ii}, R_{ti}, R_{it}` and applies self-attention
        updates (Eq. 6–7) and cross-attention updates (Eq. 10–11).

        Args:
            *model_args: Positional arguments for the model's forward.
            target_index (int, tuple of int, or None): Target output index.
            target_fn (callable or None): Custom target-score function.
            text_self_layer_ids (list of str or None): Text self-attention
                layer IDs. Auto-generated if ``None``.
            image_self_layer_ids (list of str or None): Image self-attention
                layer IDs. Auto-generated if ``None``.
            cross_layer_ids (list of tuple or None): Pairs of
                ``(text_cross_layer_id, image_cross_layer_id)`` for the
                co-attention blocks. Must be provided explicitly.
            text_token_count (int or None): Number of text tokens.
            image_token_count (int or None): Number of image tokens.
            **model_kwargs: Keyword arguments for the model's forward.

        Returns:
            dict: Keys ``"R_tt"``, ``"R_ii"``, ``"R_ti"``, ``"R_it"``
            mapping to the four relevance matrices.
        """
        if text_self_layer_ids is None:
            n_text = self._count_layers("T")
            text_self_layer_ids = [f"T.L{i}" for i in range(n_text)]
        if image_self_layer_ids is None:
            n_vision = self._count_layers("V")
            image_self_layer_ids = [f"V.L{i}" for i in range(n_vision)]

        if cross_layer_ids is None:
            raise ValueError(
                "cross_layer_ids must be provided for co-attention models. "
                "Each entry is a (text_cross_layer_id, image_cross_layer_id) "
                "pair."
            )

        all_layer_ids: List[str] = []
        all_layer_ids.extend(text_self_layer_ids)
        all_layer_ids.extend(image_self_layer_ids)
        for t_lid, i_lid in cross_layer_ids:
            all_layer_ids.extend([t_lid, i_lid])

        attn_modules = self._resolve_attn_modules(all_layer_ids)
        hook_mgr = AttentionHookManager(self.model, attn_modules)

        idx = 0
        text_self_indices = list(range(idx, idx + len(text_self_layer_ids)))
        idx += len(text_self_layer_ids)
        image_self_indices = list(range(idx, idx + len(image_self_layer_ids)))
        idx += len(image_self_layer_ids)
        cross_indices: List[Tuple[int, int]] = []
        for _ in cross_layer_ids:
            cross_indices.append((idx, idx + 1))
            idx += 2

        try:
            self.model.zero_grad()
            outputs = self.model(*model_args, **model_kwargs)
            target_score = self._build_target_score(
                outputs, target_index, target_fn
            )
            target_score.backward(retain_graph=True)

            def _get_a(module_idx: int) -> Tuple[Tensor, Tensor]:
                a = hook_mgr.get_attention(module_idx)
                g = hook_mgr.get_gradient(module_idx)
                if a.dim() == 4:
                    a, g = a[0], g[0]
                return a, g

            if text_token_count is None:
                a0, _ = _get_a(text_self_indices[0])
                text_token_count = a0.shape[-1]
            if image_token_count is None:
                a0, _ = _get_a(image_self_indices[0])
                image_token_count = a0.shape[-1]

            t = text_token_count
            i_count = image_token_count

            a_sample, _ = _get_a(0)
            device = a_sample.device
            dtype = a_sample.dtype

            R_tt = torch.eye(t, device=device, dtype=dtype)
            R_ii = torch.eye(i_count, device=device, dtype=dtype)
            R_ti = torch.zeros(t, i_count, device=device, dtype=dtype)
            R_it = torch.zeros(i_count, t, device=device, dtype=dtype)

            for mi in text_self_indices:
                a, g = _get_a(mi)
                A_bar = avg_heads(a, g)
                R_tt, R_ti = apply_self_attention_rules(R_tt, R_ti, A_bar)

            for mi in image_self_indices:
                a, g = _get_a(mi)
                A_bar = avg_heads(a, g)
                R_ii, R_it = apply_self_attention_rules(R_ii, R_it, A_bar)

            for ti_idx, it_idx in cross_indices:
                a_ti, g_ti = _get_a(ti_idx)
                A_ti_bar = avg_heads(a_ti, g_ti)
                R_tt, R_ti = apply_mm_attention_rules(
                    R_tt, R_ii, R_ti, R_it, A_ti_bar, use_eq11=True
                )

                a_it, g_it = _get_a(it_idx)
                A_it_bar = avg_heads(a_it, g_it)
                R_ii, R_it = apply_mm_attention_rules(
                    R_ii, R_tt, R_it, R_ti, A_it_bar, use_eq11=True
                )

        finally:
            hook_mgr.remove_hooks()

        result = {
            "R_tt": R_tt,
            "R_ii": R_ii,
            "R_ti": R_ti,
            "R_it": R_it,
        }
        if self.device is not None:
            result = {k: v.to(self.device) for k, v in result.items()}
        return result

    # ----------------------------------------------------------------
    # Template C: Encoder-decoder (DETR-style)
    # ----------------------------------------------------------------

    @log_usage(part_of_slo=True)
    def generate_encoder_decoder(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        *model_args: Any,
        target_index: Optional[Union[int, Tuple[int, ...]]] = None,
        target_fn: Optional[Callable[..., Tensor]] = None,
        encoder_self_layer_ids: Optional[List[str]] = None,
        decoder_self_layer_ids: Optional[List[str]] = None,
        decoder_cross_layer_ids: Optional[List[str]] = None,
        encoder_token_count: Optional[int] = None,
        decoder_query_count: Optional[int] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        **model_kwargs: Any,
    ) -> Dict[str, Tensor]:
        r"""
        Generate per-token relevance for an encoder-decoder transformer
        (Template C — DETR style).

        Maintains three relevance matrices:
        :math:`R_{ee}, R_{qq}, R_{qe}` (no :math:`R_{eq}`
        because the encoder is not conditioned on the decoder).

        Args:
            *model_args: Positional arguments for the model's forward.
            target_index (int, tuple of int, or None): Target output index.
            target_fn (callable or None): Custom target-score function.
            encoder_self_layer_ids (list of str or None): Encoder layer IDs.
            decoder_self_layer_ids (list of str or None): Decoder
                self-attention layer IDs.
            decoder_cross_layer_ids (list of str or None): Decoder
                cross-attention layer IDs (query → encoder).
            encoder_token_count (int or None): Number of encoder tokens.
            decoder_query_count (int or None): Number of decoder queries.
            **model_kwargs: Keyword arguments for the model's forward.

        Returns:
            dict: Keys ``"R_ee"``, ``"R_qq"``, ``"R_qe"`` mapping to
            the three relevance matrices.
        """
        if encoder_self_layer_ids is None:
            n_enc = self._count_layers("V")
            encoder_self_layer_ids = [f"V.L{i}" for i in range(n_enc)]
        if decoder_self_layer_ids is None:
            n_dec = self._count_layers("T")
            decoder_self_layer_ids = [f"T.L{i}" for i in range(n_dec)]
        if decoder_cross_layer_ids is None:
            decoder_cross_layer_ids = decoder_self_layer_ids.copy()

        all_layer_ids = (
            encoder_self_layer_ids
            + decoder_self_layer_ids
            + decoder_cross_layer_ids
        )
        attn_modules = self._resolve_attn_modules(all_layer_ids)
        hook_mgr = AttentionHookManager(self.model, attn_modules)

        idx = 0
        enc_indices = list(range(idx, idx + len(encoder_self_layer_ids)))
        idx += len(encoder_self_layer_ids)
        dec_self_indices = list(range(idx, idx + len(decoder_self_layer_ids)))
        idx += len(decoder_self_layer_ids)
        dec_cross_indices = list(
            range(idx, idx + len(decoder_cross_layer_ids))
        )

        try:
            self.model.zero_grad()
            outputs = self.model(*model_args, **model_kwargs)
            target_score = self._build_target_score(
                outputs, target_index, target_fn
            )
            target_score.backward(retain_graph=True)

            def _get_a(module_idx: int) -> Tuple[Tensor, Tensor]:
                a = hook_mgr.get_attention(module_idx)
                g = hook_mgr.get_gradient(module_idx)
                if a.dim() == 4:
                    a, g = a[0], g[0]
                return a, g

            if encoder_token_count is None:
                a0, _ = _get_a(enc_indices[0])
                encoder_token_count = a0.shape[-1]
            if decoder_query_count is None:
                a0, _ = _get_a(dec_self_indices[0])
                decoder_query_count = a0.shape[-1]

            e = encoder_token_count
            q = decoder_query_count

            a_sample, _ = _get_a(0)
            device = a_sample.device
            dtype = a_sample.dtype

            R_ee = torch.eye(e, device=device, dtype=dtype)
            R_qq = torch.eye(q, device=device, dtype=dtype)
            R_qe = torch.zeros(q, e, device=device, dtype=dtype)

            for mi in enc_indices:
                a, g = _get_a(mi)
                A_bar = avg_heads(a, g)
                R_ee = R_ee + A_bar @ R_ee

            for self_mi, cross_mi in zip(dec_self_indices, dec_cross_indices):
                a_qq, g_qq = _get_a(self_mi)
                A_qq_bar = avg_heads(a_qq, g_qq)
                R_qq, R_qe = apply_self_attention_rules(
                    R_qq, R_qe, A_qq_bar
                )

                a_qe, g_qe = _get_a(cross_mi)
                A_qe_bar = avg_heads(a_qe, g_qe)

                R_qq_bar = normalize_self_relevance(R_qq)
                R_ee_bar = normalize_self_relevance(R_ee)
                R_qe = (
                    R_qe
                    + R_qq_bar.transpose(-2, -1) @ A_qe_bar @ R_ee_bar
                )

        finally:
            hook_mgr.remove_hooks()

        result = {
            "R_ee": R_ee,
            "R_qq": R_qq,
            "R_qe": R_qe,
        }
        if self.device is not None:
            result = {k: v.to(self.device) for k, v in result.items()}
        return result

    # ----------------------------------------------------------------
    # Unified entry point
    # ----------------------------------------------------------------

    @log_usage(part_of_slo=True)
    def generate_relevance(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        *model_args: Any,
        target_index: Optional[Union[int, Tuple[int, ...]]] = None,
        target_fn: Optional[Callable[..., Tensor]] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs: Any,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        r"""
        Generate relevance maps using the architecture mode set at init.

        Dispatches to :meth:`generate_single_stream`,
        :meth:`generate_co_attention`, or
        :meth:`generate_encoder_decoder` depending on ``self.mode``.

        Args:
            *model_args: Positional arguments for the model's forward.
            target_index: Target output index.
            target_fn: Custom target-score function.
            **kwargs: Forwarded to the selected template method.

        Returns:
            For ``single_stream``: a ``(n, n)`` relevance matrix.
            For ``co_attention``: a dict with ``R_tt, R_ii, R_ti, R_it``.
            For ``encoder_decoder``: a dict with ``R_ee, R_qq, R_qe``.
        """
        if self.mode is None:
            raise ValueError(
                "No explainability mode set. Pass mode='single_stream', "
                "'co_attention', or 'encoder_decoder' to the constructor."
            )
        if self.mode == ExplainabilityMode.SINGLE_STREAM:
            return self.generate_single_stream(
                *model_args,
                target_index=target_index,
                target_fn=target_fn,
                **kwargs,
            )
        elif self.mode == ExplainabilityMode.CO_ATTENTION:
            return self.generate_co_attention(
                *model_args,
                target_index=target_index,
                target_fn=target_fn,
                **kwargs,
            )
        elif self.mode == ExplainabilityMode.ENCODER_DECODER:
            return self.generate_encoder_decoder(
                *model_args,
                target_index=target_index,
                target_fn=target_fn,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ----------------------------------------------------------------
    # Convenience: extract per-token relevance from a readout position
    # ----------------------------------------------------------------

    @staticmethod
    def readout(
        R: Union[Tensor, Dict[str, Tensor]],
        readout_index: int = 0,
        modality: Optional[str] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        r"""
        Extract per-token relevance from a specific readout position.

        For a classification model with a ``[CLS]`` token at index 0,
        calling ``readout(R, 0)`` extracts the row of the relevance
        matrix corresponding to ``[CLS]``, giving the relevance of every
        token to the classification decision.

        Args:
            R: Relevance output from :meth:`generate_relevance`.
            readout_index (int): Row index to read from. Default ``0``
                (typical for ``[CLS]`` token).
            modality (str or None): For multi-modal outputs, specify which
                self-relevance to read from (``"text"`` / ``"image"`` /
                ``"query"``). If ``None`` and ``R`` is a dict, returns
                a dict of readouts.

        Returns:
            Tensor or dict of Tensor: 1-D relevance vector(s).
        """
        if isinstance(R, Tensor):
            return R[readout_index]

        result = {}
        if modality == "text":
            result["text"] = R["R_tt"][readout_index]
            if "R_ti" in R:
                result["image"] = R["R_ti"][readout_index]
        elif modality == "image":
            result["image"] = R["R_ii"][readout_index]
            if "R_it" in R:
                result["text"] = R["R_it"][readout_index]
        elif modality == "query":
            if "R_qe" in R:
                result["encoder"] = R["R_qe"][readout_index]
            if "R_qq" in R:
                result["query"] = R["R_qq"][readout_index]
        else:
            for key, mat in R.items():
                if mat.shape[0] > readout_index:
                    result[key] = mat[readout_index]
        return result


# ────────────────────────────────────────────────────────────────────
# MultiModalAttribution
# ────────────────────────────────────────────────────────────────────

# Supported LayerAttribution methods for MultiModalAttribution
_SUPPORTED_LAYER_METHODS: Tuple[Type[LayerAttribution], ...] = (
    LayerActivation,
    LayerConductance,
    LayerGradientShap,
    LayerGradientXActivation,
    LayerIntegratedGradients,
)


class MultiModalAttribution(Attribution):
    r"""
    Attribution method for multimodal and transformer-based models.
    Integrates captum's existing ``LayerAttribution`` methods with the
    ``MultiModalModelWrapper`` to enable layer-level attribution using
    compact layer identifiers.

    Users specify which layer(s) to attribute using the same compact
    notation (e.g., ``"V.L1.attn"``, ``"T.L3"``), and this class
    resolves them to the appropriate modules before delegating to
    the underlying ``LayerAttribution`` method.

    Args:
        model_wrapper (MultiModalModelWrapper): A wrapped transformer model
                    providing layer resolution capabilities.
        attr_method_class (type): A ``LayerAttribution`` subclass to use
                    for computing attributions. Supported classes include:

                    - ``LayerActivation``
                    - ``LayerConductance``
                    - ``LayerGradientShap``
                    - ``LayerGradientXActivation``
                    - ``LayerIntegratedGradients``

        layer_id (str or LayerID): Target layer identifier for attribution,
                    e.g., ``"V.L1.attn"`` or ``"T.L3"``.

    Examples::

        >>> from captum.attr import (
        ...     MultiModalModelWrapper,
        ...     MultiModalAttribution,
        ...     LayerActivation,
        ... )
        >>> wrapper = MultiModalModelWrapper(model, "llama")
        >>> mm_attr = MultiModalAttribution(
        ...     wrapper, LayerActivation, "T.L3"
        ... )
        >>> activation = mm_attr.attribute(input_tensor)
    """

    SUPPORTED_METHODS: Tuple[Type[LayerAttribution], ...] = _SUPPORTED_LAYER_METHODS

    def __init__(
        self,
        model_wrapper: MultiModalModelWrapper,
        attr_method_class: Type[LayerAttribution],
        layer_id: Union[str, LayerID],
    ) -> None:
        if attr_method_class not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported attribution method: {attr_method_class.__name__}. "
                f"Supported methods: "
                f"{[m.__name__ for m in self.SUPPORTED_METHODS]}"
            )

        self.model_wrapper = model_wrapper
        self.layer_id = (
            LayerID.parse(layer_id) if isinstance(layer_id, str) else layer_id
        )
        layer_module = model_wrapper.resolve_layer(self.layer_id)

        self.attr_method: LayerAttribution = attr_method_class(
            model_wrapper.model, layer_module
        )

        super().__init__(model_wrapper.model)

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: Union[None, Tensor, Tuple[Union[Tensor, int, float], ...]] = None,
        target: Union[None, int, Tuple[int, ...], Tensor, List[Tuple[int, ...]]] = None,
        additional_forward_args: Optional[object] = None,
        attribute_to_layer_input: bool = False,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]:
        r"""
        Compute layer attributions for the specified layer using the
        configured attribution method.

        Delegates to the underlying ``LayerAttribution`` method's
        ``attribute`` call with the resolved layer module.

        Args:
            inputs (Tensor or tuple[Tensor, ...]): Input for which layer
                        attribution is computed. If the model takes a single
                        tensor as input, a single input tensor should be
                        provided. If the model takes multiple tensors, a
                        tuple of input tensors should be provided.
            baselines (Tensor, tuple, int, float, or None, optional):
                        Baselines define the starting point for attribution.
                        Required by methods such as
                        ``LayerIntegratedGradients`` and
                        ``LayerConductance``. Not used by
                        ``LayerActivation``.
                        Default: ``None``.
            target (int, tuple, Tensor, list, or None, optional): Output
                        indices for which attributions are computed. For
                        classification, this is typically the target class.
                        Default: ``None``.
            additional_forward_args (Any, optional): Additional arguments
                        for the model's forward function beyond the inputs.
                        Default: ``None``.
            attribute_to_layer_input (bool, optional): If ``True``, computes
                        attribution with respect to the layer's input
                        rather than its output.
                        Default: ``False``.
            **kwargs: Additional keyword arguments passed to the underlying
                        attribution method's ``attribute`` call. For
                        example, ``n_steps`` for ``LayerIntegratedGradients``
                        or ``n_samples`` for ``LayerGradientShap``.

        Returns:
            *Tensor* or *tuple[Tensor, ...]* or *list* of **attributions**:
            - **attributions**: Attribution values for each neuron in the
              specified layer. Shape matches the layer's output (or input
              if ``attribute_to_layer_input=True``).
        """
        # Build kwargs for the underlying attribution method
        attr_kwargs: Dict[str, Any] = {
            "inputs": inputs,
            "attribute_to_layer_input": attribute_to_layer_input,
        }

        if additional_forward_args is not None:
            attr_kwargs["additional_forward_args"] = additional_forward_args

        # Only pass baselines and target if the method supports them
        # (LayerActivation does not use baselines or target)
        if not isinstance(self.attr_method, LayerActivation):
            if baselines is not None:
                attr_kwargs["baselines"] = baselines
            if target is not None:
                attr_kwargs["target"] = target

        attr_kwargs.update(kwargs)

        return self.attr_method.attribute(**attr_kwargs)
