#!/usr/bin/env python3

# pyre-strict

from typing import Any, Dict, List, Optional, Tuple, Type, Union

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


class MultiModalModelWrapper:
    r"""
    Unified wrapper for LLM, ViT, and LVLM (Large Vision-Language Model)
    architectures. Provides a consistent interface for resolving layer
    identifiers to ``torch.nn.Module`` objects, regardless of the underlying
    model's internal naming conventions.

    This wrapper supports a compact layer notation:

    - ``"L<num>"`` — layer in the default encoder
    - ``"V.L<num>"`` — layer in the vision encoder
    - ``"T.L<num>"`` — layer in the text encoder
    - Append ``.H<num>`` for a specific attention head (e.g., ``"V.L1.H5"``)
    - Append ``.<component>`` for sub-modules (e.g., ``"T.L3.attn"``,
      ``"T.L3.mlp"``)

    Args:
        model (torch.nn.Module): The transformer model to wrap.
        arch_config (str or TransformerArchConfig): Either a registered
                    architecture name (e.g., ``"llama"``, ``"vit"``,
                    ``"llava"``) or a custom ``TransformerArchConfig``
                    instance.
        device (str or torch.device or None): Device for the model and
                    extracted activations. If ``None``, the model's current
                    device is used.

    Examples::

        >>> from captum.attr import MultiModalModelWrapper
        >>> wrapper = MultiModalModelWrapper(model, "llava")
        >>> attn_layer = wrapper.resolve_layer("V.L1.attn")
        >>> text_layer = wrapper.resolve_layer("T.L5")
    """

    def __init__(
        self,
        model: nn.Module,
        arch_config: Union[str, TransformerArchConfig],
        device: Union[str, torch.device, None] = None,
    ) -> None:
        self.model = model
        if isinstance(arch_config, str):
            self.arch_config = get_arch_config(arch_config)
        else:
            self.arch_config = arch_config
        self.device = device
        self.accessor = ActivationAccessor(model, self.arch_config, device)

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
