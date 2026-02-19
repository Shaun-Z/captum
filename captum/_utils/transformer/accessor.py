#!/usr/bin/env python3

# pyre-strict

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from captum._utils.transformer.arch_config import (
    get_arch_config,
    TransformerArchConfig,
)
from captum._utils.transformer.layer_id import LayerID
from torch import nn, Tensor


class ActivationAccessor:
    r"""
    Provides a unified interface for accessing internal activations of
    transformer models. It resolves compact layer identifiers (e.g.,
    ``"V.L1.H5"``) to the corresponding ``torch.nn.Module`` and supports
    hook-based activation extraction during forward passes.

    This class abstracts away architecture-specific module naming differences,
    allowing users to work with a consistent notation across different model
    families (GPT-2, LLaMA, ViT, CLIP, LLaVA, etc.).

    Args:
        model (torch.nn.Module): The transformer model to access.
        arch_config (str or TransformerArchConfig): Either a string naming
                    a pre-registered architecture (e.g., ``"llama"``,
                    ``"vit"``, ``"llava"``), or a ``TransformerArchConfig``
                    instance for custom architectures.
        device (str or torch.device or None): Device to move extracted
                    activations to. If ``None``, activations remain on
                    their original device.

    Examples::

        >>> from captum._utils.transformer import ActivationAccessor
        >>> accessor = ActivationAccessor(model, "llama")
        >>> layer_module = accessor.resolve_module("T.L3.attn")
        >>> activation = accessor.get_activation("T.L3", input_ids)
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

    def _parse_layer_id(self, layer_id: Union[str, LayerID]) -> LayerID:
        r"""
        Parse a layer identifier, accepting either a string or LayerID.

        Args:
            layer_id (str or LayerID): The layer identifier to parse.

        Returns:
            LayerID: The parsed layer identifier.
        """
        if isinstance(layer_id, str):
            return LayerID.parse(layer_id)
        return layer_id

    def _build_module_path(self, layer_id: LayerID) -> str:
        r"""
        Build the dot-separated module path for a given layer identifier.

        Args:
            layer_id (LayerID): The structured layer identifier.

        Returns:
            str: The full dot-separated path to the target module within
            the model.

        Raises:
            ValueError: If the encoder prefix cannot be resolved.
        """
        prefix = self.arch_config.get_encoder_prefix(layer_id.encoder_type)

        if prefix is None:
            raise ValueError(
                f"Cannot resolve encoder prefix for layer '{layer_id}'. "
                "Ensure the architecture config has appropriate encoder "
                "prefixes defined."
            )

        # Build path: prefix.layer_num[.component]
        path = f"{prefix}.{layer_id.layer_num}"

        component_name = self.arch_config.get_component_name(layer_id.component)
        if component_name is not None:
            path = f"{path}.{component_name}"

        return path

    def _get_module_by_path(self, path: str) -> nn.Module:
        r"""
        Navigate the model's module hierarchy using a dot-separated path.

        Args:
            path (str): Dot-separated module path, e.g.,
                        ``"model.layers.3.self_attn"``.

        Returns:
            torch.nn.Module: The module at the specified path.

        Raises:
            AttributeError: If the path does not exist in the model.
        """
        module = self.model
        for part in path.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module

    def resolve_module(
        self,
        layer_id: Union[str, LayerID],
    ) -> nn.Module:
        r"""
        Resolve a layer identifier to the corresponding
        ``torch.nn.Module`` within the model.

        Args:
            layer_id (str or LayerID): The layer identifier, e.g.,
                        ``"V.L1.attn"`` or a ``LayerID`` instance.

        Returns:
            torch.nn.Module: The resolved module.

        Raises:
            ValueError: If the layer identifier cannot be resolved.
            AttributeError: If the module path does not exist.

        Examples::

            >>> accessor = ActivationAccessor(model, "llama")
            >>> attn_module = accessor.resolve_module("T.L3.attn")
        """
        lid = self._parse_layer_id(layer_id)
        path = self._build_module_path(lid)
        return self._get_module_by_path(path)

    def resolve_layers(
        self,
        layer_ids: List[Union[str, LayerID]],
    ) -> List[nn.Module]:
        r"""
        Resolve multiple layer identifiers to their corresponding modules.

        Args:
            layer_ids (list of str or LayerID): Layer identifiers to resolve.

        Returns:
            list of torch.nn.Module: The resolved modules, in the same
            order as the input identifiers.
        """
        return [self.resolve_module(lid) for lid in layer_ids]

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
        Extract the activation of a specific layer during a forward pass.

        Registers a forward hook on the target module, runs the model's
        forward pass with the given inputs, and returns the captured
        activation tensor.

        Args:
            layer_id (str or LayerID): The layer identifier, e.g.,
                        ``"T.L3"`` or ``"V.L1.attn"``.
            *model_args: Positional arguments passed to the model's
                        forward method.
            attribute_to_layer_input (bool, optional): If ``True``, captures
                        the input to the layer instead of its output.
                        Default: ``False``.
            **model_kwargs: Keyword arguments passed to the model's
                        forward method.

        Returns:
            Tensor: The activation tensor captured from the specified layer.
                    If the layer outputs a tuple, the first element is
                    returned.

        Examples::

            >>> accessor = ActivationAccessor(model, "llama")
            >>> act = accessor.get_activation("T.L3", input_ids=tokens)
        """
        lid = self._parse_layer_id(layer_id)
        module = self.resolve_module(lid)

        activation: Dict[str, Optional[Tensor]] = {"value": None}

        def hook_fn(
            mod: nn.Module,
            inp: Tuple[Tensor, ...],
            output: Union[Tensor, Tuple[Tensor, ...]],
        ) -> None:
            if attribute_to_layer_input:
                act = inp[0] if isinstance(inp, tuple) else inp
            else:
                act = output[0] if isinstance(output, tuple) else output

            if isinstance(act, Tensor):
                activation["value"] = act.detach()

        handle = module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                self.model(*model_args, **model_kwargs)
        finally:
            handle.remove()

        result = activation["value"]
        if result is None:
            raise RuntimeError(
                f"No activation captured for layer '{layer_id}'. "
                "Verify that the layer is traversed during the forward pass."
            )

        if lid.head_num is not None:
            result = self._extract_head(result, lid.head_num)

        if self.device is not None:
            result = result.to(self.device)

        return result

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
            layer_ids (list of str or LayerID): Layer identifiers to
                        capture activations from.
            *model_args: Positional arguments passed to the model's
                        forward method.
            attribute_to_layer_input (bool, optional): If ``True``, captures
                        the input to each layer instead of its output.
                        Default: ``False``.
            **model_kwargs: Keyword arguments passed to the model's
                        forward method.

        Returns:
            dict: Mapping from layer identifier strings to activation
            tensors.
        """
        parsed_ids = [self._parse_layer_id(lid) for lid in layer_ids]
        modules = [self.resolve_module(lid) for lid in parsed_ids]

        activations: Dict[str, Optional[Tensor]] = {
            str(lid): None for lid in parsed_ids
        }

        handles = []

        def make_hook(
            lid: LayerID,
        ) -> Callable[
            [nn.Module, Tuple[Tensor, ...], Union[Tensor, Tuple[Tensor, ...]]],
            None,
        ]:
            def hook_fn(
                mod: nn.Module,
                inp: Tuple[Tensor, ...],
                output: Union[Tensor, Tuple[Tensor, ...]],
            ) -> None:
                if attribute_to_layer_input:
                    act = inp[0] if isinstance(inp, tuple) else inp
                else:
                    act = output[0] if isinstance(output, tuple) else output

                if isinstance(act, Tensor):
                    result = act.detach()
                    if lid.head_num is not None:
                        result = self._extract_head(result, lid.head_num)
                    activations[str(lid)] = result

            return hook_fn

        try:
            for lid, module in zip(parsed_ids, modules):
                handle = module.register_forward_hook(make_hook(lid))
                handles.append(handle)

            with torch.no_grad():
                self.model(*model_args, **model_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        result: Dict[str, Tensor] = {}
        for lid in parsed_ids:
            key = str(lid)
            val = activations[key]
            if val is None:
                raise RuntimeError(
                    f"No activation captured for layer '{key}'. "
                    "Verify that the layer is traversed during the forward pass."
                )
            if self.device is not None:
                val = val.to(self.device)
            result[key] = val

        return result

    def _extract_head(
        self,
        activation: Tensor,
        head_num: int,
    ) -> Tensor:
        r"""
        Extract a single attention head's activation from a multi-head
        activation tensor.

        Assumes the activation has shape
        ``(batch, seq_len, num_heads * head_dim)`` or
        ``(batch, num_heads, seq_len, head_dim)``.

        Args:
            activation (Tensor): The full multi-head activation tensor.
            head_num (int): Index of the head to extract.

        Returns:
            Tensor: The activation for the specified head.

        Raises:
            ValueError: If the activation shape is incompatible with
                        head extraction and ``num_attention_heads`` is not
                        configured.
        """
        if activation.dim() == 4:
            # Shape: (batch, num_heads, seq_len, head_dim)
            return activation[:, head_num]

        if activation.dim() == 3:
            # Shape: (batch, seq_len, num_heads * head_dim)
            num_heads = self.arch_config.num_attention_heads
            if num_heads is None:
                raise ValueError(
                    "Cannot extract head from 3D activation without "
                    "'num_attention_heads' in the architecture config."
                )
            head_dim = activation.shape[-1] // num_heads
            start = head_num * head_dim
            end = start + head_dim
            return activation[:, :, start:end]

        raise ValueError(
            f"Cannot extract head from activation with shape "
            f"{activation.shape}. Expected 3D or 4D tensor."
        )
