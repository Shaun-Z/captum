#!/usr/bin/env python3

# pyre-strict

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LayerID:
    r"""
    Structured identifier for a specific layer or component within a
    transformer model. Supports a compact string notation for specifying
    encoder type, layer number, head number, and component name.

    Notation Examples:
        - ``"L1"`` — layer 1 of the default encoder
        - ``"L1.H5"`` — layer 1, attention head 5
        - ``"V.L1"`` — vision encoder, layer 1
        - ``"T.L3"`` — text encoder, layer 3
        - ``"V.L1.H5"`` — vision encoder, layer 1, head 5
        - ``"V.L1.attn"`` — vision encoder, layer 1, attention module
        - ``"T.L2.mlp"`` — text encoder, layer 2, MLP module
        - ``"V.L0.output"`` — vision encoder, layer 0, output/layernorm

    Attributes:
        encoder_type (str or None): Encoder prefix. ``'V'`` for vision
                    encoder, ``'T'`` for text encoder/decoder. ``None``
                    indicates the default (or only) encoder of the model.
        layer_num (int): Zero-based layer index within the encoder.
        head_num (int or None): Attention head index. Only applicable when
                    targeting a specific attention head. ``None`` means the
                    full layer or component is targeted.
        component (str or None): Sub-module within the layer. Common values
                    include ``'attn'`` (self-attention), ``'mlp'``
                    (feed-forward), and ``'output'`` (layer norm / output
                    projection). ``None`` means the layer block itself.
    """

    encoder_type: Optional[str] = None
    layer_num: int = 0
    head_num: Optional[int] = None
    component: Optional[str] = None

    # Pattern: optional encoder prefix (V/T), required layer (L<num>),
    # optional head (H<num>) or component name (attn/mlp/output/...)
    _PATTERN: str = (
        r"^(?:(?P<encoder>[VT])\.)?L(?P<layer>\d+)"
        r"(?:\.(?:H(?P<head>\d+)|(?P<component>[a-zA-Z_]+)))?$"
    )

    @classmethod
    def parse(cls, layer_str: str) -> "LayerID":
        r"""
        Parse a compact layer identifier string into a ``LayerID`` instance.

        Args:
            layer_str (str): A string in the format described in the class
                        docstring. Examples: ``"V.L1.H5"``, ``"T.L3"``,
                        ``"L0.attn"``.

        Returns:
            LayerID: A structured representation of the layer identifier.

        Raises:
            ValueError: If ``layer_str`` does not match the expected format.

        Examples::

            >>> LayerID.parse("V.L1.H5")
            LayerID(encoder_type='V', layer_num=1, head_num=5, component=None)
            >>> LayerID.parse("T.L3")
            LayerID(encoder_type='T', layer_num=3, head_num=None, component=None)
            >>> LayerID.parse("L0.attn")
            LayerID(encoder_type=None, layer_num=0, head_num=None, component='attn')
        """
        match = re.match(cls._PATTERN, layer_str.strip())
        if not match:
            raise ValueError(
                f"Invalid layer identifier: '{layer_str}'. "
                "Expected format: [V|T.]L<num>[.H<num>|.<component>]. "
                "Examples: 'L1', 'V.L1.H5', 'T.L3.attn'"
            )

        encoder_type = match.group("encoder")
        layer_num = int(match.group("layer"))
        head_str = match.group("head")
        head_num = int(head_str) if head_str is not None else None
        component = match.group("component")

        return cls(
            encoder_type=encoder_type,
            layer_num=layer_num,
            head_num=head_num,
            component=component,
        )

    def __str__(self) -> str:
        parts = []
        if self.encoder_type:
            parts.append(self.encoder_type)
        parts.append(f"L{self.layer_num}")
        if self.head_num is not None:
            parts.append(f"H{self.head_num}")
        elif self.component:
            parts.append(self.component)
        return ".".join(parts)
