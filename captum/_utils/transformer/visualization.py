#!/usr/bin/env python3

# pyre-strict

"""
Visualization utilities for transformer activation probing.

Provides heatmap-based visualization of activations extracted via
``ActivationAccessor`` or ``MultiModalModelWrapper``. Built on top of
matplotlib, following the same conventions as the existing
``captum.attr._utils.visualization`` module.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Default layout constants for auto-computed figure sizes
_MIN_FIG_WIDTH = 12
_FIG_WIDTH_PER_PLOT = 2
_FIG_HEIGHT_PER_PLOT = 3
_STATS_FIG_WIDTH = 10

# Maximum number of feature rows to display in heatmaps before subsampling.
# Beyond this, individual features become indistinguishable in typical
# display resolutions, so we evenly subsample for a legible overview.
_MAX_HEATMAP_FEATURES = 64


def _to_numpy(t: Union[Tensor, npt.NDArray[Any]]) -> npt.NDArray[Any]:
    """Convert a tensor or ndarray to a numpy array (float32)."""
    if isinstance(t, Tensor):
        return t.detach().float().cpu().numpy()
    return np.asarray(t, dtype=np.float32)


def visualize_activations(
    activations: Dict[str, Union[Tensor, npt.NDArray[Any]]],
    tokens: Optional[Sequence[str]] = None,
    batch_index: int = 0,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: Union[str, "Colormap"] = "RdBu_r",
    show_values: bool = False,
    value_fmt: str = ".2f",
    title: Optional[str] = None,
    use_pyplot: bool = True,
) -> "Figure":
    r"""
    Visualize probed activations as a grid of heatmaps.

    Each entry in ``activations`` is rendered as a separate subplot, with
    the sequence dimension on the x-axis and the feature / hidden dimension
    on the y-axis. If ``tokens`` are provided, they are used as x-axis tick
    labels.

    Args:
        activations (dict): Mapping from layer-ID strings (e.g.,
                    ``"L0.mlp.act"``) to activation tensors. Each tensor
                    should have shape ``(batch, seq_len, hidden_dim)`` or
                    ``(seq_len, hidden_dim)``.
        tokens (sequence of str, optional): Token strings for the x-axis.
                    If ``None``, integer indices are used.
                    Default: ``None``
        batch_index (int): Which sample in the batch to visualize.
                    Default: ``0``
        figsize (tuple of int, optional): Figure size ``(width, height)``.
                    If ``None``, automatically computed based on the number
                    of subplots.
                    Default: ``None``
        cmap (str or Colormap): Colormap for the heatmaps.
                    Default: ``"RdBu_r"``
        show_values (bool): If ``True``, overlay numerical values on cells.
                    Only practical for small sequence lengths.
                    Default: ``False``
        value_fmt (str): Format string for cell values when
                    ``show_values=True``.
                    Default: ``".2f"``
        title (str, optional): Overall figure title.
                    Default: ``None``
        use_pyplot (bool): If ``True``, use ``plt.show()`` to display.
                    If ``False``, just return the figure without showing.
                    Default: ``True``

    Returns:
        matplotlib.figure.Figure: The figure object containing the
        heatmap grid.

    Example::

        >>> from captum._utils.transformer.visualization import (
        ...     visualize_activations,
        ... )
        >>> acts = accessor.get_multi_layer_activations(
        ...     ["L0.mlp.c_fc", "L0.mlp.act"], input_ids
        ... )
        >>> fig = visualize_activations(acts, tokens=["The", "cat", "sat"])
    """
    assert HAS_MATPLOTLIB, (
        "matplotlib is required for visualization. "
        "Please run 'pip install matplotlib'."
    )

    names = list(activations.keys())
    n_plots = len(names)
    if n_plots == 0:
        raise ValueError("activations dict is empty, nothing to visualize.")

    if figsize is None:
        figsize = (
            max(_MIN_FIG_WIDTH, n_plots * _FIG_WIDTH_PER_PLOT),
            _FIG_HEIGHT_PER_PLOT * n_plots,
        )

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
    axes_list: List[Axes] = [axes[i, 0] for i in range(n_plots)]

    for ax, name in zip(axes_list, names):
        data = _to_numpy(activations[name])

        # Handle batch dimension
        if data.ndim == 3:
            data = data[batch_index]
        elif data.ndim == 1:
            data = data[np.newaxis, :]

        # data is now (seq_len, hidden_dim) — transpose for display
        # so x-axis = sequence position, y-axis = feature dimension
        # For heatmap: rows = feature dim (top features at top), cols = seq pos
        display_data = data.T  # (hidden_dim, seq_len)

        # Subsample feature dim if too large for legible display
        h_dim = display_data.shape[0]
        if h_dim > _MAX_HEATMAP_FEATURES:
            indices = np.linspace(0, h_dim - 1, _MAX_HEATMAP_FEATURES, dtype=int)
            display_data = display_data[indices]

        vmax = np.abs(display_data).max()
        vmin = -vmax if vmax > 0 else -1.0

        im = ax.imshow(
            display_data,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(name, fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel(f"features\n(d={h_dim})", fontsize=8)

        # X-axis: tokens or indices
        seq_len = display_data.shape[1]
        if tokens is not None and len(tokens) == seq_len:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(range(seq_len), fontsize=7)

        ax.set_yticks([])

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

        # Overlay values
        if show_values and seq_len <= 20 and display_data.shape[0] <= 20:
            for i in range(display_data.shape[0]):
                for j in range(display_data.shape[1]):
                    val = display_data[i, j]
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(
                        j, i, f"{val:{value_fmt}}",
                        ha="center", va="center",
                        fontsize=6, color=color,
                    )

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    fig.tight_layout()

    if use_pyplot:
        plt.show()

    return fig


def visualize_activation_stats(
    activations: Dict[str, Union[Tensor, npt.NDArray[Any]]],
    tokens: Optional[Sequence[str]] = None,
    batch_index: int = 0,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    use_pyplot: bool = True,
) -> "Figure":
    r"""
    Visualize summary statistics (mean, std, max, min) of probed activations
    across the feature dimension, producing a compact per-token bar chart
    for each layer.

    This is useful when the hidden dimension is too large for per-element
    heatmaps.

    Args:
        activations (dict): Mapping from layer-ID strings to activation
                    tensors with shape ``(batch, seq_len, hidden_dim)`` or
                    ``(seq_len, hidden_dim)``.
        tokens (sequence of str, optional): Token strings for the x-axis.
                    Default: ``None``
        batch_index (int): Which sample in the batch to visualize.
                    Default: ``0``
        figsize (tuple of int, optional): Figure size ``(width, height)``.
                    Default: ``None``
        title (str, optional): Overall figure title.
                    Default: ``None``
        use_pyplot (bool): If ``True``, display the plot.
                    Default: ``True``

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    assert HAS_MATPLOTLIB, (
        "matplotlib is required for visualization. "
        "Please run 'pip install matplotlib'."
    )

    names = list(activations.keys())
    n_plots = len(names)
    if n_plots == 0:
        raise ValueError("activations dict is empty, nothing to visualize.")

    if figsize is None:
        figsize = (_STATS_FIG_WIDTH, _FIG_HEIGHT_PER_PLOT * n_plots)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
    axes_list: List[Axes] = [axes[i, 0] for i in range(n_plots)]

    for ax, name in zip(axes_list, names):
        data = _to_numpy(activations[name])
        if data.ndim == 3:
            data = data[batch_index]
        elif data.ndim == 1:
            data = data[np.newaxis, :]

        # data: (seq_len, hidden_dim)
        seq_len = data.shape[0]
        means = data.mean(axis=-1)
        stds = data.std(axis=-1)

        x = np.arange(seq_len)
        ax.bar(x, means, color="#4772b3", alpha=0.8, label="mean")
        ax.errorbar(
            x, means, yerr=stds,
            fmt="none", ecolor="#d0365b", capsize=2, label="±std",
        )
        ax.set_title(name, fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("value", fontsize=8)

        if tokens is not None and len(tokens) == seq_len:
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x, fontsize=7)

        ax.legend(fontsize=7, loc="upper right")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    fig.tight_layout()

    if use_pyplot:
        plt.show()

    return fig


def visualize_activation_distribution(
    activations: Dict[str, Union[Tensor, npt.NDArray[Any]]],
    batch_index: int = 0,
    figsize: Optional[Tuple[int, int]] = None,
    bins: int = 50,
    title: Optional[str] = None,
    use_pyplot: bool = True,
) -> "Figure":
    r"""
    Visualize the value distribution of each probed activation as
    overlaid histograms.

    This is useful for comparing activation magnitudes across layers or
    for detecting dead neurons (e.g., after ReLU).

    Args:
        activations (dict): Mapping from layer-ID strings to activation
                    tensors.
        batch_index (int): Which sample in the batch to visualize.
                    Default: ``0``
        figsize (tuple of int, optional): Figure size.
                    Default: ``None``
        bins (int): Number of histogram bins.
                    Default: ``50``
        title (str, optional): Overall figure title.
                    Default: ``None``
        use_pyplot (bool): If ``True``, display the plot.
                    Default: ``True``

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    assert HAS_MATPLOTLIB, (
        "matplotlib is required for visualization. "
        "Please run 'pip install matplotlib'."
    )

    names = list(activations.keys())
    if len(names) == 0:
        raise ValueError("activations dict is empty, nothing to visualize.")

    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    for name in names:
        data = _to_numpy(activations[name])
        if data.ndim == 3:
            data = data[batch_index]
        ax.hist(data.flatten(), bins=bins, alpha=0.5, label=name, density=True)

    ax.set_xlabel("Activation Value", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=8)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    else:
        ax.set_title("Activation Value Distribution", fontsize=13)

    fig.tight_layout()

    if use_pyplot:
        plt.show()

    return fig
