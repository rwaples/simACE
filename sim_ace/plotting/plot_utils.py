"""Shared plotting utilities for sim_ace."""

from __future__ import annotations

from typing import Any

import numpy as np

PAIR_COLORS: dict[str, str] = {
    "MZ": "C0",
    "FS": "C1",
    "MO": "C3",
    "FO": "C5",
    "MHS": "C2",
    "PHS": "C6",
    "1C": "C4",
}


def save_placeholder_plot(
    output_path: Any, message: str, figsize: tuple[float, float] = (6, 4), dpi: int = 150
) -> None:
    """Save a single-panel figure with centered message text."""
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _make_heatmap_cmap():
    """Truncated GnBu (green-to-blue) starting at 40% — dark enough for white text."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    gnbu = plt.cm.GnBu
    return mcolors.LinearSegmentedColormap.from_list("GnBu_dark", gnbu(np.linspace(0.4, 1.0, 256)))


HEATMAP_CMAP = _make_heatmap_cmap()


def annotate_heatmap(ax, proportions, counts, fmt_prop=".2f", prop_size=18, count_size=11) -> None:
    """Add two-line annotations to a heatmap: large bold proportion, smaller count.

    Args:
        ax: Matplotlib axes containing the heatmap.
        proportions: 2-D array-like of proportion values.
        counts: 2-D array-like of count values (int or float).
        fmt_prop: Format spec for proportion values.
        prop_size: Font size for the proportion line.
        count_size: Font size for the count line.
    """
    proportions = np.asarray(proportions)
    counts = np.asarray(counts)
    for i in range(proportions.shape[0]):
        for j in range(proportions.shape[1]):
            p = proportions[i, j]
            c = counts[i, j]
            c_str = f"n={int(c)}" if float(c) == int(c) else f"n={c:.0f}"
            ax.text(
                j + 0.5,
                i + 0.38,
                f"{p:{fmt_prop}}",
                ha="center",
                va="center",
                fontsize=prop_size,
                fontweight="bold",
                color="white",
            )
            ax.text(j + 0.5, i + 0.62, c_str, ha="center", va="center", fontsize=count_size, color=(1, 1, 1, 0.7))


def finalize_plot(
    output_path: Any, dpi: int = 150, tight_rect: list[float] | None = None, subsample_note: str = ""
) -> None:
    """tight_layout + savefig(bbox_inches='tight') + close current figure."""
    import warnings

    import matplotlib.pyplot as plt

    if subsample_note:
        fig = plt.gcf()
        fig.text(
            0.99,
            0.01,
            subsample_note,
            fontsize=8,
            color="0.5",
            ha="right",
            va="bottom",
            fontstyle="italic",
            transform=fig.transFigure,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
        if tight_rect is not None:
            plt.tight_layout(rect=tight_rect)
        else:
            plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def draw_split_violin(
    ax,
    data_left,
    data_right,
    pos,
    color_left="C0",
    color_right="C1",
    width=0.8,
):
    """Draw a split violin at *pos* (left half / right half).

    Replicates seaborn's ``violinplot(split=True, cut=0)`` using raw
    matplotlib, which is significantly faster for large arrays.
    """
    for data, color, side in [
        (data_left, color_left, "left"),
        (data_right, color_right, "right"),
    ]:
        if data is None or len(data) < 2:
            continue
        parts = ax.violinplot(
            [data],
            positions=[pos],
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=width,
        )
        for body in parts["bodies"]:
            verts = body.get_paths()[0].vertices
            if side == "left":
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
            else:
                verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_linewidth(0.8)
            body.set_alpha(1.0)
        # Inner box: Q1–Q3 bar + median dot (matches seaborn inner="box")
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        x_inner = pos - width * 0.06 if side == "left" else pos + width * 0.06
        ax.vlines(x_inner, q1, q3, color="black", linewidth=1.5, zorder=4)
        ax.plot(x_inner, med, "o", color="white", ms=3, mew=0, zorder=5)


def draw_colored_violins(
    ax,
    datasets,
    positions,
    colors,
    alpha=0.7,
    width=0.8,
    zorder=3,
):
    """Draw violins at *positions* with per-category *colors*.

    Replicates seaborn's ``violinplot(inner=None, cut=0)`` for
    categorically-coloured violin groups.  Only groups with >= 2 values
    are drawn.
    """
    valid = [(p, d, c) for p, d, c in zip(positions, datasets, colors, strict=True) if len(d) >= 2]
    if not valid:
        return
    v_pos, v_data, v_colors = zip(*valid, strict=True)
    parts = ax.violinplot(
        list(v_data),
        positions=list(v_pos),
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=width,
    )
    for body, color in zip(parts["bodies"], v_colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(alpha)
        body.set_zorder(zorder)
