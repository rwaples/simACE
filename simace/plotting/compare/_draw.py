"""Drawing primitives shared by the comparison figures: rep envelopes, envelope lines, grouped bars, saving."""

from __future__ import annotations

__all__ = [
    "SCENARIO_PALETTE",
    "Envelope",
    "bar_offsets",
    "draw_envelope_line",
    "draw_grouped_bars",
    "envelope_arrays",
    "padded_range",
    "rep_envelope",
    "save",
    "scenario_color",
]

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from simace.plotting.plot_utils import finalize_plot

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# Ordered palette for scenario lines in comparison plots.  Colorblind-safe,
# picked to read cleanly when 2-5 scenarios are overlaid.
SCENARIO_PALETTE: tuple[str, ...] = (
    "#4477AA",  # blue
    "#EE6677",  # rose
    "#228833",  # green
    "#CCBB44",  # olive
    "#AA3377",  # purple
)


def scenario_color(index: int) -> str:
    """The palette color of the *index*-th scenario, cycling past the palette's end."""
    return SCENARIO_PALETTE[index % len(SCENARIO_PALETTE)]


@dataclass(frozen=True)
class Envelope:
    """Mean and min/max across reps of one quantity; all NaN when no rep has a finite value.

    For a small number of reps (typical: 3) a parametric CI is less honest
    than min/max across reps, so min/max is the envelope.
    """

    mean: float
    low: float
    high: float


def rep_envelope(values: Iterable[float] | np.ndarray) -> Envelope:
    """The :class:`Envelope` of the finite *values* (one per rep)."""
    arr = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return Envelope(math.nan, math.nan, math.nan)
    return Envelope(float(finite.mean()), float(finite.min()), float(finite.max()))


def envelope_arrays(envelopes: Sequence[Envelope]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(means, lows, highs)`` arrays of *envelopes*."""
    return (
        np.array([e.mean for e in envelopes], dtype=float),
        np.array([e.low for e in envelopes], dtype=float),
        np.array([e.high for e in envelopes], dtype=float),
    )


def draw_envelope_line(
    ax: Axes, x: Sequence[float], envelopes: Sequence[Envelope], color: str, label: str, marker: str = "o"
) -> None:
    """Plot the envelope means as a marked line over a translucent min/max band."""
    ax.plot(x, [e.mean for e in envelopes], color=color, marker=marker, label=label)
    ax.fill_between(x, [e.low for e in envelopes], [e.high for e in envelopes], color=color, alpha=0.15, linewidth=0)


def bar_offsets(x: np.ndarray, group_width: float, n_bars: int, index: int) -> np.ndarray:
    """Centers of the *index*-th of *n_bars* bars side by side in each group of width *group_width* around *x*."""
    bar_width = group_width / n_bars
    return x - group_width / 2 + bar_width / 2 + index * bar_width


def draw_grouped_bars(
    ax: Axes, offsets: np.ndarray, envelopes: Sequence[Envelope], bar_width: float, color: str, label: str | None
) -> None:
    """Bars at the envelope means with min/max error bars; groups with no finite mean are left empty."""
    means, lows, highs = envelope_arrays(envelopes)
    mask = np.isfinite(means)
    err_low = np.where(mask, means - lows, 0.0)
    err_high = np.where(mask, highs - means, 0.0)
    ax.bar(
        offsets[mask],
        means[mask],
        width=bar_width * 0.95,
        color=color,
        label=label,
        yerr=[err_low[mask], err_high[mask]],
        capsize=3,
        linewidth=0,
    )


def padded_range(values: np.ndarray) -> tuple[float, float]:
    """The 0.1%-99.9% quantile range of *values*, widened by 5% on each side: a shared axis that ignores outliers."""
    lo = float(np.quantile(values, 0.001))
    hi = float(np.quantile(values, 0.999))
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def save(fig: plt.Figure, output_path: Path | str, rect: tuple[float, float, float, float] | None = None) -> None:
    """Write ``fig`` to ``output_path`` through :func:`finalize_plot`, creating its directory first."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(fig.number)  # finalize_plot lays out and saves the current figure
    finalize_plot(output_path, tight_rect=None if rect is None else list(rect))
    logger.info("Wrote %s", output_path)
