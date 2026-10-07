"""Shared plotting utilities for simace."""

from __future__ import annotations

__all__ = [
    "HEATMAP_CMAP",
    "PAIR_COLORS",
    "PAIR_TYPE_SANE_BAND",
    "annotate_heatmap",
    "draw_colored_violins",
    "draw_regression_band",
    "draw_split_violin",
    "finalize_plot",
    "finalize_relationship_type_panels",
    "mean_regression",
    "param_as_float",
    "relationship_type_legend_handles",
    "save_placeholder_plot",
    "savefig_dpi",
    "setup_relationship_type_panel",
    "sort_generation_keys",
]

# Re-export from plot_style for backward compatibility
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from simace.plotting.plot_style import PAIR_COLORS as PAIR_COLORS

if TYPE_CHECKING:
    from collections.abc import Iterable

    import matplotlib.pyplot as plt


# Stretched "+" marker (1.7x wider than tall) for the per-pair-type observed mean.
# Built once at import to avoid rebuilding per call.
_WIDE_PLUS = MarkerStyle("+", transform=Affine2D().scale(1.7, 1.0))


def _marker_obs_per_rep() -> dict:
    return dict(s=42, color="0.45", zorder=5)


def _marker_obs_mean(color: str | list[str] = "black") -> dict:
    return dict(marker=_WIDE_PLUS, s=200, color=color, linewidths=2.8, zorder=8)


def _marker_obs_mean_halo() -> dict:
    """Wider black ``+`` drawn under the coloured cross to give it a halo."""
    return dict(marker=_WIDE_PLUS, s=200, color="black", linewidths=5.0, zorder=7)


def _marker_liab() -> dict:
    return dict(marker="D", s=95, facecolor="white", edgecolor="black", linewidths=1.5, zorder=6)


def _marker_param(color: str | list[str] | None = None) -> dict:
    if color is None:
        from simace.plotting.plot_style import COLOR_AFFECTED

        color = COLOR_AFFECTED
    return dict(marker="*", s=240, color=color, zorder=7, edgecolor="none")


# Tetrachoric values outside this band are treated as outliers for ylim purposes
# and clipped to the axis edge with a small caret.
PAIR_TYPE_SANE_BAND: tuple[float, float] = (-0.15, 1.05)


def param_as_float(val: float | dict | None, default: float = 0.0) -> float:
    """Convert a scalar or per-generation dict param to a single float.

    For per-gen dicts, returns the value at the lowest key (founder generation).
    """
    if val is None:
        return default
    if isinstance(val, dict):
        return float(val[min(val)])
    return float(val)


# PAIR_COLORS is now imported from plot_style (see re-export above)


def mean_regression(
    rep_stats: list[dict[str, Any]], x: np.ndarray, y: np.ndarray, n_key: str = "n"
) -> tuple[float, float, float, float | None, int]:
    """Return ``(slope, intercept, r, stderr, n)`` averaged over per-rep OLS stats.

    When no rep has stats, fit ``y ~ x`` directly instead. ``stderr`` is the
    slope's standard error, or None when no rep reports one.
    """
    if not rep_stats:
        from simace.core.numerics import fast_linregress

        slope, intercept, r, stderr, _pvalue = fast_linregress(x, y)
        return slope, intercept, r, stderr, len(x)
    stderrs = [s["stderr"] for s in rep_stats if s.get("stderr") is not None]
    return (
        np.mean([s["slope"] for s in rep_stats]),
        np.mean([s["intercept"] for s in rep_stats]),
        np.mean([s["r"] for s in rep_stats]),
        float(np.mean(stderrs)) if stderrs else None,
        int(np.mean([s[n_key] for s in rep_stats])),
    )


def draw_regression_band(
    ax: plt.Axes, x: np.ndarray, slope: float, intercept: float, stderr: float | None, n: int, color: str
) -> None:
    """Shade the 95% confidence band of the OLS line ``slope * x + intercept`` fitted to ``x``."""
    if stderr is None or n <= 2:
        return
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)
    if ss_x <= 1e-12:
        return
    from scipy.stats import t as t_dist

    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_hat = slope * x_smooth + intercept
    s = stderr * np.sqrt(ss_x)  # residual SE, from stderr_slope = s / sqrt(SS_x)
    t_crit = t_dist.ppf(0.975, df=n - 2)
    se_fit = s * np.sqrt(1.0 / n + (x_smooth - x_mean) ** 2 / ss_x)
    ax.fill_between(x_smooth, y_hat - t_crit * se_fit, y_hat + t_crit * se_fit, alpha=0.15, color=color, zorder=2)


def sort_generation_keys(keys: Iterable[str]) -> list[str]:
    """Sort generation keys (``gen3``, ``gen_12``) by their trailing number, not as strings.

    Keys without a trailing number (``all``, ``pooled``) sort after the numbered ones.
    """

    def key(k: str) -> tuple[bool, int, str]:
        digits = k[len(k.rstrip("0123456789")) :]
        return (not digits, int(digits or 0), k)

    return sorted(keys, key=key)


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


def annotate_heatmap(
    ax: plt.Axes,
    proportions: np.ndarray,
    counts: np.ndarray,
    fmt_prop: str = ".2f",
    prop_size: int = 18,
    count_size: int = 11,
    prop_color: str | tuple[float, ...] = "white",
    count_color: str | tuple[float, ...] = (1, 1, 1, 0.7),
) -> None:
    """Add two-line annotations to a heatmap: large bold proportion, smaller count.

    Args:
        ax: Matplotlib axes containing the heatmap.
        proportions: 2-D array-like of proportion values.
        counts: 2-D array-like of count values (int or float).
        fmt_prop: Format spec for proportion values.
        prop_size: Font size for the proportion line.
        count_size: Font size for the count line.
        prop_color: Color for the proportion annotation.
        count_color: Color for the count annotation.
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
                color=prop_color,
            )
            ax.text(j + 0.5, i + 0.62, c_str, ha="center", va="center", fontsize=count_size, color=count_color)


def savefig_dpi() -> int:
    """Return the configured ``savefig.dpi``, or 150 when it is matplotlib's ``"figure"`` sentinel."""
    import matplotlib.pyplot as plt

    configured = plt.rcParams["savefig.dpi"]
    return 150 if configured == "figure" else int(configured)


def finalize_plot(
    output_path: Any,
    dpi: int | None = None,
    tight_rect: list[float] | None = None,
    subsample_note: str = "",
    scenario: str = "",
) -> None:
    """tight_layout + savefig(bbox_inches='tight') + close current figure.

    ``dpi=None`` reads ``savefig.dpi`` from the active rcParams, which
    :func:`~simace.plotting.plot_style.apply_nature_style` pins to 150. That is
    the single knob for output resolution: a consumer that needs print-grade
    rasters raises the rcParam once rather than threading a dpi through every
    plot function. Falls back to 150 when the rcParam is the matplotlib default
    sentinel ``"figure"``.
    """
    import warnings

    import matplotlib.pyplot as plt

    if dpi is None:
        dpi = savefig_dpi()

    from simace.plotting.plot_style import add_scenario_label

    fig = plt.gcf()
    if scenario:
        add_scenario_label(fig, scenario)
    if subsample_note:
        fig.text(
            0.99,
            0.015,
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
            plt.tight_layout(rect=tight_rect)  # ty: ignore[invalid-argument-type]
        else:
            plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def draw_split_violin(
    ax,
    data_left,
    data_right,
    pos,
    color_left=None,
    color_right=None,
    width=0.8,
):
    """Draw a split violin at *pos* (left half / right half).

    Replicates seaborn's ``violinplot(split=True, cut=0)`` using raw
    matplotlib, which is significantly faster for large arrays.
    """
    from simace.plotting.plot_style import COLOR_AFFECTED, COLOR_UNAFFECTED

    if color_left is None:
        color_left = COLOR_UNAFFECTED
    if color_right is None:
        color_right = COLOR_AFFECTED
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
            body.set_linewidth(0.5)
            body.set_alpha(1.0)
        # Inner box: Q1–Q3 bar + median dot (matches seaborn inner="box")
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        x_inner = pos - width * 0.06 if side == "left" else pos + width * 0.06
        ax.vlines(x_inner, q1, q3, color="black", linewidth=1.0, zorder=4)
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


def setup_relationship_type_panel(
    ax,
    relationship_types: list[str],
    n_pairs_per_ptype: dict[str, int],
    n_reps: int,
    observed_per_rep: dict[str, list[float]],
    liability_r: dict[str, float] | None = None,
    parametric_r: dict[str, float] | None = None,
    show_violins_threshold: int = 4,
    pair_colors: dict[str, str] | None = None,
    rng_seed: int = 42,
) -> dict:
    """Render one per-pair-type comparison panel except the per-rep observed dots.

    For each pair type at x = i:
      * faint coloured violin (only when ``n_reps >= show_violins_threshold``)
      * mean-of-observed wide cross
      * open diamond at mean liability r (if provided)
      * filled red star at parametric E[r] (if provided)

    The per-rep observed dots are deferred so :func:`finalize_relationship_type_panels`
    can decide a shared y-axis range across panels and clip outliers to the
    axis edges. Bold pair-type labels and parenthesised pair counts are drawn
    here; titles and y-labels remain caller-specific.

    Returns ``{"ax", "ref_values", "obs_records"}``.
    """
    from simace.plotting.plot_style import enable_value_gridlines

    if pair_colors is None:
        pair_colors = PAIR_COLORS

    ref_values: list[float] = []
    obs_records: list[tuple[Any, float, float]] = []

    if n_reps >= show_violins_threshold:
        datasets = [list(observed_per_rep.get(pt, [])) for pt in relationship_types]
        if any(len(d) >= 2 for d in datasets):
            colors = [pair_colors[pt] for pt in relationship_types]
            draw_colored_violins(ax, datasets, list(range(len(relationship_types))), colors)

    # Light vertical separators between categories
    for i in range(len(relationship_types) - 1):
        ax.axvline(i + 0.5, color="0.88", linewidth=0.6, zorder=0)

    rng = np.random.default_rng(rng_seed)
    # Draw one collection per marker style across relationship types.
    mean_x: list[int] = []
    mean_y: list[float] = []
    mean_colors: list[str] = []
    for i, ptype in enumerate(relationship_types):
        rep_vals = list(observed_per_rep.get(ptype, []))
        if not rep_vals:
            continue
        if len(rep_vals) > 1:
            jitter = rng.uniform(-0.08, 0.08, len(rep_vals))
        else:
            jitter = np.zeros(1)
        for x, v in zip(i + jitter, rep_vals, strict=False):
            obs_records.append((ax, float(x), float(v)))
        mean_v = float(np.mean(rep_vals))
        mean_x.append(i)
        mean_y.append(mean_v)
        mean_colors.append(pair_colors[ptype])
        ref_values.append(mean_v)
    if mean_x:
        ax.scatter(mean_x, mean_y, **_marker_obs_mean_halo())
        ax.scatter(mean_x, mean_y, **_marker_obs_mean(color=mean_colors))

    if liability_r:
        x_values: list[int] = []
        y_values: list[float] = []
        for i, ptype in enumerate(relationship_types):
            v = liability_r.get(ptype)
            if v is not None:
                x_values.append(i)
                y_values.append(float(v))
        if x_values:
            ax.scatter(x_values, y_values, **_marker_liab())
            ref_values.extend(y_values)

    if parametric_r:
        x_values = []
        y_values = []
        colors: list[str] = []
        for i, ptype in enumerate(relationship_types):
            v = parametric_r.get(ptype)
            if v is not None:
                x_values.append(i)
                y_values.append(float(v))
                colors.append(pair_colors[ptype])
        if x_values:
            ax.scatter(x_values, y_values, **_marker_param(color=colors))
            ref_values.extend(y_values)

    ax.set_xticks(range(len(relationship_types)))
    ax.set_xticklabels(relationship_types, fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", labelsize=11)
    for i, pt in enumerate(relationship_types):
        ax.annotate(
            f"({n_pairs_per_ptype.get(pt, 0) // max(n_reps, 1):,})",
            xy=(i, 0),
            xytext=(0, -28),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            color="0.35",
        )
    ax.set_xlabel("")
    ax.set_xlim(-0.6, len(relationship_types) - 0.4)
    enable_value_gridlines(ax)

    return {"ax": ax, "ref_values": ref_values, "obs_records": obs_records}


def finalize_relationship_type_panels(
    panel_states: list[dict],
    sane_band: tuple[float, float] = PAIR_TYPE_SANE_BAND,
) -> tuple[float, float]:
    """Apply a shared y-limit across all panels and draw observed dots.

    The y-limit is anchored on reference markers (mean observed, liability,
    parametric) plus observed values inside ``sane_band``. Observed
    values outside the band are rendered as small carets at the axis edge so
    one or two low-n outliers don't blow out the panel.
    """
    sane_lo, sane_hi = sane_band
    all_ref: list[float] = []
    all_obs_sane: list[float] = []
    for state in panel_states:
        all_ref.extend(state.get("ref_values", []))
        for _ax, _x, v in state.get("obs_records", []):
            if sane_lo <= v <= sane_hi:
                all_obs_sane.append(v)

    seed = all_ref + all_obs_sane
    if seed:
        ymax = max(seed)
        ymin = min(seed)
        ymin = min(ymin, 0.0)
        span = max(ymax - ymin, 0.05)
        pad = 0.10 * span
        ylim_lo = max(ymin - pad, sane_lo)
        ylim_hi = min(ymax + 1.5 * pad, sane_hi)
    else:
        ylim_lo, ylim_hi = -0.1, 1.1

    for state in panel_states:
        state["ax"].set_ylim(ylim_lo, ylim_hi)

    for state in panel_states:
        ax = state["ax"]
        # Keep clipped carets separate from in-range circles, but batch each shape.
        normal_x: list[float] = []
        normal_y: list[float] = []
        high_x: list[float] = []
        low_x: list[float] = []
        for _ax, x, v in state.get("obs_records", []):
            if v > ylim_hi:
                high_x.append(x)
            elif v < ylim_lo:
                low_x.append(x)
            else:
                normal_x.append(x)
                normal_y.append(v)
        if normal_x:
            ax.scatter(normal_x, normal_y, **_marker_obs_per_rep())
        if high_x:
            ax.scatter(high_x, [ylim_hi] * len(high_x), marker="^", s=42, color="0.45", zorder=5)
        if low_x:
            ax.scatter(low_x, [ylim_lo] * len(low_x), marker="v", s=42, color="0.45", zorder=5)

    return ylim_lo, ylim_hi


def relationship_type_legend_handles(
    has_observed_mean: bool = True,
    has_liability: bool = True,
    has_parametric: bool = False,
) -> list:
    """Return ``Line2D`` proxies for ``fig.legend``.

    Markers match those used by :func:`setup_relationship_type_panel`. Only the
    requested series are included.
    """
    from matplotlib.lines import Line2D

    from simace.plotting.plot_style import COLOR_AFFECTED

    handles = [
        Line2D([0], [0], marker="o", color="0.45", linestyle="None", markersize=7, label="Observed r (per rep)"),
    ]
    if has_observed_mean:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=_WIDE_PLUS,
                color="black",
                linestyle="None",
                markersize=14,
                markeredgewidth=2.2,
                label="Observed r (mean)",
            )
        )
    if has_liability:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="black",
                linestyle="None",
                markersize=9,
                markerfacecolor="white",
                markeredgewidth=1.5,
                label="Liability r (mean)",
            )
        )
    if has_parametric:
        handles.append(
            Line2D([0], [0], marker="*", color=COLOR_AFFECTED, linestyle="None", markersize=16, label="Parametric E[r]")
        )
    return handles
