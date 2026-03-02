"""Pipeline DAG diagram for the atlas title page.

Renders a single-page figure showing the Snakemake pipeline structure with
each step's relevant parameters displayed inside its box.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAG structure
# ---------------------------------------------------------------------------

# Each step: (key, display_title, color, param_names)
_PIPELINE_STEPS: list[tuple[str, str, str, list[str]]] = [
    (
        "simulate",
        "Simulate pedigree liability",
        "#D4E6F1",
        [
            "N", "G_sim", "G_ped", "fam_size", "p_mztwin", "p_nonsocial_father",
            "_ACE1", "_ACE2", "_rAC",
        ],
    ),
    (
        "phenotype_weibull",
        "Phenotype (survival)",
        "#D5F5E3",
        ["G_pheno", "_weibull1", "_weibull2"],
    ),
    (
        "censor",
        "Censor",
        "#FCF3CF",
        ["censor_age", "gen_censoring", "_mortality"],
    ),
    (
        "phenotype_threshold",
        "Phenotype (threshold)",
        "#E8DAEF",
        ["G_pheno", "_prev12"],
    ),
    (
        "sample_weibull",
        "Sample",
        "#EAECEE",
        ["N_sample"],
    ),
    (
        "sample_threshold",
        "Sample",
        "#EAECEE",
        ["N_sample"],
    ),
]

# Edges as (source_key, target_key)
_PIPELINE_EDGES: list[tuple[str, str]] = [
    ("simulate", "phenotype_weibull"),
    ("simulate", "phenotype_threshold"),
    ("phenotype_weibull", "censor"),
    ("censor", "sample_weibull"),
    ("phenotype_threshold", "sample_threshold"),
]

# Step positions in data coordinates (cx, cy)
_STEP_POSITIONS: dict[str, tuple[float, float]] = {
    "simulate":             (0.27, 0.84),
    "phenotype_weibull":    (0.27, 0.52),
    "phenotype_threshold":  (0.73, 0.52),
    "censor":               (0.27, 0.28),
    "sample_weibull":       (0.27, 0.09),
    "sample_threshold":     (0.73, 0.09),
}

# Publication-friendly display names for parameters
_PARAM_DISPLAY: dict[str, str] = {
    "G_sim": "G\u209C\u2092\u209C\u2090\u2097",  # G_total via subscript
    "G_ped": "G\u1D68\u2091\u209C",  # not great with subscripts, use plain
}
# Simpler approach: just use a clean mapping
_PARAM_DISPLAY = {
    "N": "N",
    "G_sim": "G_total",
    "G_ped": "G_pedigree",
    "fam_size": "mean family size",
    "p_mztwin": "p_mztwin",
    "p_nonsocial_father": "p_nonsocial_father",
    "G_pheno": "G_pheno",
    "censor_age": "max age",
    "gen_censoring": "gen. windows",
    "N_sample": "N_sample",
}

# ---------------------------------------------------------------------------
# Font sizes (pts) — single place to tune readability
# ---------------------------------------------------------------------------
_FONT_TITLE = 24       # scenario title at top of page
_FONT_BOX_TITLE = 14   # step name inside each box
_FONT_TABLE = 11       # parameter names and values
_FONT_META = 12        # seed / replicates in scenario area


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _display_name(name: str) -> str:
    """Return a publication-friendly display name for a parameter."""
    return _PARAM_DISPLAY.get(name, name)


def _format_param_value(name: str, value: object) -> str:
    """Format a single parameter value compactly."""
    if isinstance(value, dict):
        return "(see params)"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e6:
            return str(int(value))
        return f"{value:g}"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _get_param_rows(
    param_names: list[str], params: dict,
) -> list[tuple[str, str]]:
    """Return (display_name, formatted_value) pairs for the step's parameters."""
    rows = []
    for name in param_names:
        # Compact variance-component rows: [A, C, E] per trait
        if name == "_ACE1" and "A1" in params:
            a = float(params.get("A1", 0))
            c = float(params.get("C1", 0))
            e = 1.0 - a - c
            rows.append(("[A1, C1, E1]", f"[{a:g}, {c:g}, {e:g}]"))
            continue
        if name == "_ACE2" and "A2" in params:
            a = float(params.get("A2", 0))
            c = float(params.get("C2", 0))
            e = 1.0 - a - c
            rows.append(("[A2, C2, E2]", f"[{a:g}, {c:g}, {e:g}]"))
            continue
        if name == "_rAC" and "rA" in params:
            ra = float(params.get("rA", 0))
            rc = float(params.get("rC", 0))
            rows.append(("[rA, rC]", f"[{ra:g}, {rc:g}]"))
            continue
        # Compact Weibull params per trait: [beta, scale, rho]
        if name == "_weibull1" and "beta1" in params:
            b = _format_param_value("beta1", params["beta1"])
            s = _format_param_value("scale1", params["scale1"])
            r = _format_param_value("rho1", params["rho1"])
            rows.append(("trait 1 [\u03b2, scale, \u03c1]", f"[{b}, {s}, {r}]"))
            continue
        if name == "_weibull2" and "beta2" in params:
            b = _format_param_value("beta2", params["beta2"])
            s = _format_param_value("scale2", params["scale2"])
            r = _format_param_value("rho2", params["rho2"])
            rows.append(("trait 2 [\u03b2, scale, \u03c1]", f"[{b}, {s}, {r}]"))
            continue
        # Compact mortality: [scale, rho]
        if name == "_mortality" and "death_scale" in params:
            s = _format_param_value("death_scale", params["death_scale"])
            r = _format_param_value("death_rho", params["death_rho"])
            rows.append(("mortality [scale, \u03c1]", f"[{s}, {r}]"))
            continue
        # Compact prevalence: one row per trait (scalar or dict)
        if name == "_prev12":
            for t, key in [(1, "prevalence1"), (2, "prevalence2")]:
                if key in params:
                    rows.append((f"prevalence {t}", _format_param_value(key, params[key])))
            continue
        if name not in params:
            continue
        val = params[name]
        formatted = _format_param_value(name, val)
        if name == "N_sample" and (val == 0 or val == "0"):
            formatted += " (all)"
        rows.append((_display_name(name), formatted))
    return rows


def _draw_step_box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    title: str,
    rows: list[tuple[str, str]],
    color: str,
    dimmed: bool = False,
) -> None:
    """Draw a rounded box with title and a name/value parameter table."""
    alpha = 0.35 if dimmed else 1.0
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=color,
        edgecolor="0.55" if dimmed else "0.3",
        linewidth=1.6,
        linestyle="--" if dimmed else "-",
        alpha=alpha,
    )
    ax.add_patch(box)

    text_color = "0.55" if dimmed else "black"
    name_color = "0.6" if dimmed else "0.25"

    title_h = 0.048  # height reserved for the title area
    top = cy + h / 2

    # Title centred in the title area
    ax.text(
        cx, top - title_h / 2, title,
        fontsize=_FONT_BOX_TITLE, fontweight="bold",
        fontfamily="sans-serif", color=text_color,
        ha="center", va="center",
    )

    if not rows:
        return

    # Horizontal separator between title and table body
    sep_y = top - title_h
    pad = 0.012  # inset from box edges
    ax.plot(
        [cx - w / 2 + pad, cx + w / 2 - pad], [sep_y, sep_y],
        color="0.65" if dimmed else "0.45", linewidth=0.8, clip_on=False,
    )

    # Table layout — position value column based on longest name
    row_h = 0.024
    char_w = 0.0085  # approx data-units per character at 11pt mono on 11in fig
    max_name_len = max(len(n) for n, _ in rows)
    name_x = cx - w / 2 + pad + 0.005  # left-aligned names
    val_x = name_x + (max_name_len + 2) * char_w  # 2-char gap after names

    for i, (name, val) in enumerate(rows):
        row_y = sep_y - (i + 0.55) * row_h
        ax.text(
            name_x, row_y, name,
            fontsize=_FONT_TABLE, fontfamily="monospace", color=name_color,
            ha="left", va="center",
        )
        ax.text(
            val_x, row_y, val,
            fontsize=_FONT_TABLE, fontweight="bold", fontfamily="monospace",
            color=text_color,
            ha="left", va="center",
        )


def _draw_pipeline_arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    dimmed: bool = False,
) -> None:
    """Draw an arrow between two points."""
    color = "0.65" if dimmed else "0.35"
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.6,head_width=0.3",
            color=color,
            linewidth=2.0,
            linestyle="--" if dimmed else "-",
            shrinkA=0,
            shrinkB=0,
        ),
    )


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def render_pipeline_figure(
    params: dict,
    scenario: str = "",
) -> plt.Figure:
    """Build and return the pipeline DAG figure (without saving).

    Args:
        params: Merged scenario parameters dict.
        scenario: Scenario name for the title.

    Returns:
        The matplotlib Figure object.
    """
    sampling_active = True  # always show sample boxes normally

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Scenario area — right column, aligned with simulate box
    if scenario:
        fig.text(
            0.71, 0.92, "Scenario",
            fontsize=_FONT_TABLE, fontfamily="sans-serif", color="0.4",
            ha="center", va="bottom",
        )
        fig.text(
            0.71, 0.91, scenario,
            fontsize=_FONT_TITLE, fontweight="bold", fontfamily="sans-serif",
            ha="center", va="top",
        )
    # Seed + replicates below scenario name
    meta_parts = []
    seed = params.get("seed")
    if seed is not None:
        meta_parts.append(f"seed = {seed}")
    reps = params.get("replicates")
    if reps is not None:
        meta_parts.append(f"replicates = {reps}")
    std = params.get("standardize")
    if std is not None:
        meta_parts.append(f"standardize = {str(std).lower()}")
    if meta_parts:
        fig.text(
            0.71, 0.82, "\n".join(meta_parts),
            fontsize=_FONT_META, fontfamily="monospace", color="0.4",
            ha="center", va="top", linespacing=1.5,
        )

    # Build step info lookup
    step_info = {}
    for key, display, color, pnames in _PIPELINE_STEPS:
        step_info[key] = (display, color, pnames)

    # Build rows for each step and compute box sizes
    step_rows: dict[str, list[tuple[str, str]]] = {}
    box_sizes: dict[str, tuple[float, float]] = {}
    max_w = 0.22
    for key in _STEP_POSITIONS:
        display, _, pnames = step_info[key]
        rows = _get_param_rows(pnames, params)
        step_rows[key] = rows
        # Width: based on longest name+value pair
        max_name = max((len(n) for n, _ in rows), default=0) if rows else 0
        max_val = max((len(v) for _, v in rows), default=0) if rows else 0
        char_w = 0.0085  # approx data-units per character at 11pt mono on 11in fig
        table_w = 0.046 + (max_name + 2 + max_val) * char_w
        title_w = len(display) * 0.010 + 0.04
        w = max(0.22, table_w, title_w)
        max_w = max(max_w, w)
        # Height: title area + rows
        h = 0.055 + 0.024 * max(len(rows), 1)
        box_sizes[key] = (w, h)
    # Uniform width across all boxes
    for key in box_sizes:
        _, h = box_sizes[key]
        box_sizes[key] = (max_w, h)

    # Determine which steps/edges are dimmed (sample when inactive)
    sample_keys = {"sample_weibull", "sample_threshold"}

    # Draw arrows first (behind boxes)
    for src, dst in _PIPELINE_EDGES:
        sx, sy = _STEP_POSITIONS[src]
        dx, dy = _STEP_POSITIONS[dst]
        _, sh = box_sizes[src]
        _, dh = box_sizes[dst]
        dimmed = (dst in sample_keys) and not sampling_active
        _draw_pipeline_arrow(ax, sx, sy - sh / 2, dx, dy + dh / 2, dimmed=dimmed)

    # Draw boxes
    for key, (cx, cy) in _STEP_POSITIONS.items():
        display, color, _ = step_info[key]
        w, h = box_sizes[key]
        dimmed = (key in sample_keys) and not sampling_active
        _draw_step_box(ax, cx, cy, w, h, display, step_rows[key], color, dimmed=dimmed)

    return fig


def plot_pipeline(
    params: dict,
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Render the pipeline DAG diagram and save to file.

    Args:
        params: Merged scenario parameters dict.
        output_path: Where to save the figure.
        scenario: Scenario name for the title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = render_pipeline_figure(params, scenario=scenario)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pipeline diagram saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli() -> None:
    """Command-line interface for standalone pipeline diagram rendering."""
    from sim_ace.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(
        description="Render pipeline DAG diagram with scenario parameters.",
    )
    add_logging_args(parser)
    parser.add_argument(
        "--params", required=True,
        help="Path to params.yaml (merged scenario parameters).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output image path (e.g. /tmp/pipeline.png).",
    )
    parser.add_argument(
        "--scenario", default="",
        help="Scenario name for the title.",
    )
    args = parser.parse_args()
    init_logging(args)

    try:
        _yaml_loader = yaml.CSafeLoader
    except AttributeError:
        _yaml_loader = yaml.SafeLoader

    with open(args.params, encoding="utf-8") as fh:
        params = yaml.load(fh, Loader=_yaml_loader)

    plot_pipeline(params, args.output, scenario=args.scenario)


if __name__ == "__main__":
    cli()
