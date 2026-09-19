"""Pedigree relationship pair counts diagram.

Draws a schematic multi-generational pedigree centred on a highlighted
"Proband" individual.  Each of the 10 relationship types is represented
by colouring the border of the related individual's node and placing a
labelled annotation box nearby.  Mean pair counts (averaged across
replicates) are shown inside each annotation box.

Family structure (4 generations):

  Gen 0  Great-grandparents (GGF + GGM)
  Gen 1  Grandfather + Grandmother  |  Great-uncle (sib of Grandfather)
  Gen 2  Father + Mother  |  Uncle (sib of Father)  |  GU-child
  Gen 3  *Proband*  MZ-twin  Full-sib  Pat-HS  Mat-HS  Cousin  2nd-Cousin
"""

__all__ = ["plot_pedigree_relationship_counts"]

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt

from simace.core.relationships import DEFAULT_MAX_DEGREE
from simace.plotting.plot_style import PEDIGREE_COLORS
from simace.plotting.plot_utils import finalize_plot, save_placeholder_plot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# (x, y, sex)  sex: "M" = male (square), "F" = female (circle)
NODES: dict[str, tuple[float, float, str]] = {
    # Gen 0 — Great-grandparents
    "ggf": (3.0, 10.0, "M"),
    "ggm": (5.0, 10.0, "F"),
    # Gen 1 — Grandparents + great-uncle
    "gf": (2.0, 7.5, "M"),
    "gm": (4.0, 7.5, "F"),
    "great_uncle": (8.5, 7.5, "M"),
    # Gen 2 — Parents, uncle, GU-child
    "father": (1.5, 5.0, "M"),
    "mother": (5.0, 5.0, "F"),
    "uncle": (7.0, 5.0, "M"),
    "gu_child": (10.5, 5.0, "F"),
    # Gen 3 — Proband and relatives
    # pat_hs on father's side (left), mat_hs on mother's side (right)
    "pat_hs": (-1.5, 2.0, "M"),
    "mz_twin": (0.5, 2.0, "M"),
    "proband": (2.5, 2.0, "M"),
    "full_sib": (4.5, 2.0, "F"),
    "mat_hs": (7.0, 2.0, "F"),
    "cousin": (9.5, 2.0, "M"),
    "second_cousin": (12.0, 2.0, "F"),
}

PROBAND_NODE = "proband"
NODE_RADIUS = 0.34

# Marriage lines (double horizontal bar)
MARRIAGES = [
    ("ggf", "ggm"),
    ("gf", "gm"),
    ("father", "mother"),
]

# Descent lines: (parent_spec, children, linestyle)
DESCENTS: list[tuple[tuple[str, str] | str, list[str], str]] = [
    (("ggf", "ggm"), ["gf", "great_uncle"], "solid"),
    (("gf", "gm"), ["father", "uncle"], "solid"),
    (("father", "mother"), ["mz_twin", "proband", "full_sib"], "solid"),
    ("father", ["pat_hs"], "dotted"),
    ("mother", ["mat_hs"], "dotted"),
    ("uncle", ["cousin"], "solid"),
    ("great_uncle", ["gu_child"], "solid"),
    ("gu_child", ["second_cousin"], "solid"),
]

MZ_TWIN_NODES = ("mz_twin", "proband")

# ---------------------------------------------------------------------------
# Relationship → node mapping and label placement
# ---------------------------------------------------------------------------

# Relationship order for consistent colour assignment
RELATIONSHIP_ORDER = [
    "MZ",
    "FS",
    "MHS",
    "PHS",
    "MO",
    "FO",
    "1C",
    "GP",
    "Av",
    "2C",
]

# Map each relationship to the pedigree node representing the other individual
RELATIONSHIP_NODES: dict[str, str] = {
    "MZ": "mz_twin",
    "FS": "full_sib",
    "PHS": "pat_hs",
    "MHS": "mat_hs",
    "FO": "father",
    "MO": "mother",
    "GP": "gf",
    "Av": "uncle",
    "1C": "cousin",
    "2C": "second_cousin",
}

# Label placement relative to each node: (dx, dy, ha, va)
# Most Gen 3 labels go below; Gen 1/2 labels go to the side.
LABEL_OFFSETS: dict[str, tuple[float, float, str, str]] = {
    "MZ": (0.0, -0.55, "center", "top"),
    "FS": (0.0, -0.55, "center", "top"),
    "PHS": (0.0, -0.55, "center", "top"),
    "MHS": (0.0, -0.55, "center", "top"),
    "FO": (0.0, -0.55, "center", "top"),
    "MO": (0.0, -0.55, "center", "top"),
    "GP": (-1.3, 0.55, "center", "bottom"),
    "Av": (0.6, -0.55, "center", "top"),
    "1C": (0.0, -0.55, "center", "top"),
    "2C": (0.0, -0.55, "center", "top"),
}

# Short display names for annotation labels
_SHORT_LABELS: dict[str, str] = {
    "FO": "Father",
    "MO": "Mother",
    "GP": "Grandparent",
    "MHS": "Mat. HS",
    "PHS": "Pat. HS",
    "MZ": "MZ twin",
    "FS": "Full sib",
}

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _mean_stat(all_stats: list[dict[str, Any]], key: str) -> float | None:
    """Return the mean of a top-level numeric stat across replicates, or None."""
    vals = [s[key] for s in all_stats if key in s]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _nc(name: str) -> tuple[float, float]:
    """Node center shorthand."""
    return NODES[name][0], NODES[name][1]


def _draw_node(
    ax: plt.Axes,
    x: float,
    y: float,
    sex: str,
    *,
    fill: str | tuple[float, float, float] = "white",
    edgecolor: str = "black",
    linewidth: float = 0.8,
) -> None:
    r = NODE_RADIUS
    if sex == "M":
        patch = mpatches.FancyBboxPatch(
            (x - r, y - r),
            2 * r,
            2 * r,
            boxstyle="round,pad=0.02",
            facecolor=fill,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )
    else:
        patch = mpatches.Circle(
            (x, y),
            r,
            facecolor=fill,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )
    ax.add_patch(patch)


def _draw_marriage(ax: plt.Axes, a: str, b: str) -> None:
    ax_x, ay_y = _nc(a)
    bx_x, by_y = _nc(b)
    r = NODE_RADIUS
    for offset in [-0.04, 0.04]:
        ax.plot(
            [ax_x + r, bx_x - r],
            [ay_y + offset, by_y + offset],
            color="black",
            linewidth=0.8,
            zorder=1,
        )


def _draw_descent(
    ax: plt.Axes,
    parent_spec: tuple[str, str] | str,
    children: list[str],
    linestyle: str = "solid",
) -> None:
    if isinstance(parent_spec, tuple):
        px = (_nc(parent_spec[0])[0] + _nc(parent_spec[1])[0]) / 2
        py = (_nc(parent_spec[0])[1] + _nc(parent_spec[1])[1]) / 2
    else:
        px, py = _nc(parent_spec)

    child_positions = [_nc(c) for c in children]
    child_y = child_positions[0][1]
    mid_y = (py - NODE_RADIUS + child_y + NODE_RADIUS) / 2
    kw: dict[str, Any] = dict(color="black", linewidth=0.8, linestyle=linestyle, zorder=1)
    ax.plot([px, px], [py - NODE_RADIUS, mid_y], **kw)

    if len(children) == 1:
        cx, cy = child_positions[0]
        ax.plot([px, cx], [mid_y, mid_y], **kw)
        ax.plot([cx, cx], [mid_y, cy + NODE_RADIUS], **kw)
    else:
        xs = [cp[0] for cp in child_positions]
        ax.plot([min(xs), max(xs)], [mid_y, mid_y], **kw)
        for cx, cy in child_positions:
            ax.plot([cx, cx], [mid_y, cy + NODE_RADIUS], **kw)


def _draw_mz_bracket(ax: plt.Axes, a: str, b: str) -> None:
    ax_x, ay_y = _nc(a)
    bx_x, by_y = _nc(b)
    r = NODE_RADIUS
    for offset in [-0.06, 0.06]:
        ax.plot(
            [ax_x + r, bx_x - r],
            [ay_y + offset, by_y + offset],
            color="black",
            linewidth=0.8,
            zorder=2,
        )


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------


def plot_pedigree_relationship_counts(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    stats_key: str = "pair_counts",
    generations_label: str = "",
    max_degree: int | None = DEFAULT_MAX_DEGREE,
    legend_below: bool = False,
    legend_title: str | None = None,
) -> None:
    """Draw a proband-centric pedigree diagram with relationship pair counts.

    Args:
        all_stats: Per-replicate stats dicts.
        output_path: Where to save the figure.
        scenario: Scenario name for the title.
        stats_key: Key in stats dict to read pair counts from.
        generations_label: Label appended to title (e.g. "G_ped = 6").
        max_degree: Extraction depth recorded by the run, named in the legend
            title. If None, omit the depth claim for a legacy artifact. Which
            codes count as computed comes from the data: the report maps an
            uncomputed code to ``None``.
        legend_below: Put the colour key in one horizontal row under the
            diagram instead of a block inside the top right, and tighten the x
            range to match.  Default False keeps the atlas layout.
        legend_title: Heading over the colour key.  Default None keeps
            ``"Relationship (degree <= max_degree)"``, which tells a reader of
            the atlas what the run extracted.  A presentation that has said so
            elsewhere passes the bare word instead.
    """
    output_path = Path(output_path)

    # Check for pair_counts data
    has_data = any(s.get(stats_key) for s in all_stats)
    if not has_data:
        save_placeholder_plot(
            output_path,
            "No pair count data available\n(re-run stats to generate)",
        )
        return

    # Average pair counts across replicates.  A code the run did not compute
    # is ``None`` in the report and simply never enters ``counts``, so it is
    # labelled "not computed" below rather than drawn as a zero.
    counts: dict[str, float] = {}
    reps_per_code: dict[str, int] = {}
    n_reps = 0
    for s in all_stats:
        pc = s.get(stats_key)
        if not pc:
            continue
        n_reps += 1
        for name, cnt in pc.items():
            if cnt is None:
                continue
            counts[name] = counts.get(name, 0) + cnt
            reps_per_code[name] = reps_per_code.get(name, 0) + 1
    counts = {k: v / reps_per_code[k] for k, v in counts.items()}

    # Colour palette (Nature Genetics muted style)
    rel_colors = {name: PEDIGREE_COLORS[name] for name in RELATIONSHIP_ORDER}

    # Build map: node → relationship colour (for node border colouring)
    node_rel_color: dict[str, str] = {}
    for rel_name in RELATIONSHIP_ORDER:
        node = RELATIONSHIP_NODES[rel_name]
        node_rel_color[node] = rel_colors[rel_name]

    # Create figure
    # Poster panel 4 places this in a 246.87 mm cell. At 14 inches wide that
    # is a 0.69 scale and the 10 pt node labels printed at 6.9 pt, the smallest
    # type on the sheet. Sizes below are chosen so the printed result lands near
    # the 12-13 pt the rest of the deck uses; the atlas copy gains the same.
    # The x range runs past the pedigree so the inside legend clears it, and
    # ``legend_below`` reclaims that column. 12.9 is a floor, not a preference:
    # the second-cousin node sits at x = 12.0 and its symbol is a patch, so the
    # axes clip it away if the range ends short, while its text label is not
    # clipped and stays behind as a number with no node. 11.4 did exactly that.
    #
    # Widening past 12.9 is free but pointless. With the key in a row the saved
    # width is set by the legend, not the drawing, so ``bbox_inches`` returns
    # the same pixels for anything from 11.4 to 13.2; a wider range only packs
    # the pedigree into them more tightly.
    #
    # Printed size has one lever here and it is not figsize. ``set_aspect
    # ("equal")`` fixes the drawing's shape, so a larger figure adds margin that
    # ``bbox_inches`` trims straight back off. A consumer pinning the image to a
    # fixed width gets a bigger drawing only from a narrower aspect.
    # Type scale. A narrower drawing is scaled up harder by a consumer that
    # pins the image to a fixed width: 0.78 becomes 0.92 for the poster's
    # 246.87 mm cell. Point sizes are absolute, so every label would print 18%
    # larger while node spacing grew only 10%, and the 1C and 2C count labels
    # ran together. Scaling type by the inverse holds printed sizes where the
    # numbers below put them and gives the extra width to the gaps instead.
    pt = 0.85 if legend_below else 1.0

    _fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-3.0, 12.9 if legend_below else 14.5)
    ax.set_ylim(-0.5, 11.5)
    ax.set_aspect("equal")
    ax.set_axis_off()

    title = "Pedigree Relationship Pair Counts"
    if generations_label:
        title += f"  ({generations_label})"
    ax.set_title(title, fontsize=round(23 * pt), fontweight="bold", pad=16)

    # Generation labels
    for g, y in {0: 10.0, 1: 7.5, 2: 5.0, 3: 2.0}.items():
        ax.text(
            -2.7,
            y,
            f"Gen {g}",
            fontsize=round(15 * pt),
            ha="center",
            va="center",
            fontstyle="italic",
            color="grey",
        )

    # --- Draw structural pedigree elements ---
    for a, b in MARRIAGES:
        _draw_marriage(ax, a, b)
    for parent_spec, children, ls in DESCENTS:
        _draw_descent(ax, parent_spec, children, linestyle=ls)
    _draw_mz_bracket(ax, *MZ_TWIN_NODES)

    # --- Draw nodes ---
    # Related nodes filled with relationship colour; proband highlighted blue.
    for name, (x, y, sex) in NODES.items():
        if name == PROBAND_NODE:
            _draw_node(ax, x, y, sex, fill="black", linewidth=1.4)
        elif name in node_rel_color:
            color = node_rel_color[name]
            # Lighten the colour for the fill (blend with white)
            r, g, b = mcolors.to_rgb(color)
            light = (0.6 + 0.4 * r, 0.6 + 0.4 * g, 0.6 + 0.4 * b)
            _draw_node(
                ax,
                x,
                y,
                sex,
                fill=light,
                edgecolor=color,
                linewidth=1.4,
            )
        else:
            _draw_node(ax, x, y, sex)

    # Proband label
    px, py = _nc(PROBAND_NODE)
    ax.text(
        px,
        py - NODE_RADIUS - 0.25,
        "Proband",
        fontsize=round(19 * pt),
        ha="center",
        va="top",
        fontweight="bold",
    )

    # --- Relationship labels placed directly next to each node ---
    for rel_name in RELATIONSHIP_ORDER:
        node = RELATIONSHIP_NODES[rel_name]
        nx, ny = _nc(node)
        dx, dy, ha, va = LABEL_OFFSETS[rel_name]
        color = rel_colors[rel_name]

        display = _SHORT_LABELS.get(rel_name, rel_name)
        if rel_name in counts:
            label = f"{display}\n({counts[rel_name]:,.0f})"
        else:
            label = f"{display}\nnot computed"

        ax.text(
            nx + dx,
            ny + dy,
            label,
            fontsize=round(18 * pt),
            ha=ha,
            va=va,
            color=color,
            fontweight="bold",
            zorder=5,
            # A white halo, not a bbox. The dotted descent lines pass straight
            # through the Father and Mother counts, and at this size that reads
            # as a strikethrough; a box round every label would be worse.
            path_effects=[mpatheffects.withStroke(linewidth=4, foreground="white")],
        )

    # Legend
    # A colour key, nothing more. Every count is printed beside its own node, so
    # repeating them here bought nothing and made the legend wide enough to grow
    # into the pedigree. The "not computed" suffix stays: it is the one thing the
    # legend can say that a missing number cannot, and it appears only when a
    # code falls outside the requested degree.
    handles = [
        mpatches.Patch(color=rel_colors[n], label=n if n in counts else f"{n} (not computed)")
        for n in RELATIONSHIP_ORDER
    ]
    if legend_title is None:
        key_title = "Relationship" if max_degree is None else f"Relationship (degree ≤ {max_degree})"
    else:
        key_title = legend_title
    if legend_below:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=10,
            fontsize=round(15 * pt),
            title=key_title,
            title_fontsize=round(16 * pt),
            frameon=False,
            columnspacing=1.1,
            handletextpad=0.4,
        )
    else:
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=round(15 * pt),
            title=key_title,
            title_fontsize=round(16 * pt),
        )

    # Population metadata annotation
    if stats_key == "pair_counts_ped":
        n_ind_key, n_gen_key = "n_individuals_ped", "n_generations_ped"
    else:
        n_ind_key, n_gen_key = "n_individuals", "n_generations"
    mean_n_ind = _mean_stat(all_stats, n_ind_key)
    mean_n_gen = _mean_stat(all_stats, n_gen_key)

    footer_parts = [f"Mean across {n_reps} replicate{'s' if n_reps != 1 else ''}"]
    if mean_n_gen is not None:
        footer_parts.append(f"{round(mean_n_gen)} generations")
    if mean_n_ind is not None:
        footer_parts.append(f"{round(mean_n_ind):,} individuals")
    ax.text(
        0.99,
        0.01,
        "  |  ".join(footer_parts),
        transform=ax.transAxes,
        fontsize=round(13 * pt),
        ha="right",
        va="bottom",
        color="grey",
    )

    finalize_plot(output_path, scenario=scenario)
    logger.info("Pedigree counts plot saved to %s", output_path)


def cli() -> None:
    """Command-line interface for pedigree relationship counts plot."""
    from simace.core.cli_base import add_logging_args, init_logging
    from simace.core.yaml_io import load_yaml

    parser = argparse.ArgumentParser(description="Plot pedigree relationship pair counts diagram")
    add_logging_args(parser)
    parser.add_argument("--stats", nargs="+", required=True, help="Stats YAML paths")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--scenario", default="", help="Scenario name for title")
    args = parser.parse_args()
    init_logging(args)

    all_stats = [load_yaml(p) for p in args.stats]

    plot_pedigree_relationship_counts(all_stats, args.output, args.scenario)
