"""Central visual style for Nature Genetics-inspired plots."""

from __future__ import annotations

__all__ = [
    "CENSORING_COLORS",
    "COLOR_AFFECTED",
    "COLOR_FEMALE",
    "COLOR_MALE",
    "COLOR_OBSERVED",
    "COLOR_TRUE",
    "COLOR_UNAFFECTED",
    "PAIR_COLORS",
    "add_scenario_label",
    "apply_nature_style",
    "enable_value_gridlines",
]

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Color palettes — muted Okabe-Ito-inspired, colorblind-safe
# ---------------------------------------------------------------------------

PAIR_COLORS: dict[str, str] = {
    "MZ": "#4477AA",  # muted blue
    "FS": "#EE6677",  # muted rose
    "MO": "#228833",  # muted green
    "FO": "#CCBB44",  # muted olive
    "MHS": "#66CCEE",  # muted cyan
    "PHS": "#AA3377",  # muted purple
    "1C": "#BBBBBB",  # neutral grey
}

# Extended palette for pedigree diagrams (10 relationship types)
PEDIGREE_COLORS: dict[str, str] = {
    **PAIR_COLORS,
    "GP": "#CC6633",   # muted brown
    "Av": "#999933",   # muted dark olive
    "2C": "#DDDDDD",   # light grey
}

# Affected / unaffected status
COLOR_AFFECTED = "#CC4444"    # muted red
COLOR_UNAFFECTED = "#999999"  # medium grey

# Sex encoding
COLOR_FEMALE = "#44AA99"  # teal
COLOR_MALE = "#EE7733"    # orange

# Observed vs true data lines
COLOR_OBSERVED = "#4477AA"  # blue (matches MZ pair color)
COLOR_TRUE = "#888888"      # grey

# Censoring cascade
CENSORING_COLORS: dict[str, str] = {
    "observed": "#228833",  # green
    "death": "#CC4444",     # red
    "right": "#AA3377",     # purple
    "left": "#EE7733",      # orange
}

# Secondary data-series colors (for bar charts, regression lines, etc.)
COLOR_SERIES_1 = "#4477AA"  # blue
COLOR_SERIES_2 = "#EE6677"  # rose
COLOR_SERIES_3 = "#228833"  # green


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_nature_style() -> None:
    """Set matplotlib rcParams for Nature Genetics-inspired figures."""
    mpl.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Legend
        "legend.frameon": False,
        "legend.fontsize": 9,
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def enable_value_gridlines(ax) -> None:
    """Add faint horizontal gridlines for plots where absolute values matter."""
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.15, color="0.7", zorder=0)
    ax.set_axisbelow(True)


def add_scenario_label(fig, scenario: str) -> None:
    """Add small grey italic scenario label at the bottom-right of a figure."""
    if scenario:
        fig.text(
            0.99,
            0.005,
            scenario,
            fontsize=11,
            color="0.5",
            ha="right",
            va="bottom",
            fontstyle="italic",
            transform=fig.transFigure,
        )
