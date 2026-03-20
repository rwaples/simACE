"""Render an epidemiological Table 1 summarising the simulated study population."""

from __future__ import annotations

import logging
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------


def _fmt_int(v) -> str:
    """Format an integer with thousands separator, or '—' if None."""
    if v is None:
        return "—"
    return f"{round(v):,}"


def _fmt_pct(v) -> str:
    """Format a fraction as a percentage string, or '—'."""
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _fmt_f(v, decimals: int = 2) -> str:
    """Format a float to *decimals* places, or '—'."""
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _fmt_range(vals: list, fmt_func=_fmt_int) -> str:
    """Format as 'mean [min–max]' when len > 1, else just the value."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return "—"
    if len(clean) == 1:
        return fmt_func(clean[0])
    m = mean(clean)
    return f"{fmt_func(m)} [{fmt_func(min(clean))}\u2013{fmt_func(max(clean))}]"


def _fmt_range_pct(vals: list) -> str:
    return _fmt_range(vals, fmt_func=_fmt_pct)


def _fmt_range_f(vals: list, decimals: int = 2) -> str:
    return _fmt_range(vals, fmt_func=lambda v: _fmt_f(v, decimals))


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _safe_get(d: dict, *keys, default=None):
    """Nested dict get."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _compute_aoo_quartiles(ci_data: dict) -> dict:
    """From a cumulative incidence dict, interpolate ages at 25/50/75% of final."""
    ages = ci_data.get("ages")
    # Support both 'incidence' and 'observed_values' key names
    inc = ci_data.get("incidence") or ci_data.get("observed_values")
    if not ages or not inc:
        return {"q1": None, "median": None, "q3": None}
    ages = np.asarray(ages, dtype=float)
    inc = np.asarray(inc, dtype=float)
    final = inc[-1] if len(inc) > 0 else 0
    if final < 1e-6:
        return {"q1": None, "median": None, "q3": None}
    result = {}
    for label, frac in [("q1", 0.25), ("median", 0.50), ("q3", 0.75)]:
        target = frac * final
        idx = np.searchsorted(inc, target)
        result[label] = float(ages[min(idx, len(ages) - 1)])
    return result


def _sex_n(stats: dict, trait: str = "trait1"):
    """Extract female/male n from cumulative_incidence_by_sex."""
    by_sex = _safe_get(stats, "cumulative_incidence_by_sex", trait, default={})
    f_n = _safe_get(by_sex, "female", "n")
    m_n = _safe_get(by_sex, "male", "n")
    return f_n, m_n


def _aggregate_cascade(all_stats: list[dict], trait: str):
    """Aggregate censoring cascade across generations and reps."""
    death_pcts, right_pcts, left_pcts = [], [], []
    for stats in all_stats:
        cascade = _safe_get(stats, "censoring_cascade", trait, default={})
        total_affected = 0
        total_death = 0
        total_right = 0
        total_left = 0
        for gen_data in cascade.values():
            if not isinstance(gen_data, dict):
                continue
            ta = gen_data.get("true_affected", 0)
            total_affected += ta
            total_death += gen_data.get("death_censored", 0)
            total_right += gen_data.get("right_censored", 0)
            total_left += gen_data.get("left_truncated", 0)
        if total_affected > 0:
            death_pcts.append(total_death / total_affected)
            right_pcts.append(total_right / total_affected)
            left_pcts.append(total_left / total_affected)
    return death_pcts, right_pcts, left_pcts


# ---------------------------------------------------------------------------
# Table drawing
# ---------------------------------------------------------------------------

# Layout constants (normalised figure coords on 11 × 8.5 landscape)
_LEFT = 0.04  # left margin for labels
_COL1 = 0.52  # column 1 (Trait 1) x position
_COL2 = 0.76  # column 2 (Trait 2) x position
_RIGHT = 0.96  # right edge for single-value column
_ROW_H = 0.021  # row height as fraction of figure height
_FONT = "sans-serif"
_FONT_SIZE = 8.0
_HEADER_SIZE = 9.5
_TITLE_SIZE = 14


def _draw_row_bg(ax, y: float, shade: bool) -> None:
    """Draw alternating row background."""
    if shade:
        ax.add_patch(
            FancyBboxPatch(
                (_LEFT - 0.01, y - _ROW_H * 0.35),
                _RIGHT - _LEFT + 0.02,
                _ROW_H * 0.85,
                boxstyle="square,pad=0",
                facecolor="#f0f0f0",
                edgecolor="none",
                transform=ax.figure.transFigure,
                clip_on=False,
            )
        )


def _draw_section_header(fig, y: float, text: str) -> float:
    """Draw a bold section header and return updated y."""
    fig.text(
        _LEFT,
        y,
        text,
        fontsize=_HEADER_SIZE,
        fontweight="bold",
        fontfamily=_FONT,
        va="top",
        transform=fig.transFigure,
    )
    return y - _ROW_H * 1.2


def _draw_row(
    fig, ax, y: float, label: str, value: str, shade: bool, color: str = "black",
) -> float:
    """Draw a single-value row (label on left, value on right)."""
    _draw_row_bg(ax, y, shade)
    fig.text(
        _LEFT + 0.02,
        y,
        label,
        fontsize=_FONT_SIZE,
        fontfamily=_FONT,
        color=color,
        va="top",
        transform=fig.transFigure,
    )
    fig.text(
        _RIGHT,
        y,
        value,
        fontsize=_FONT_SIZE,
        fontfamily=_FONT,
        color=color,
        va="top",
        ha="right",
        transform=fig.transFigure,
    )
    return y - _ROW_H


def _draw_row2(fig, ax, y: float, label: str, v1: str, v2: str, shade: bool) -> float:
    """Draw a two-column row (Trait 1 / Trait 2)."""
    _draw_row_bg(ax, y, shade)
    fig.text(
        _LEFT + 0.02,
        y,
        label,
        fontsize=_FONT_SIZE,
        fontfamily=_FONT,
        va="top",
        transform=fig.transFigure,
    )
    fig.text(
        _COL1 + 0.10,
        y,
        v1,
        fontsize=_FONT_SIZE,
        fontfamily=_FONT,
        va="top",
        ha="right",
        transform=fig.transFigure,
    )
    fig.text(
        _COL2 + 0.10,
        y,
        v2,
        fontsize=_FONT_SIZE,
        fontfamily=_FONT,
        va="top",
        ha="right",
        transform=fig.transFigure,
    )
    return y - _ROW_H


def _draw_col_headers(fig, y: float) -> float:
    """Draw Trait 1 / Trait 2 column headers."""
    fig.text(
        _COL1 + 0.10,
        y,
        "Trait 1",
        fontsize=_FONT_SIZE,
        fontweight="bold",
        fontfamily=_FONT,
        va="top",
        ha="right",
        transform=fig.transFigure,
    )
    fig.text(
        _COL2 + 0.10,
        y,
        "Trait 2",
        fontsize=_FONT_SIZE,
        fontweight="bold",
        fontfamily=_FONT,
        va="top",
        ha="right",
        transform=fig.transFigure,
    )
    return y - _ROW_H


# ---------------------------------------------------------------------------
# Main rendering function
# ---------------------------------------------------------------------------


def render_table1_figure(
    all_stats: list[dict],
    scenario_params: dict,
    scenario: str = "",
) -> plt.Figure:
    """Build and return the Table 1 figure (11 x 8.5 landscape).

    Args:
        all_stats: List of phenotype_stats dicts, one per replicate.
        scenario_params: Merged scenario config parameters.
        scenario: Scenario name for the title.

    Returns:
        matplotlib Figure ready for ``pdf.savefig()``.
    """
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    p = scenario_params
    n_reps = len(all_stats)

    # ── Title ──────────────────────────────────────────────────────────
    title = f"Table 1.  Study Population Characteristics — {scenario}"
    fig.text(
        0.50,
        0.96,
        title,
        fontsize=_TITLE_SIZE,
        fontweight="bold",
        fontfamily="serif",
        ha="center",
        va="top",
        transform=fig.transFigure,
    )
    # Thin rule below title
    fig.add_artist(
        plt.Line2D([_LEFT, _RIGHT], [0.935, 0.935], color="black", lw=0.8, transform=fig.transFigure, clip_on=False)
    )

    y = 0.89
    shade = False

    # ── A. Population ─────────────────────────────────────────────────
    y = _draw_section_header(fig, y, "A.  Population")
    shade = False

    # Deterministic values (constant across reps) — use first rep directly
    s0 = all_stats[0]
    n_ind = s0.get("n_individuals")
    n_ped = s0.get("n_individuals_ped")
    n_gen = s0.get("n_generations")

    mz_pairs = [_safe_get(s, "pair_counts", "MZ twin") for s in all_stats]

    y = _draw_row(fig, ax, y, "Total phenotyped individuals, n", _fmt_int(n_ind), True)
    y = _draw_row(fig, ax, y, "Full pedigree individuals, n", _fmt_int(n_ped), False)
    y = _draw_row(fig, ax, y, "Generations observed", str(n_gen) if n_gen else "\u2014", True)

    # Female/male with percentages — also constant across reps
    f_n = _sex_n(s0, "trait1")[0]
    m_n = _sex_n(s0, "trait1")[1]
    if f_n is not None and n_ind:
        f_str = f"{_fmt_int(f_n)} ({_fmt_pct(f_n / n_ind)})"
        m_str = f"{_fmt_int(m_n)} ({_fmt_pct(m_n / n_ind)})" if m_n else "\u2014"
    else:
        f_str = _fmt_int(f_n)
        m_str = _fmt_int(m_n)

    y = _draw_row(fig, ax, y, "  Female, n (%)", f_str, False)
    y = _draw_row(fig, ax, y, "  Male, n (%)", m_str, True)
    y = _draw_row(fig, ax, y, "MZ twin pairs, n", _fmt_range(mz_pairs), False)

    # Mean family size (overall + by sex)
    fam_means = [_safe_get(s, "family_size", "mean") for s in all_stats]
    fam_f = [_safe_get(s, "family_size", "mean_female") for s in all_stats]
    fam_m = [_safe_get(s, "family_size", "mean_male") for s in all_stats]
    y = _draw_row(fig, ax, y, "Mean family size", _fmt_range_f(fam_means, 2), True)
    y = _draw_row(fig, ax, y, "  Female offspring per family", _fmt_range_f(fam_f, 2), False)
    y = _draw_row(fig, ax, y, "  Male offspring per family", _fmt_range_f(fam_m, 2), True)

    y = _draw_row(
        fig,
        ax,
        y,
        "Maximum follow-up age",
        f"{p.get('censor_age', '—')} years",
        False,
    )

    # Person-years
    total_py = [_safe_get(s, "person_years", "total") for s in all_stats]
    if any(v is not None for v in total_py):
        y = _draw_row(fig, ax, y, "Total person-years of follow-up", _fmt_range(total_py), True)

    # Sampling info — always shown, grayed out when defaults
    n_sample = p.get("N_sample", 0)
    car = p.get("case_ascertainment_ratio", 1.0)
    sample_active = n_sample and n_sample > 0
    car_active = car != 1.0
    sample_color = "black" if sample_active else "0.55"
    car_color = "black" if car_active else "0.55"
    sample_val = _fmt_int(n_sample) if sample_active else "none (full population)"
    car_val = f"{car:.1f}\u00d7" if car_active else "1.0\u00d7 (no enrichment)"
    y = _draw_row(fig, ax, y, "Sampled individuals, n", sample_val, False, color=sample_color)
    y = _draw_row(fig, ax, y, "Case ascertainment ratio", car_val, True, color=car_color)

    y -= _ROW_H * 0.4

    # ── B. Disease Characteristics ────────────────────────────────────
    y = _draw_section_header(fig, y, "B.  Disease Characteristics")
    y = _draw_col_headers(fig, y)
    shade = False

    # Prevalence
    prev1 = [_safe_get(s, "prevalence", "trait1") for s in all_stats]
    prev2 = [_safe_get(s, "prevalence", "trait2") for s in all_stats]
    shade = not shade
    y = _draw_row2(
        fig,
        ax,
        y,
        "Observed prevalence",
        _fmt_range_pct(prev1),
        _fmt_range_pct(prev2),
        shade,
    )

    # Sex-specific prevalence
    fprev1 = [_safe_get(s, "cumulative_incidence_by_sex", "trait1", "female", "prevalence") for s in all_stats]
    mprev1 = [_safe_get(s, "cumulative_incidence_by_sex", "trait1", "male", "prevalence") for s in all_stats]
    fprev2 = [_safe_get(s, "cumulative_incidence_by_sex", "trait2", "female", "prevalence") for s in all_stats]
    mprev2 = [_safe_get(s, "cumulative_incidence_by_sex", "trait2", "male", "prevalence") for s in all_stats]
    shade = not shade
    y = _draw_row2(fig, ax, y, "  Prevalence, female", _fmt_range_pct(fprev1), _fmt_range_pct(fprev2), shade)
    shade = not shade
    y = _draw_row2(fig, ax, y, "  Prevalence, male", _fmt_range_pct(mprev1), _fmt_range_pct(mprev2), shade)

    # Affected n
    aff1 = []
    aff2 = []
    for s in all_stats:
        ja = s.get("joint_affection", {}).get("counts", {})
        aff1.append(ja.get("both", 0) + ja.get("trait1_only", 0))
        aff2.append(ja.get("both", 0) + ja.get("trait2_only", 0))
    shade = not shade
    y = _draw_row2(fig, ax, y, "Affected, n", _fmt_range(aff1), _fmt_range(aff2), shade)

    # Incidence rate (per 1,000 person-years at risk)
    py_t1 = [_safe_get(s, "person_years", "trait1") for s in all_stats]
    py_t2 = [_safe_get(s, "person_years", "trait2") for s in all_stats]

    def _incidence_rate(affected_list, py_list):
        rates = []
        for a, py in zip(affected_list, py_list):
            if a is not None and py and py > 0:
                rates.append(a / py * 1000)
            else:
                rates.append(None)
        return rates

    ir1 = _incidence_rate(aff1, py_t1)
    ir2 = _incidence_rate(aff2, py_t2)
    shade = not shade
    y = _draw_row2(
        fig,
        ax,
        y,
        "Incidence rate (per 1,000 PY)",
        _fmt_range_f(ir1, 1),
        _fmt_range_f(ir2, 1),
        shade,
    )

    # Age at onset quartiles
    def _aoo_quartile(all_stats, trait, key):
        vals = []
        for s in all_stats:
            ci = _safe_get(s, "cumulative_incidence", trait, default={})
            q = _compute_aoo_quartiles(ci)
            if q[key] is not None:
                vals.append(q[key])
        return _fmt_range_f(vals, 1)

    shade = not shade
    y = _draw_row2(
        fig,
        ax,
        y,
        "Age at onset, Q1",
        _aoo_quartile(all_stats, "trait1", "q1"),
        _aoo_quartile(all_stats, "trait2", "q1"),
        shade,
    )
    shade = not shade
    y = _draw_row2(
        fig,
        ax,
        y,
        "Age at onset, median",
        _aoo_quartile(all_stats, "trait1", "median"),
        _aoo_quartile(all_stats, "trait2", "median"),
        shade,
    )
    shade = not shade
    y = _draw_row2(
        fig,
        ax,
        y,
        "Age at onset, Q3",
        _aoo_quartile(all_stats, "trait1", "q3"),
        _aoo_quartile(all_stats, "trait2", "q3"),
        shade,
    )

    # Co-affected
    coaff = [_safe_get(s, "joint_affection", "counts", "both", default=0) for s in all_stats]
    coaff_pct = [_safe_get(s, "joint_affection", "proportions", "both") for s in all_stats]
    shade = not shade
    coaff_str = f"{_fmt_range(coaff)} ({_fmt_range_pct(coaff_pct)})"
    y = _draw_row2(fig, ax, y, "Co-affected, n (%)", coaff_str, "", shade)

    y -= _ROW_H * 0.4

    # ── C. Censoring ──────────────────────────────────────────────────
    y = _draw_section_header(fig, y, "C.  Censoring")
    shade = False

    # Per-generation rows: N, window, sensitivity
    # Use trait1 cascade as reference (windows are the same for both traits)
    cascade0 = _safe_get(all_stats[0], "censoring_cascade", "trait1", default={})
    gen_keys = sorted(cascade0.keys()) if cascade0 else []
    for gk in gen_keys:
        # n_gen is deterministic — use first rep
        gen_n = _safe_get(s0, "censoring_cascade", "trait1", gk, "n_gen")
        window = _safe_get(s0, "censoring_cascade", "trait1", gk, "window")
        n_str = _fmt_int(gen_n)
        win_str = f"ages {window[0]:.0f}\u2013{window[1]:.0f}" if window else ""
        shade = not shade
        y = _draw_row(
            fig,
            ax,
            y,
            f"  {gk}:  n={n_str},  {win_str}",
            "",
            shade,
        )

    # Overall mortality rate per 1,000 person-years
    mort_per_1k = []
    for s in all_stats:
        deaths = _safe_get(s, "person_years", "deaths")
        total = _safe_get(s, "person_years", "total")
        if deaths is not None and total and total > 0:
            mort_per_1k.append(deaths / total * 1000)
    if mort_per_1k:
        shade = not shade
        y = _draw_row(
            fig,
            ax,
            y,
            "Mortality rate (per 1,000 PY)",
            _fmt_range_f(mort_per_1k, 1),
            shade,
        )

    # Bottom rule
    fig.add_artist(
        plt.Line2D(
            [_LEFT, _RIGHT], [y - 0.005, y - 0.005], color="black", lw=0.8, transform=fig.transFigure, clip_on=False
        )
    )

    # Footnote
    if n_reps > 1:
        fig.text(
            _LEFT,
            y - 0.02,
            f"Values are mean [min\u2013max] across {n_reps} replicates where applicable.",
            fontsize=7,
            fontfamily=_FONT,
            color="0.4",
            va="top",
            transform=fig.transFigure,
        )

    return fig
