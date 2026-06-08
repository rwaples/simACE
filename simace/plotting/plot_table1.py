"""Render an epidemiological Table 1 summarising the simulated study population."""

__all__ = [
    "Table1Row",
    "Table1Section",
    "Table1Summary",
    "build_table1_summary",
    "render_table1_figure",
]

import logging
from dataclasses import dataclass
from statistics import mean
from typing import Literal

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


def _fmt_split(vals: list, fmt_func=_fmt_int) -> tuple[str, str]:
    """Return (value, range_annotation) as separate strings."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return ("\u2014", "")
    m = mean(clean)
    val = fmt_func(m)
    if len(clean) <= 1 or min(clean) == max(clean):
        return (val, "")
    return (val, f"[{fmt_func(min(clean))}\u2013{fmt_func(max(clean))}]")


def _fmt_split_pct(vals: list) -> tuple[str, str]:
    return _fmt_split(vals, fmt_func=_fmt_pct)


def _fmt_split_f(vals: list, decimals: int = 2) -> tuple[str, str]:
    return _fmt_split(vals, fmt_func=lambda v: _fmt_f(v, decimals))


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
    # Support varying key names across stat types
    inc = ci_data.get("incidence") or ci_data.get("observed_values") or ci_data.get("values")
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
# Structured Table 1 data for native HTML rendering
# ---------------------------------------------------------------------------


#: How a section's rows map onto the matplotlib drawers. The HTML renderer
#: ignores this (it renders generically from ``columns``); the PDF view uses it
#: to pick a row drawer, since ``value_range`` and ``two_trait`` are both
#: two-column but lay out differently.
SectionLayout = Literal["value_range", "two_trait", "four_sex_trait"]


@dataclass(frozen=True)
class Table1Row:
    """One row in the structured Table 1 summary.

    A single-value ``values`` tuple is a span row (one composite string drawn
    across the value columns); a multi-value tuple fills the section's columns.
    ``subrow`` marks an indented sub-row in the PDF view (ignored by HTML).
    """

    label: str
    values: tuple[str, ...]
    muted: bool = False
    subrow: bool = False


@dataclass(frozen=True)
class Table1Section:
    """A Table 1 section with a fixed set of value columns."""

    title: str
    columns: tuple[str, ...]
    rows: tuple[Table1Row, ...]
    layout: SectionLayout = "value_range"


@dataclass(frozen=True)
class Table1Summary:
    """Structured Table 1 content shared by non-PDF renderers."""

    title: str
    sections: tuple[Table1Section, ...]
    footnotes: tuple[str, ...]


def build_table1_summary(
    all_stats: list[dict],
    scenario_params: dict,
    scenario: str = "",
) -> Table1Summary:
    """Build structured Table 1 content from plotting stats.

    The PDF atlas keeps its bespoke matplotlib layout, while the HTML atlas uses
    this table model to render native HTML rows from the same report fields.
    """
    if not all_stats:
        raise ValueError("Table 1 requires at least one stats record")

    p = scenario_params
    scenario_label = scenario or str(p.get("scenario", ""))
    title = "Study Population Characteristics"
    if scenario_label:
        title += f" — {scenario_label}"

    n_reps = len(all_stats)
    s0 = all_stats[0]
    n_ind = s0.get("n_individuals")
    n_ped = s0.get("n_individuals_ped")
    n_gen = s0.get("n_generations")

    f_n = _sex_n(s0, "trait1")[0]
    m_n = _sex_n(s0, "trait1")[1]
    f_pct_str = f"({_fmt_pct(f_n / n_ind)})" if f_n is not None and n_ind else ""
    m_pct_str = f"({_fmt_pct(m_n / n_ind)})" if m_n is not None and n_ind else ""

    n_sample = p.get("N_sample", 0)
    car = p.get("case_ascertainment_ratio", 1.0)
    sample_active = n_sample and n_sample > 0
    car_active = car != 1.0
    sample_val = _fmt_int(n_sample) if sample_active else "none"
    sample_rng = "" if sample_active else "(full population)"
    car_val = f"{car:.1f}\u00d7" if car_active else "1.0\u00d7"
    car_rng = "" if car_active else "(no enrichment)"

    def _value_range(label: str, value: str, rng: str = "", *, muted: bool = False, subrow: bool = False) -> Table1Row:
        return Table1Row(label, (value, rng), muted=muted, subrow=subrow)

    def _span(label: str, value: str, *, subrow: bool = False) -> Table1Row:
        """A row whose single composite value spans the section's value columns."""
        return Table1Row(label, (value,), subrow=subrow)

    def _dist_line(stats_key):
        parts = []
        for k in ["1", "2", "3", "4+"]:
            vals = [_safe_get(s, "family_size", stats_key, k) for s in all_stats]
            clean = [v for v in vals if v is not None]
            pct = _fmt_pct(mean(clean)) if clean else "\u2014"
            parts.append(f"{k}: {pct}")
        return " / ".join(parts)

    def _person_dist_line():
        parts = []
        for k in ["0", "1", "2", "3", "4+"]:
            vals = [_safe_get(s, "family_size", "person_offspring_dist", k) for s in all_stats]
            clean = [v for v in vals if v is not None]
            pct = _fmt_pct(mean(clean)) if clean else "\u2014"
            parts.append(f"{k}: {pct}")
        return " / ".join(parts)

    def _mates_line(sex):
        m1 = [_safe_get(s, "family_size", "mates_by_sex", f"{sex}_1") for s in all_stats]
        m2 = [_safe_get(s, "family_size", "mates_by_sex", f"{sex}_2+") for s in all_stats]
        c1 = [v for v in m1 if v is not None]
        c2 = [v for v in m2 if v is not None]
        p1 = _fmt_pct(mean(c1)) if c1 else "\u2014"
        p2 = _fmt_pct(mean(c2)) if c2 else "\u2014"
        return f"1: {p1} / 2+: {p2}"

    population_rows = [
        _value_range("Total phenotyped individuals, n", _fmt_int(n_ind)),
        _value_range("Full pedigree individuals, n", _fmt_int(n_ped)),
        _value_range("Generations observed", str(n_gen) if n_gen else "\u2014"),
        _value_range("Female, n (%)", _fmt_int(f_n), f_pct_str, subrow=True),
        _value_range("Male, n (%)", _fmt_int(m_n), m_pct_str, subrow=True),
        _value_range("Sampled individuals, n", sample_val, sample_rng, muted=not sample_active),
        _value_range("Case ascertainment ratio", car_val, car_rng, muted=not car_active),
    ]

    v, rng = _fmt_split_f([_safe_get(s, "family_size", "mean") for s in all_stats], 2)
    population_rows.extend(
        [
            _value_range("Offspring per mating, mean", v, rng),
            _span("Distribution (1 / 2 / 3 / 4+)", _dist_line("size_dist"), subrow=True),
            _span("Offspring per person\u00b9 (0 / 1 / 2 / 3 / 4+)", _person_dist_line()),
            _span("Mates per mother (1 / 2+)", _mates_line("female")),
            _span("Mates per father (1 / 2+)", _mates_line("male")),
        ]
    )
    v, rng = _fmt_split_pct([_safe_get(s, "family_size", "frac_with_full_sib") for s in all_stats])
    population_rows.append(_value_range("With \u2265 1 full sib phenotyped, %", v, rng))

    ps_pheno = {str(k): [_safe_get(s, "parent_status", "phenotyped", str(k)) for s in all_stats] for k in [0, 1, 2]}
    ps_ped = {str(k): [_safe_get(s, "parent_status", "in_pedigree", str(k)) for s in all_stats] for k in [0, 1, 2]}

    def _parent_pct(counts_list):
        return [c / n_ind if c is not None and n_ind else None for c in counts_list]

    def _parent_summary(ps_dict):
        vals = _parent_pct(ps_dict["0"])
        p0 = _fmt_pct(vals[0]) if vals and vals[0] is not None else "\u2014"
        vals = _parent_pct(ps_dict["1"])
        p1 = _fmt_pct(vals[0]) if vals and vals[0] is not None else "\u2014"
        vals = _parent_pct(ps_dict["2"])
        p2 = _fmt_pct(vals[0]) if vals and vals[0] is not None else "\u2014"
        return f"0: {p0} / 1: {p1} / 2: {p2}"

    if any(v is not None for v in ps_pheno["0"]):
        population_rows.append(_value_range("Parents phenotyped (0 / 1 / 2)", _parent_summary(ps_pheno)))
    if any(v is not None for v in ps_ped["0"]):
        population_rows.append(_value_range("Parents in pedigree (0 / 1 / 2)", _parent_summary(ps_ped)))

    censor_age = p.get("censor_age", "—")
    population_rows.append(_value_range("Maximum follow-up age", f"{censor_age} years"))
    total_py = [_safe_get(s, "person_years", "total") for s in all_stats]
    if any(v is not None for v in total_py):
        v, rng = _fmt_split(total_py)
        population_rows.append(_value_range("Total person-years of follow-up", v, rng))
        mean_fu = [py / n_ind for py in total_py if py is not None] if n_ind else []
        if mean_fu:
            v, rng = _fmt_split_f(mean_fu, 1)
            population_rows.append(_value_range("Mean follow-up per person, years", v, rng))
    deaths = [_safe_get(s, "person_years", "deaths") for s in all_stats]
    if any(d is not None for d in deaths):
        v, rng = _fmt_split(deaths)
        population_rows.append(_value_range("Deaths during follow-up, n", v, rng))

    def _affected_by_sex(prev_list, n_list):
        return [
            round(p * n) if p is not None and n is not None else None for p, n in zip(prev_list, n_list, strict=True)
        ]

    fprev1 = [_safe_get(s, "cumulative_incidence_by_sex", "trait1", "female", "prevalence") for s in all_stats]
    mprev1 = [_safe_get(s, "cumulative_incidence_by_sex", "trait1", "male", "prevalence") for s in all_stats]
    fprev2 = [_safe_get(s, "cumulative_incidence_by_sex", "trait2", "female", "prevalence") for s in all_stats]
    mprev2 = [_safe_get(s, "cumulative_incidence_by_sex", "trait2", "male", "prevalence") for s in all_stats]
    fn1 = [_safe_get(s, "cumulative_incidence_by_sex", "trait1", "female", "n") for s in all_stats]
    mn1 = [_safe_get(s, "cumulative_incidence_by_sex", "trait1", "male", "n") for s in all_stats]
    fn2 = [_safe_get(s, "cumulative_incidence_by_sex", "trait2", "female", "n") for s in all_stats]
    mn2 = [_safe_get(s, "cumulative_incidence_by_sex", "trait2", "male", "n") for s in all_stats]
    total_py_list = [_safe_get(s, "person_years", "total") for s in all_stats]

    def _incidence_rate(prev_list, n_sex_list, py_total_list, n_total):
        rates = []
        for prev, n_sex, py in zip(prev_list, n_sex_list, py_total_list, strict=True):
            if prev is not None and n_sex and py and py > 0 and n_total:
                affected = prev * n_sex
                py_sex = py * n_sex / n_total
                rates.append(affected / py_sex * 1000 if py_sex > 0 else None)
            else:
                rates.append(None)
        return rates

    def _aoo_sex_quartile(all_stats, trait, sex, key):
        vals = []
        for s in all_stats:
            ci = _safe_get(s, "cumulative_incidence_by_sex", trait, sex, default={})
            q = _compute_aoo_quartiles(ci)
            if q[key] is not None:
                vals.append(q[key])
        return _fmt_range_f(vals, 1)

    disease_rows = [
        Table1Row(
            "Observed prevalence",
            (_fmt_range_pct(fprev1), _fmt_range_pct(mprev1), _fmt_range_pct(fprev2), _fmt_range_pct(mprev2)),
        ),
        Table1Row(
            "Affected, n",
            (
                _fmt_range(_affected_by_sex(fprev1, fn1)),
                _fmt_range(_affected_by_sex(mprev1, mn1)),
                _fmt_range(_affected_by_sex(fprev2, fn2)),
                _fmt_range(_affected_by_sex(mprev2, mn2)),
            ),
        ),
        Table1Row(
            "Incidence rate (per 1,000 PY)",
            (
                _fmt_range_f(_incidence_rate(fprev1, fn1, total_py_list, n_ind), 1),
                _fmt_range_f(_incidence_rate(mprev1, mn1, total_py_list, n_ind), 1),
                _fmt_range_f(_incidence_rate(fprev2, fn2, total_py_list, n_ind), 1),
                _fmt_range_f(_incidence_rate(mprev2, mn2, total_py_list, n_ind), 1),
            ),
        ),
    ]
    for qkey, qlabel in [("q1", "Q1"), ("median", "Median"), ("q3", "Q3")]:
        disease_rows.append(
            Table1Row(
                f"Age at onset, {qlabel}",
                (
                    _aoo_sex_quartile(all_stats, "trait1", "female", qkey),
                    _aoo_sex_quartile(all_stats, "trait1", "male", qkey),
                    _aoo_sex_quartile(all_stats, "trait2", "female", qkey),
                    _aoo_sex_quartile(all_stats, "trait2", "male", qkey),
                ),
            )
        )
    coaff_f = [_safe_get(s, "joint_affection", "by_sex", "female") for s in all_stats]
    coaff_m = [_safe_get(s, "joint_affection", "by_sex", "male") for s in all_stats]
    disease_rows.append(Table1Row("Co-affected, %", (_fmt_range_pct(coaff_f), _fmt_range_pct(coaff_m), "", "")))

    censoring_rows = []
    cascade0 = _safe_get(s0, "censoring_cascade", "trait1", default={})
    gen_keys = sorted(cascade0.keys()) if cascade0 else []
    for gk in gen_keys:
        gen_n = _safe_get(s0, "censoring_cascade", "trait1", gk, "n_gen")
        window = _safe_get(s0, "censoring_cascade", "trait1", gk, "window")
        n_str = _fmt_int(gen_n)
        win_str = f"ages {window[0]:.0f}\u2013{window[1]:.0f}" if window else ""
        obs1 = [_safe_get(s, "censoring_cascade", "trait1", gk, "observed") for s in all_stats]
        obs2 = [_safe_get(s, "censoring_cascade", "trait2", gk, "observed") for s in all_stats]
        gn = [_safe_get(s, "censoring_cascade", "trait1", gk, "n_gen") for s in all_stats]
        prev_g1 = [o / n if o is not None and n else None for o, n in zip(obs1, gn, strict=True)]
        prev_g2 = [o / n if o is not None and n else None for o, n in zip(obs2, gn, strict=True)]
        censoring_rows.append(
            Table1Row(
                f"{gk}: n={n_str}, {win_str}",
                (_fmt_range_pct(prev_g1), _fmt_range_pct(prev_g2)),
                subrow=True,
            )
        )

    mort_per_1k = []
    for s in all_stats:
        deaths = _safe_get(s, "person_years", "deaths")
        total = _safe_get(s, "person_years", "total")
        if deaths is not None and total and total > 0:
            mort_per_1k.append(deaths / total * 1000)
    if mort_per_1k:
        censoring_rows.append(Table1Row("Mortality rate (per 1,000 PY)", (_fmt_range_f(mort_per_1k, 1),)))

    footnotes = []
    if n_reps > 1:
        footnotes.append(f"Values are mean [min\u2013max] across {n_reps} replicates where applicable.")
    footnotes.append(
        "\u00b9 Includes youngest generation, whose offspring are outside the phenotyped cohort"
        " (100% childless by design)."
    )

    return Table1Summary(
        title=title,
        sections=(
            Table1Section("A. Population", ("Value", "Range"), tuple(population_rows), layout="value_range"),
            Table1Section(
                "B. Disease Characteristics",
                ("Trait 1 Female", "Trait 1 Male", "Trait 2 Female", "Trait 2 Male"),
                tuple(disease_rows),
                layout="four_sex_trait",
            ),
            Table1Section("C. Censoring", ("Trait 1", "Trait 2"), tuple(censoring_rows), layout="two_trait"),
        ),
        footnotes=tuple(footnotes),
    )


# ---------------------------------------------------------------------------
# Table drawing
# ---------------------------------------------------------------------------

# Layout constants (normalised figure coords on 11 × 8.5 landscape)
_LEFT = 0.04  # left margin for labels
_COL1 = 0.52  # column 1 (Trait 1) x position
_COL2 = 0.76  # column 2 (Trait 2) x position
_RIGHT = 0.96  # right edge for single-value column
_ROW_H = 0.020  # row height as fraction of figure height
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
    fig,
    ax,
    y: float,
    label: str,
    value: str,
    shade: bool,
    color: str = "black",
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


_DECIMAL_X = 0.66  # x position where the decimal point aligns


def _draw_row3(
    fig,
    ax,
    y: float,
    label: str,
    value: str,
    rng: str,
    shade: bool,
    color: str = "black",
) -> float:
    """Draw a three-column row: label | value (decimal-aligned) | [min–max]."""
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
    # Decimal-point alignment: split at '.' and render halves
    if "." in value:
        int_part, frac_part = value.split(".", 1)
        fig.text(
            _DECIMAL_X,
            y,
            int_part + ".",
            fontsize=_FONT_SIZE,
            fontfamily="monospace",
            color=color,
            va="top",
            ha="right",
            transform=fig.transFigure,
        )
        fig.text(
            _DECIMAL_X,
            y,
            frac_part,
            fontsize=_FONT_SIZE,
            fontfamily="monospace",
            color=color,
            va="top",
            ha="left",
            transform=fig.transFigure,
        )
    else:
        # No decimal — right-align at the decimal position
        fig.text(
            _DECIMAL_X,
            y,
            value,
            fontsize=_FONT_SIZE,
            fontfamily="monospace",
            color=color,
            va="top",
            ha="right",
            transform=fig.transFigure,
        )
    if rng:
        fig.text(
            _RIGHT,
            y,
            rng,
            fontsize=_FONT_SIZE,
            fontfamily="monospace",
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


# 4-column center positions: T1-F, T1-M, T2-F, T2-M
_C4 = [0.46, 0.60, 0.74, 0.88]


def _draw_col4_headers(fig, y: float) -> float:
    """Draw Trait 1 / Trait 2 header with Female / Male sub-headers."""
    t1_center = (_C4[0] + _C4[1]) / 2
    t2_center = (_C4[2] + _C4[3]) / 2
    fig.text(
        t1_center,
        y,
        "Trait 1",
        fontsize=_FONT_SIZE,
        fontweight="bold",
        fontfamily=_FONT,
        va="top",
        ha="center",
        transform=fig.transFigure,
    )
    fig.text(
        t2_center,
        y,
        "Trait 2",
        fontsize=_FONT_SIZE,
        fontweight="bold",
        fontfamily=_FONT,
        va="top",
        ha="center",
        transform=fig.transFigure,
    )
    y -= _ROW_H
    for x, label in zip(_C4, ["Female", "Male", "Female", "Male"], strict=True):
        fig.text(
            x,
            y,
            label,
            fontsize=_FONT_SIZE,
            fontweight="bold",
            fontfamily=_FONT,
            color="0.35",
            va="top",
            ha="center",
            transform=fig.transFigure,
        )
    return y - _ROW_H


def _draw_row4(
    fig,
    ax,
    y: float,
    label: str,
    t1f: str,
    t1m: str,
    t2f: str,
    t2m: str,
    shade: bool,
) -> float:
    """Draw a 4-column row: label | T1-F | T1-M | T2-F | T2-M."""
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
    for x, val in zip(_C4, [t1f, t1m, t2f, t2m], strict=True):
        fig.text(
            x,
            y,
            val,
            fontsize=_FONT_SIZE,
            fontfamily=_FONT,
            va="top",
            ha="center",
            transform=fig.transFigure,
        )
    return y - _ROW_H


def _draw_section_col_headers(fig, y: float, section: Table1Section) -> float:
    """Draw the column headers a section's layout calls for.

    ``value_range`` sections (Section A) carry no column-header row in the PDF.
    """
    if section.layout == "four_sex_trait":
        return _draw_col4_headers(fig, y)
    if section.layout == "two_trait":
        return _draw_col_headers(fig, y)
    return y


def _draw_model_row(fig, ax, y: float, section: Table1Section, row: Table1Row, shade: bool) -> float:
    """Draw one ``Table1Row`` with the drawer its layout and value count select."""
    label = f"  {row.label}" if row.subrow else row.label
    color = "0.55" if row.muted else "black"
    if section.layout == "four_sex_trait":
        t1f, t1m, t2f, t2m = (*row.values, "", "", "", "")[:4]
        return _draw_row4(fig, ax, y, label, t1f, t1m, t2f, t2m, shade)
    if len(row.values) == 1:
        # A single composite value spans the section's value columns.
        return _draw_row(fig, ax, y, label, row.values[0], shade, color=color)
    if section.layout == "two_trait":
        return _draw_row2(fig, ax, y, label, row.values[0], row.values[1], shade)
    return _draw_row3(fig, ax, y, label, row.values[0], row.values[1], shade, color=color)


# ---------------------------------------------------------------------------
# Main rendering function
# ---------------------------------------------------------------------------


def render_table1_figure(
    all_stats: list[dict],
    scenario_params: dict,
    scenario: str = "",
) -> plt.Figure:
    """Build and return the Table 1 figure (11 x 8.5 landscape).

    The matplotlib view consumes :func:`build_table1_summary`; the HTML atlas
    renders the same model, so the two atlases stay consistent by construction.

    Args:
        all_stats: List of stats report dicts, one per replicate.
        scenario_params: Merged scenario config parameters.
        scenario: Scenario name for the title.

    Returns:
        matplotlib Figure ready for ``pdf.savefig()``.
    """
    summary = build_table1_summary(all_stats, scenario_params, scenario)

    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = f"Table 1.  Study Population Characteristics — {scenario}"
    fig.text(
        0.50,
        0.96,
        title,
        fontsize=_TITLE_SIZE,
        fontweight="bold",
        fontfamily="sans-serif",
        ha="center",
        va="top",
        transform=fig.transFigure,
    )
    # Thin rule below the title.
    fig.add_artist(
        plt.Line2D([_LEFT, _RIGHT], [0.935, 0.935], color="black", lw=0.8, transform=fig.transFigure, clip_on=False)
    )

    y = 0.89
    for idx, section in enumerate(summary.sections):
        if idx:
            y -= _ROW_H * 0.4  # gap between sections
        y = _draw_section_header(fig, y, section.title)
        y = _draw_section_col_headers(fig, y, section)
        shade = False
        for row in section.rows:
            shade = not shade
            y = _draw_model_row(fig, ax, y, section, row, shade)

    # Bottom rule.
    fig.add_artist(
        plt.Line2D(
            [_LEFT, _RIGHT], [y - 0.005, y - 0.005], color="black", lw=0.8, transform=fig.transFigure, clip_on=False
        )
    )

    if summary.footnotes:
        fig.text(
            _LEFT,
            y - 0.02,
            "  ".join(summary.footnotes),
            fontsize=6,
            fontfamily=_FONT,
            color="0.4",
            va="top",
            wrap=True,
            transform=fig.transFigure,
        )

    return fig
