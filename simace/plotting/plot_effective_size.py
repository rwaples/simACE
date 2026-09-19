"""Per-scenario effective-size atlas plots.

Reads per-rep ``effective_size.yaml`` files emitted by the Ne pipeline and
renders four figures plus a multi-page PDF atlas. No new computation: the
plots visualise YAML payloads, with theoretical references pulled from
:mod:`simace.analysis.stats.effective_size`.
"""

from __future__ import annotations

__all__ = [
    "cli",
    "gather_effective_size",
    "main",
    "plot_drift_signals",
    "plot_estimators_overview",
    "plot_family_size_variance",
    "plot_ne_by_generation",
]

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS

from simace.analysis.stats.effective_size import (
    family_size_variance_expected_ztp,
    ne_v_expected_ztp,
)
from simace.core.yaml_io import load_yaml
from simace.plotting.plot_style import COLOR_EXPECTED, COLOR_OBSERVED

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


_NE_KEYS_ORDERED = ALL_EFFECTIVE_SIZE_ESTIMATORS

# Where each estimator's per-cohort Ne series lives, and how its x axis reads.
# Records are indexed by *observed* generation label, so a series row carries
# the label its record gave it rather than a 0..n-1 slot.  Ne_I, Ne_C and
# Ne_GC report Ne per adjacent-cohort transition, so their Ne is one row
# shorter than their drift means; Ne_V reports it per parent generation.
_NE_SERIES_AXIS: dict[str, tuple[str, str]] = {
    "ne_inbreeding": ("transition", "transition"),
    "ne_coancestry": ("transition", "transition"),
    "ne_variance_family_size": ("transition", "parent generation"),
    "ne_sex_ratio": ("cohort", "generation"),
    "ne_individual_delta_f": ("cohort", "generation"),
    "ne_group_coancestry": ("transition", "transition"),
}

# Drift means, all indexed by the record's ``generations`` cohort labels:
# the payload key each estimator writes, and the series column it lands in.
_DRIFT_COLUMNS: dict[str, str] = {
    "ne_inbreeding": "mean_f_per_gen",
    "ne_coancestry": "mean_theta_per_gen",
    "ne_group_coancestry": "mean_group_coancestry_per_gen",
}
_DRIFT_FIELD: dict[str, str] = {
    "ne_inbreeding": "mean_f",
    "ne_coancestry": "mean_theta",
    "ne_group_coancestry": "mean_group_coancestry",
}

_VARIANCE_COLUMNS = ("v_mm", "v_mf", "v_fm", "v_ff", "cov_m", "cov_f")
_SERIES_VALUE_FIELDS = ("ne", "mean_f", "mean_theta", "mean_group_coancestry", *_VARIANCE_COLUMNS)

# Short labels for axis titles.
_SHORT_LABEL = {
    "ne_inbreeding": "Ne_I (inbreeding)",
    "ne_coancestry": "Ne_C (coancestry)",
    "ne_variance_family_size": "Ne_V (family-size var.)",
    "ne_sex_ratio": "Ne_sr (sex ratio)",
    "ne_individual_delta_f": "Ne_iΔF",
    "ne_long_term_contributions": "Ne_LTC",
    "ne_hill_overlapping": "Ne_H (Hill)",
    "ne_group_coancestry": "Ne_GC (group coancestry)",
}


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def gather_effective_size(
    yaml_paths: list[Path] | list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read per-rep yamls into long-form scalar and series DataFrames.

    Args:
        yaml_paths: Paths to ``effective_size.yaml`` files. Rep index is
            assigned by enumeration order (1-based).

    Returns:
        Tuple ``(scalar_df, series_df)``:

        * ``scalar_df`` — one row per ``(rep, estimator)`` with columns
          ``rep``, ``estimator``, ``ne``, ``expected``. Missing/null Ne
          becomes ``NaN`` (no row dropped).
        * ``series_df`` — one row per ``(rep, estimator, kind, label)`` for
          the six estimators that expose a per-cohort or per-transition
          vector. Columns: ``rep``, ``estimator``, ``kind`` (``"cohort"``
          or ``"transition"``), ``x`` (numeric plotting position taken from
          the label), ``label`` (the record's own observed label), ``ne``,
          ``mean_f``, ``mean_theta``, ``mean_group_coancestry``, ``v_mm``,
          ``v_mf``, ``v_fm``, ``v_ff``, ``cov_m``, ``cov_f``. Fields not
          applicable to a given row are ``NaN``. An estimator the library
          reported as unavailable contributes a scalar row with ``ne`` NaN
          and no series rows.
    """
    scalar_rows: list[dict] = []
    series_rows: list[dict] = []
    for rep_idx, path in enumerate(yaml_paths, start=1):
        data = load_yaml(str(path))
        for est in _NE_KEYS_ORDERED:
            entry = data.get(est) or {}
            scalar_rows.append(
                {
                    "rep": rep_idx,
                    "estimator": est,
                    "ne": _coerce_float(entry.get("ne")),
                    "expected": _coerce_float(entry.get("expected")),
                }
            )
            if est not in _NE_SERIES_AXIS:
                continue
            series_rows.extend(_extract_series_rows(rep_idx, est, entry))

    scalar_schema = {"rep": pl.Int64, "estimator": pl.String, "ne": pl.Float64, "expected": pl.Float64}
    scalar_df = pl.DataFrame(scalar_rows, schema=scalar_schema)
    series_columns = [
        "rep",
        "estimator",
        "kind",
        "x",
        "label",
        "ne",
        "mean_f",
        "mean_theta",
        "mean_group_coancestry",
        "v_mm",
        "v_mf",
        "v_fm",
        "v_ff",
        "cov_m",
        "cov_f",
    ]
    series_schema: dict[str, type[pl.DataType]] = dict.fromkeys(series_columns, pl.Float64)
    series_schema.update({"rep": pl.Int64, "estimator": pl.String, "kind": pl.String, "label": pl.String})
    series_df = pl.DataFrame(series_rows, schema=series_schema)
    return scalar_df, series_df


def _extract_series_rows(rep: int, estimator: str, entry: dict) -> list[dict]:
    """Flatten one estimator's per-cohort and per-transition payload into rows.

    Every array is read against the label array the record carries, so a
    payload whose labels are missing or mismatched (an unavailable estimator,
    or a record from before the labelled contract) contributes no rows for
    that axis rather than a fabricated 0..n-1 axis.
    """
    rows: list[dict] = []
    kind, _ = _NE_SERIES_AXIS[estimator]
    generations = _label_array(entry.get("generations"))

    drift_key = _DRIFT_COLUMNS.get(estimator)
    if drift_key is not None:
        drift = entry.get(drift_key)
        if generations is not None and _seq_len(drift) == len(generations):
            rows.extend(
                _series_row(rep, estimator, "cohort", float(g), str(g), **{_DRIFT_FIELD[estimator]: _seq_at(drift, i)})
                for i, g in enumerate(generations)
            )

    if kind == "cohort":
        ne_vec = entry.get("ne_per_gen")
        if generations is not None and _seq_len(ne_vec) == len(generations):
            rows.extend(
                _series_row(rep, estimator, "cohort", float(g), str(g), ne=_seq_at(ne_vec, i))
                for i, g in enumerate(generations)
            )
        return rows

    if estimator == "ne_variance_family_size":
        parents = _label_array(entry.get("parent_generations"))
        ne_vec = entry.get("ne_per_transition")
        if parents is None or _seq_len(ne_vec) != len(parents):
            return rows
        rows.extend(
            _series_row(
                rep,
                estimator,
                "transition",
                float(pgen),
                str(pgen),
                ne=_seq_at(ne_vec, i),
                **{col: _seq_at(entry.get(col), i) for col in _VARIANCE_COLUMNS},
            )
            for i, pgen in enumerate(parents)
        )
        return rows

    src = _label_array(entry.get("transition_from"))
    dst = _label_array(entry.get("transition_to"))
    ne_vec = entry.get("ne_per_gen")
    if src is None or dst is None or len(src) != len(dst) or _seq_len(ne_vec) != len(src):
        return rows
    rows.extend(
        _series_row(rep, estimator, "transition", (a + b) / 2.0, f"{a}→{b}", ne=_seq_at(ne_vec, i))
        for i, (a, b) in enumerate(zip(src, dst, strict=True))
    )
    return rows


def _series_row(rep: int, estimator: str, kind: str, x: float, label: str, **values: float) -> dict:
    """One series row: the axis it belongs to, its label, and the fields it carries."""
    row = dict.fromkeys(_SERIES_VALUE_FIELDS, float("nan"))
    row.update(values)
    row.update({"rep": rep, "estimator": estimator, "kind": kind, "x": x, "label": label})
    return row


def _label_array(labels: object) -> list[int] | None:
    """A record's observed-label array as ints, or ``None`` when it carries none."""
    if not isinstance(labels, list) or not labels or any(v is None for v in labels):
        return None
    return [int(v) for v in labels]


def _seq_len(seq: object) -> int:
    return len(seq) if isinstance(seq, list) else -1


def _seq_at(seq: list | None, i: int) -> float:
    if seq is None or i >= len(seq):
        return float("nan")
    return _coerce_float(seq[i])


def _coerce_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_estimators_overview(
    scalar_df: pl.DataFrame,
    scenario_subtitle: str,
    out: Path,
    ext: str = "png",
) -> None:
    """Figure 1: per-rep aggregate Ne across all 8 estimators (log y).

    One column per estimator; per-rep dots in ``COLOR_OBSERVED``; horizontal
    ``expected`` reference in ``COLOR_EXPECTED`` (dashed) where defined.
    Reps with ``ne is None`` (commonly Ne_LTC at G_ped=6) are shown as open
    markers below the lower x-axis. Mean ± SD across reps annotated above
    each column.
    """
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    estimators = list(_NE_KEYS_ORDERED)
    positions = np.arange(len(estimators))

    # Plot range derived from finite values across all estimators.
    finite = scalar_df["ne"].drop_nans().drop_nulls()
    if not finite.is_empty():
        finite_arr = finite.to_numpy()
        lo = max(float(finite_arr.min()) * 0.5, 1.0)
        hi = float(finite_arr.max()) * 2.0
    else:
        lo, hi = 1.0, 1e6

    null_marker_y = lo * 0.6  # below the visible range, but log-axis-friendly

    for x, est in zip(positions, estimators, strict=True):
        sub = scalar_df.filter(pl.col("estimator") == est)
        finite_vals = sub["ne"].drop_nans().drop_nulls().to_numpy()
        null_count = len(sub) - len(finite_vals)

        if finite_vals.size:
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, finite_vals.size)
            ax.scatter(
                np.full_like(finite_vals, x) + jitter,
                finite_vals,
                color=COLOR_OBSERVED,
                alpha=0.85,
                s=30,
                zorder=3,
            )
            mean = float(finite_vals.mean())
            sd = float(finite_vals.std(ddof=1)) if finite_vals.size > 1 else 0.0
            ax.annotate(
                f"{mean:,.0f}\n±{sd:,.0f}",
                xy=(x, hi),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.3",
            )

        if null_count:
            ax.scatter(
                [x] * null_count,
                [null_marker_y] * null_count,
                facecolors="none",
                edgecolors=COLOR_OBSERVED,
                s=30,
                zorder=3,
            )

        expected_vals = sub["expected"].drop_nans().drop_nulls().unique(maintain_order=True)
        if len(expected_vals):
            exp = float(expected_vals[0])
            ax.hlines(
                exp,
                x - 0.35,
                x + 0.35,
                colors=COLOR_EXPECTED,
                linestyles="--",
                linewidth=1.5,
                zorder=2,
            )

    ax.set_yscale("log")
    ax.set_ylim(lo * 0.4, hi * 1.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([_SHORT_LABEL[e] for e in estimators], rotation=30, ha="right")
    ax.set_ylabel("Ne (log scale)")
    ax.set_title(f"Effective population size by estimator\n{scenario_subtitle}", fontsize=10)

    _apply_log_ne_yticks(ax)
    ax.grid(axis="y", which="both", alpha=0.3)
    _save(fig, out / f"effective_size.estimators.{ext}")


def _apply_label_xticks(ax, sub: pl.DataFrame) -> None:
    """Tick the x axis at the observed labels the records carry, in x order.

    Labels come from the data, not from a ``range(n)``: a pedigree whose
    generations skip a value gets ticks only where a cohort exists.
    """
    ticks = sub.select("x", "label").unique(maintain_order=True).sort("x")
    if ticks.is_empty():
        return
    ax.set_xticks(ticks["x"].to_numpy())
    ax.set_xticklabels(ticks["label"].to_list(), fontsize=8)


def _apply_log_ne_yticks(ax) -> None:
    """Log-y axis with dense, comma-formatted Ne ticks (Ne values ≥ 1)."""
    _apply_log_yticks(ax, formatter=lambda v: f"{int(v):,}")


def _apply_log_yticks(ax, formatter) -> None:
    """Configure log-y axis with dense ticks using ``formatter(value) → str``."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2.0, 3.0, 5.0, 7.0), numticks=60))
    fmt = mticker.FuncFormatter(lambda v, _pos: formatter(v))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="y", which="major", labelsize=9)
    ax.tick_params(axis="y", which="minor", labelsize=7, labelcolor="0.4")


def _format_small_float(v: float) -> str:
    """Compact label for small positive floats (drift / variance values)."""
    if v <= 0:
        return ""
    if v >= 0.1:
        return f"{v:.3f}"
    if v >= 1e-4:
        return f"{v:.4f}"
    return f"{v:.1e}"


def plot_ne_by_generation(
    series_df: pl.DataFrame,
    scalar_df: pl.DataFrame,
    out: Path,
    ext: str = "png",
) -> None:
    """Figure 2: per-generation/per-transition Ne curves (2×3 grid).

    ``scalar_df`` supplies the per-estimator ``expected`` reference line.
    """
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), sharey=False)
    for ax, (est, (kind, xlabel)) in zip(axes.flat, _NE_SERIES_AXIS.items(), strict=True):
        sub = series_df.filter((pl.col("estimator") == est) & (pl.col("kind") == kind))
        if sub.is_empty():
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")
            ax.set_title(_SHORT_LABEL[est])
            continue

        for (_rep,), rep_df in sub.group_by("rep", maintain_order=True):
            ne_vals = rep_df["ne"].to_numpy().astype(float)
            ys = np.where(ne_vals > 0, ne_vals, np.nan)
            ax.plot(
                rep_df["x"].to_numpy(),
                ys,
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.7,
            )

        expected_vals = (
            scalar_df.filter(pl.col("estimator") == est)["expected"]
            .drop_nans()
            .drop_nulls()
            .unique(maintain_order=True)
        )
        expected_val = float(expected_vals[0]) if len(expected_vals) else None
        if expected_val is not None and expected_val > 0:
            ax.axhline(
                expected_val,
                color=COLOR_EXPECTED,
                linestyle="--",
                linewidth=1.5,
            )

        # Log scale with dense ticks; per-panel autoscale (the expected
        # axhline ensures the reference is included). Explicitly NOT shared
        # with siblings so narrow-range panels (Ne_V, Ne_sr) don't get
        # squashed by wide-range siblings (Ne_GC).
        ax.set_yscale("log")
        ax.autoscale_view()
        _apply_log_ne_yticks(ax)

        _apply_label_xticks(ax, sub)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Ne (log)")
        ax.set_title(_SHORT_LABEL[est], fontsize=10)
        ax.grid(which="both", alpha=0.3)

    fig.suptitle("Ne by generation / transition", fontsize=12)
    _save(fig, out / f"effective_size.by_generation.{ext}")


def plot_drift_signals(series_df: pl.DataFrame, out: Path, ext: str = "png") -> None:
    """Figure 3: mean F, θ, group coancestry per generation (1×3)."""
    panels = [
        ("ne_inbreeding", "mean_f", "mean F"),
        ("ne_coancestry", "mean_theta", "mean θ"),
        ("ne_group_coancestry", "mean_group_coancestry", "mean group coancestry"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, (est, col, label) in zip(axes, panels, strict=True):
        sub = series_df.filter((pl.col("estimator") == est) & (pl.col("kind") == "cohort"))
        if sub.is_empty() or sub[col].drop_nans().drop_nulls().is_empty():
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")
            ax.set_title(label)
            continue
        for (_rep,), rep_df in sub.group_by("rep", maintain_order=True):
            # Clip non-positive values (e.g. F=0 in gens 0/1) so log scale
            # doesn't choke; gaps appear as missing markers.
            col_vals = rep_df[col].to_numpy().astype(float)
            ys = np.where(col_vals > 0, col_vals, np.nan)
            ax.plot(
                rep_df["x"].to_numpy(),
                ys,
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.7,
            )
        ax.set_yscale("log")
        ax.autoscale_view()
        _apply_log_yticks(ax, formatter=_format_small_float)
        _apply_label_xticks(ax, sub)
        ax.set_xlabel("generation")
        ax.set_ylabel(f"{label} (log)")
        ax.set_title(label, fontsize=10)
        ax.grid(which="both", alpha=0.3)
    fig.suptitle("Drift signals underlying slope-based Ne estimators", fontsize=12)
    _save(fig, out / f"effective_size.drift.{ext}")


def plot_family_size_variance(
    series_df: pl.DataFrame,
    expected_v: float | None,
    expected_cov: float | None,
    out: Path,
    ext: str = "png",
) -> None:
    """Figure 4: per-transition v_** and cov_* with ZTP closed-form references."""
    sub = series_df.filter(pl.col("estimator") == "ne_variance_family_size")  # one axis only: parent generation
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    ax_v, ax_c = axes

    if sub.is_empty():
        for ax in axes:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")
        _save(fig, out / f"effective_size.family_size_variance.{ext}")
        return

    v_cols = ["v_mm", "v_mf", "v_fm", "v_ff"]
    cov_cols = ["cov_m", "cov_f"]

    palette_v = ["#4477AA", "#EE7733", "#228833", "#CC3311"]
    palette_c = ["#4477AA", "#EE7733"]

    for col, color in zip(v_cols, palette_v, strict=True):
        for (_rep,), rep_df in sub.group_by("rep", maintain_order=True):
            ax_v.plot(
                rep_df["x"].to_numpy(),
                rep_df[col].to_numpy(),
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.55,
                color=color,
            )
        # Add a single legend handle per quadrant.
        ax_v.plot([], [], color=color, label=col)

    for col, color in zip(cov_cols, palette_c, strict=True):
        for (_rep,), rep_df in sub.group_by("rep", maintain_order=True):
            ax_c.plot(
                rep_df["x"].to_numpy(),
                rep_df[col].to_numpy(),
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.55,
                color=color,
            )
        ax_c.plot([], [], color=color, label=col)

    if expected_v is not None:
        ax_v.axhline(
            expected_v, color=COLOR_EXPECTED, linestyle="--", linewidth=1.5, label=f"ZTP exp. ≈ {expected_v:.3f}"
        )
    if expected_cov is not None:
        ax_c.axhline(
            expected_cov, color=COLOR_EXPECTED, linestyle="--", linewidth=1.5, label=f"ZTP exp. ≈ {expected_cov:.3f}"
        )

    for ax, title in [
        (ax_v, "Offspring-count variance v_** (per parent generation)"),
        (ax_c, "Between-mate covariance cov_* (per parent generation)"),
    ]:
        _apply_label_xticks(ax, sub)
        ax.set_xlabel("parent generation")
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.autoscale_view()
        _apply_log_yticks(ax, formatter=_format_small_float)
        ax.legend(fontsize=8, loc="best")
        ax.grid(which="both", alpha=0.3)

    fig.suptitle("Family-size variance / covariance vs ZTP closed-form", fontsize=12)
    _save(fig, out / f"effective_size.family_size_variance.{ext}")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _build_subtitle(params: dict, scenario: str | None = None) -> str:
    """Compose the Figure 1 subtitle from the rep params.yaml dict.

    Branches on ``mating_model``: under WF, omits the ZTP λ term and
    reports ``Ne_V≈N`` (the idealized-WF expectation) instead of the
    ZTP-derived value.
    """
    scenario = scenario or params.get("scenario") or params.get("scenario_name") or "scenario"
    mating_model = params.get("mating_model", "standard")
    n = params.get("N")
    lam = params.get("mating_lambda")
    g_ped = params.get("G_ped")
    parts = [str(scenario)]
    if n is not None:
        parts.append(f"N={int(n):,}")
    if mating_model == "wright_fisher":
        parts.append("WF")
    elif lam is not None:
        parts.append(f"λ={float(lam):g}")
    if g_ped is not None:
        parts.append(f"G_ped={int(g_ped)}")
    if n is not None:
        if mating_model == "wright_fisher":
            parts.append(f"Ne_V≈{int(n):,}")
        elif lam is not None:
            parts.append(f"Ne_V≈{ne_v_expected_ztp(float(n), float(lam)):,.0f}")
    return "  ".join(parts)


def _infer_scenario_from_path(params_path: str | Path) -> str | None:
    """Best-effort scenario name from a ``results/<folder>/<scenario>/rep*/params.yaml`` path."""
    parts = Path(params_path).resolve().parts
    if len(parts) < 3:
        return None
    # ...results/<folder>/<scenario>/rep<N>/params.yaml
    for i, p in enumerate(parts):
        if p.startswith("rep") and p[3:].isdigit() and i >= 1:
            return parts[i - 1]
    return None


@dataclass(frozen=True)
class EffectiveSizeContext:
    """Prepared inputs shared by the effective-size renderers."""

    scalar_df: pl.DataFrame
    series_df: pl.DataFrame
    subtitle: str
    expected_v: float | None
    expected_cov: float | None


@dataclass(frozen=True)
class EffectiveSizeRenderSpec:
    """One effective-size plot: output basename and how to render it.

    The plot functions self-name their file; ``basename`` is the manifest-facing
    label kept in lockstep with
    :data:`simace.plotting.atlas_manifest.EFFECTIVE_SIZE_ATLAS` by the
    renderer-coverage test in ``tests/plotting/test_atlas_manifest.py``.
    """

    basename: str
    render: Callable[[EffectiveSizeContext, Path, str], None]


# Registry binding each effective-size basename to its renderer. Adding a plot
# means adding a PlotEntry to EFFECTIVE_SIZE_ATLAS *and* a spec here; the
# renderer-coverage test fails if the two basename sets diverge.
EFFECTIVE_SIZE_RENDERERS: tuple[EffectiveSizeRenderSpec, ...] = (
    EffectiveSizeRenderSpec(
        "effective_size.estimators",
        lambda ctx, out, ext: plot_estimators_overview(ctx.scalar_df, ctx.subtitle, out, ext=ext),
    ),
    EffectiveSizeRenderSpec(
        "effective_size.by_generation",
        lambda ctx, out, ext: plot_ne_by_generation(ctx.series_df, ctx.scalar_df, out, ext=ext),
    ),
    EffectiveSizeRenderSpec(
        "effective_size.drift",
        lambda ctx, out, ext: plot_drift_signals(ctx.series_df, out, ext=ext),
    ),
    EffectiveSizeRenderSpec(
        "effective_size.family_size_variance",
        lambda ctx, out, ext: plot_family_size_variance(ctx.series_df, ctx.expected_v, ctx.expected_cov, out, ext=ext),
    ),
)


def main(
    yaml_paths: list[str],
    params_path: str,
    output_dir: str | Path,
    plot_ext: str = "png",
) -> None:
    """Build all four Ne plots plus the HTML atlas."""
    from simace.plotting.atlas_manifest import EFFECTIVE_SIZE_ATLAS
    from simace.plotting.plot_style import apply_nature_style
    from simace.plotting.render_atlas import render_atlas

    apply_nature_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    params = load_yaml(params_path)
    scalar_df, series_df = gather_effective_size([Path(p) for p in yaml_paths])

    subtitle = _build_subtitle(params, scenario=_infer_scenario_from_path(params_path))

    mating_model = params.get("mating_model", "standard")
    if mating_model == "wright_fisher":
        # Sex-structured WF: per-sex family-size is Poisson-like with mean 1
        # offspring per sex-slot, so the within-sex variance v → 1 and the
        # within-mating covariance cov → 0 (no per-pair fecundity).  Matches
        # the limits documented in simace/analysis/stats/effective_size.py.
        expected_v: float | None = 1.0
        expected_cov: float | None = 0.0
    else:
        lam = params.get("mating_lambda")
        if lam is None:
            expected_v = expected_cov = None
        else:
            ztp = family_size_variance_expected_ztp(float(lam))
            expected_v = ztp["v"]
            expected_cov = ztp["cov"]

    ctx = EffectiveSizeContext(
        scalar_df=scalar_df,
        series_df=series_df,
        subtitle=subtitle,
        expected_v=expected_v,
        expected_cov=expected_cov,
    )
    for spec in EFFECTIVE_SIZE_RENDERERS:
        spec.render(ctx, out, plot_ext)

    # HTML is the primary atlas rendering (ADR 0010); pass a .pdf output path
    # to render the on-demand PDF instead.
    render_atlas(
        list(EFFECTIVE_SIZE_ATLAS),
        out,
        out / "effective_size.atlas.html",
        plot_ext=plot_ext,
    )


def cli() -> None:
    """Command-line entry: render Ne atlas from a list of yamls."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Plot Ne atlas from per-rep effective_size.yaml files")
    add_logging_args(parser)
    parser.add_argument("--yaml", required=True, nargs="+", help="Per-rep effective_size.yaml paths")
    parser.add_argument("--params", required=True, help="Per-rep params.yaml")
    parser.add_argument("--output-dir", required=True, help="Output plots directory")
    parser.add_argument("--plot-format", choices=["png", "pdf"], default="png")
    args = parser.parse_args()
    init_logging(args)
    main(args.yaml, args.params, args.output_dir, plot_ext=args.plot_format)
