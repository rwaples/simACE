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
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from simace.analysis.stats.effective_size import (
    family_size_variance_expected_ztp,
    ne_v_expected_ztp,
)
from simace.core.yaml_io import load_yaml
from simace.plotting.plot_style import COLOR_EXPECTED, COLOR_OBSERVED

logger = logging.getLogger(__name__)


_NE_KEYS_ORDERED = (
    "ne_inbreeding",
    "ne_coancestry",
    "ne_variance_family_size",
    "ne_sex_ratio",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_hill_overlapping",
    "ne_caballero_toro",
)

# Estimators with a per-generation/per-transition vector that Figure 2 plots.
_PER_GEN_ESTIMATORS = (
    "ne_inbreeding",
    "ne_coancestry",
    "ne_variance_family_size",
    "ne_sex_ratio",
    "ne_individual_delta_f",
    "ne_caballero_toro",
)

# Short labels for axis titles.
_SHORT_LABEL = {
    "ne_inbreeding": "Ne_I (inbreeding)",
    "ne_coancestry": "Ne_C (coancestry)",
    "ne_variance_family_size": "Ne_V (family-size var.)",
    "ne_sex_ratio": "Ne_sr (sex ratio)",
    "ne_individual_delta_f": "Ne_iΔF",
    "ne_long_term_contributions": "Ne_LTC",
    "ne_hill_overlapping": "Ne_H (Hill)",
    "ne_caballero_toro": "Ne_CT (Caballero-Toro)",
}


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def gather_effective_size(
    yaml_paths: list[Path] | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read per-rep yamls into long-form scalar and series DataFrames.

    Args:
        yaml_paths: Paths to ``effective_size.yaml`` files. Rep index is
            assigned by enumeration order (1-based).

    Returns:
        Tuple ``(scalar_df, series_df)``:

        * ``scalar_df`` — one row per ``(rep, estimator)`` with columns
          ``rep``, ``estimator``, ``ne``, ``expected``. Missing/null Ne
          becomes ``NaN`` (no row dropped).
        * ``series_df`` — one row per ``(rep, estimator, index)`` for the
          six estimators that expose a per-gen/per-transition vector.
          Columns: ``rep``, ``estimator``, ``index``, ``kind``
          (``"generation"`` or ``"transition"``), ``ne``, ``mean_f``,
          ``mean_theta``, ``mean_self_coancestry``, ``v_mm``, ``v_mf``,
          ``v_fm``, ``v_ff``, ``cov_m``, ``cov_f``. Fields not applicable
          to a given estimator are ``NaN``.
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
            if est not in _PER_GEN_ESTIMATORS:
                continue
            series_rows.extend(_extract_series_rows(rep_idx, est, entry))

    scalar_df = pd.DataFrame(scalar_rows, columns=["rep", "estimator", "ne", "expected"])
    series_columns = [
        "rep",
        "estimator",
        "index",
        "kind",
        "ne",
        "mean_f",
        "mean_theta",
        "mean_self_coancestry",
        "v_mm",
        "v_mf",
        "v_fm",
        "v_ff",
        "cov_m",
        "cov_f",
    ]
    series_df = pd.DataFrame(series_rows, columns=series_columns)
    return scalar_df, series_df


def _extract_series_rows(rep: int, estimator: str, entry: dict) -> list[dict]:
    """Flatten one estimator's per-gen/per-transition payload into rows."""
    if estimator == "ne_variance_family_size":
        kind = "transition"
        ne_vec = entry.get("ne_per_transition") or []
    else:
        kind = "generation"
        ne_vec = entry.get("ne_per_gen") or []

    n = len(ne_vec)
    if n == 0:
        return []

    mean_f = entry.get("mean_f_per_gen") if estimator == "ne_inbreeding" else None
    mean_theta = entry.get("mean_theta_per_gen") if estimator == "ne_coancestry" else None
    mean_self_co = entry.get("mean_self_coancestry_per_gen") if estimator == "ne_caballero_toro" else None

    return [
        {
            "rep": rep,
            "estimator": estimator,
            "index": i,
            "kind": kind,
            "ne": _coerce_float(ne_vec[i]),
            "mean_f": _seq_at(mean_f, i),
            "mean_theta": _seq_at(mean_theta, i),
            "mean_self_coancestry": _seq_at(mean_self_co, i),
            "v_mm": _seq_at(entry.get("v_mm"), i),
            "v_mf": _seq_at(entry.get("v_mf"), i),
            "v_fm": _seq_at(entry.get("v_fm"), i),
            "v_ff": _seq_at(entry.get("v_ff"), i),
            "cov_m": _seq_at(entry.get("cov_m"), i),
            "cov_f": _seq_at(entry.get("cov_f"), i),
        }
        for i in range(n)
    ]


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
    scalar_df: pd.DataFrame,
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
    finite = scalar_df["ne"].dropna()
    if not finite.empty:
        lo = max(finite.min() * 0.5, 1.0)
        hi = finite.max() * 2.0
    else:
        lo, hi = 1.0, 1e6

    null_marker_y = lo * 0.6  # below the visible range, but log-axis-friendly

    for x, est in zip(positions, estimators, strict=True):
        sub = scalar_df[scalar_df["estimator"] == est]
        finite_vals = sub["ne"].dropna().to_numpy()
        null_count = int(sub["ne"].isna().sum())

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

        expected_vals = sub["expected"].dropna().unique()
        if expected_vals.size:
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
    series_df: pd.DataFrame,
    scalar_df: pd.DataFrame,
    out: Path,
    ext: str = "png",
) -> None:
    """Figure 2: per-generation/per-transition Ne curves (2×3 grid).

    ``scalar_df`` supplies the per-estimator ``expected`` reference line.
    """
    panels = [
        ("ne_inbreeding", "generation"),
        ("ne_coancestry", "generation"),
        ("ne_variance_family_size", "transition"),
        ("ne_sex_ratio", "generation"),
        ("ne_individual_delta_f", "generation"),
        ("ne_caballero_toro", "generation"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), sharey=False)
    for ax, (est, kind) in zip(axes.flat, panels, strict=True):
        sub = series_df[series_df["estimator"] == est]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")
            ax.set_title(_SHORT_LABEL[est])
            continue

        for _rep, rep_df in sub.groupby("rep"):
            xs = rep_df["index"].to_numpy()
            if kind == "transition":
                xs = xs + 0.5  # transition g→g+1 sits between gens g and g+1
            ys = rep_df["ne"].where(rep_df["ne"] > 0).to_numpy()
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.7,
            )

        expected_vals = scalar_df.loc[scalar_df["estimator"] == est, "expected"].dropna().unique()
        expected_val = float(expected_vals[0]) if expected_vals.size else None
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
        # squashed by wide-range siblings (Ne_CT).
        ax.set_yscale("log")
        ax.autoscale_view()
        _apply_log_ne_yticks(ax)

        n = int(sub["index"].max()) + 1
        if kind == "transition":
            ax.set_xticks(np.arange(n) + 0.5)
            ax.set_xticklabels([f"{g}→{g + 1}" for g in range(n)], fontsize=8)
            ax.set_xlabel("transition (g→g+1)")
        else:
            ax.set_xticks(np.arange(n))
            ax.set_xlabel("generation")

        ax.set_ylabel("Ne (log)")
        ax.set_title(_SHORT_LABEL[est], fontsize=10)
        ax.grid(which="both", alpha=0.3)

    fig.suptitle("Ne by generation / transition", fontsize=12)
    _save(fig, out / f"effective_size.by_generation.{ext}")


def plot_drift_signals(series_df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Figure 3: mean F, θ, self-coancestry per generation (1×3)."""
    panels = [
        ("ne_inbreeding", "mean_f", "mean F"),
        ("ne_coancestry", "mean_theta", "mean θ"),
        ("ne_caballero_toro", "mean_self_coancestry", "mean self-coancestry (founders)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, (est, col, label) in zip(axes, panels, strict=True):
        sub = series_df[series_df["estimator"] == est]
        if sub.empty or sub[col].dropna().empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")
            ax.set_title(label)
            continue
        for _rep, rep_df in sub.groupby("rep"):
            # Clip non-positive values (e.g. F=0 in gens 0/1) so log scale
            # doesn't choke; gaps appear as missing markers.
            ys = rep_df[col].where(rep_df[col] > 0).to_numpy()
            ax.plot(
                rep_df["index"].to_numpy(),
                ys,
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.7,
            )
        ax.set_yscale("log")
        ax.autoscale_view()
        _apply_log_yticks(ax, formatter=_format_small_float)
        ax.set_xlabel("generation")
        ax.set_ylabel(f"{label} (log)")
        ax.set_title(label, fontsize=10)
        ax.grid(which="both", alpha=0.3)
    fig.suptitle("Drift signals underlying slope-based Ne estimators", fontsize=12)
    _save(fig, out / f"effective_size.drift.{ext}")


def plot_family_size_variance(
    series_df: pd.DataFrame,
    expected_v: float | None,
    expected_cov: float | None,
    out: Path,
    ext: str = "png",
) -> None:
    """Figure 4: per-transition v_** and cov_* with ZTP closed-form references."""
    sub = series_df[series_df["estimator"] == "ne_variance_family_size"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    ax_v, ax_c = axes

    if sub.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")
        _save(fig, out / f"effective_size.family_size_variance.{ext}")
        return

    v_cols = ["v_mm", "v_mf", "v_fm", "v_ff"]
    cov_cols = ["cov_m", "cov_f"]

    n = int(sub["index"].max()) + 1
    xs_base = np.arange(n) + 0.5

    palette_v = ["#4477AA", "#EE7733", "#228833", "#CC3311"]
    palette_c = ["#4477AA", "#EE7733"]

    for col, color in zip(v_cols, palette_v, strict=True):
        for _rep, rep_df in sub.groupby("rep"):
            xs = rep_df["index"].to_numpy() + 0.5
            ax_v.plot(
                xs,
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
        for _rep, rep_df in sub.groupby("rep"):
            xs = rep_df["index"].to_numpy() + 0.5
            ax_c.plot(
                xs,
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
        (ax_v, "Offspring-count variance v_** (per transition)"),
        (ax_c, "Between-mate covariance cov_* (per transition)"),
    ]:
        ax.set_xticks(xs_base)
        ax.set_xticklabels([f"{g}→{g + 1}" for g in range(n)], fontsize=8)
        ax.set_xlabel("transition (g→g+1)")
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


def main(
    yaml_paths: list[str],
    params_path: str,
    output_dir: str | Path,
    plot_ext: str = "png",
) -> None:
    """Build all four Ne plots plus the atlas PDF."""
    from simace.plotting.atlas_manifest import EFFECTIVE_SIZE_ATLAS
    from simace.plotting.plot_atlas import assemble_atlas
    from simace.plotting.plot_style import apply_nature_style

    apply_nature_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    params = load_yaml(params_path)
    scalar_df, series_df = gather_effective_size([Path(p) for p in yaml_paths])

    subtitle = _build_subtitle(params, scenario=_infer_scenario_from_path(params_path))
    plot_estimators_overview(scalar_df, subtitle, out, ext=plot_ext)
    plot_ne_by_generation(series_df, scalar_df, out, ext=plot_ext)
    plot_drift_signals(series_df, out, ext=plot_ext)

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
    plot_family_size_variance(series_df, expected_v, expected_cov, out, ext=plot_ext)

    assemble_atlas(
        list(EFFECTIVE_SIZE_ATLAS),
        out,
        out / "effective_size.atlas.pdf",
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
