"""Gather EPIMIGHT meta-analysis results across bias scenarios and compute bias metrics.

Reads EPIMIGHT h² meta-analysis TSVs, true_parameters.json, and ltm_falconer.json
for each scenario × kind.  Produces a consolidated DataFrame with bias metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fit_ace.constants import KIND_ORDER
from sim_ace import setup_logging

logger = logging.getLogger(__name__)

# Scenario name parsing helpers
_PREV_MAP = {"K01": 0.01, "K05": 0.05, "K10": 0.10, "K20": 0.20, "K40": 0.40}
_CENSOR_MAP = {"nocensor": "none", "death": "death_only", "window": "window_only", "both": "both"}
_MODEL_MAP = {"ltm": "adult_ltm", "weibull": "weibull", "cure": "cure_frailty", "cox": "adult_cox"}


def parse_scenario_name(name: str) -> dict[str, Any]:
    """Extract metadata from an ebias_* scenario name."""
    parts = name.removeprefix("ebias_").split("_")
    # e.g. ltm_K10_C0_nocensor → ["ltm", "K10", "C0", "nocensor"]
    # e.g. weibull_K10_C0_nocensor → ["weibull", "K10", "C0", "nocensor"]
    model_key = parts[0]
    prev_key = parts[1] if len(parts) > 1 else ""
    c_key = parts[2] if len(parts) > 2 else ""
    censor_key = parts[3] if len(parts) > 3 else ""

    return {
        "phenotype_model": _MODEL_MAP.get(model_key, model_key),
        "prevalence": _PREV_MAP.get(prev_key, np.nan),
        "C": 0.2 if c_key == "C02" else 0.0,
        "censor_label": _CENSOR_MAP.get(censor_key, censor_key),
        "has_death_censor": censor_key in ("death", "both"),
        "has_window_censor": censor_key in ("window", "both"),
    }


def load_epimight_meta(scenario_dir: Path, kind: str, disorder: str = "d1") -> dict[str, float] | None:
    """Load EPIMIGHT fixed-effect meta-analysis h² for one kind."""
    path = scenario_dir / "tsv" / f"h2_{disorder}_meta_{kind}.tsv"
    if not path.exists():
        logger.warning("Missing EPIMIGHT meta: %s", path)
        return None
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def load_true_h2(scenario_dir: Path, trait: int = 1) -> float | None:
    """Load liability-scale true h² from true_parameters.json."""
    path = scenario_dir / "true_parameters.json"
    if not path.exists():
        return None
    with open(path) as f:
        params = json.load(f)
    return params.get(f"h2_trait{trait}_true")


def load_ltm_falconer(ltm_path: Path, kind: str) -> dict[str, Any] | None:
    """Load LTM Falconer h² for one kind."""
    if not ltm_path.exists():
        return None
    with open(ltm_path) as f:
        data = json.load(f)
    return data.get(kind)


def compute_bias_metrics(
    h2_est: float,
    h2_se: float,
    h2_l95: float,
    h2_u95: float,
    h2_true_liability: float,
    h2_ltm: float | None,
) -> dict[str, float]:
    """Compute bias metrics for a single EPIMIGHT estimate."""
    m: dict[str, float] = {}

    # Bias relative to liability h²
    m["abs_bias_liability"] = h2_est - h2_true_liability
    m["rel_bias_liability"] = (h2_est - h2_true_liability) / h2_true_liability if h2_true_liability else np.nan
    m["attenuation_ratio"] = h2_est / h2_true_liability if h2_true_liability else np.nan
    m["ci_covers_liability"] = float(h2_l95 <= h2_true_liability <= h2_u95)

    # Bias relative to LTM Falconer
    if h2_ltm is not None and not np.isnan(h2_ltm) and h2_ltm != 0:
        m["abs_bias_ltm"] = h2_est - h2_ltm
        m["rel_bias_ltm"] = (h2_est - h2_ltm) / h2_ltm
        m["ci_covers_ltm"] = float(h2_l95 <= h2_ltm <= h2_u95)
    else:
        m["abs_bias_ltm"] = np.nan
        m["rel_bias_ltm"] = np.nan
        m["ci_covers_ltm"] = np.nan

    return m


def gather_epimight_bias_results(
    results_dir: Path,
    scenarios: list[str],
    kinds: list[str] | None = None,
    rep: int = 1,
    trait: int = 1,
    subdir: str = "epimight",
) -> pd.DataFrame:
    """Gather bias metrics across all scenarios and kinds.

    Args:
        subdir: EPIMIGHT output subdirectory name (``"epimight"`` for
            standard analysis, ``"epimight_single"`` for single-pair).

    Returns DataFrame with one row per (scenario, kind).
    """
    if kinds is None:
        kinds = KIND_ORDER

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        base = results_dir / scenario / f"rep{rep}"
        epi_dir = base / subdir
        ltm_path = base / "ltm_falconer.json"

        h2_true_liability = load_true_h2(epi_dir, trait)
        if h2_true_liability is None:
            logger.warning("No true_parameters.json for %s, skipping", scenario)
            continue

        meta_info = parse_scenario_name(scenario)

        for kind in kinds:
            meta = load_epimight_meta(epi_dir, kind, f"d{trait}")
            ltm_data = load_ltm_falconer(ltm_path, kind)

            if meta is None:
                logger.warning("No EPIMIGHT meta for %s / %s, skipping", scenario, kind)
                continue

            h2_est = meta.get("fixed_meta", np.nan)
            h2_se = meta.get("fixed_se", np.nan)
            h2_l95 = meta.get("fixed_l95", np.nan)
            h2_u95 = meta.get("fixed_u95", np.nan)
            h2_ltm = ltm_data["h2_falconer"] if ltm_data else None

            bias = compute_bias_metrics(h2_est, h2_se, h2_l95, h2_u95, h2_true_liability, h2_ltm)

            row: dict[str, Any] = {
                "scenario": scenario,
                "kind": kind,
                **meta_info,
                "h2_epimight": h2_est,
                "h2_se": h2_se,
                "h2_l95": h2_l95,
                "h2_u95": h2_u95,
                "h2_true_liability": h2_true_liability,
                "h2_ltm_falconer": h2_ltm if h2_ltm is not None else np.nan,
                "se_ltm_falconer": ltm_data["se_h2"] if ltm_data and ltm_data.get("se_h2") else np.nan,
                "r_tetrachoric": ltm_data["r_tetrachoric"] if ltm_data and ltm_data.get("r_tetrachoric") else np.nan,
                **bias,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main(results_dir: str, scenarios: list[str], output_path: str, **kwargs: Any) -> None:
    """Gather results and write to TSV."""
    df = gather_epimight_bias_results(Path(results_dir), scenarios, **kwargs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Gather EPIMIGHT bias analysis results")
    parser.add_argument("--results-dir", required=True, help="Base results directory for the bias folder")
    parser.add_argument("--scenarios", nargs="+", required=True, help="List of scenario names")
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    setup_logging(log_file=args.log_file)
    main(args.results_dir, args.scenarios, args.output)


if __name__ == "__main__":
    cli()
