"""
Gather validation results from all scenarios into a single TSV file.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any
import csv
import re
import yaml
from pathlib import Path

import platform

from sim_ace.utils import get_nested

logger = logging.getLogger(__name__)


def extract_metrics(validation_path: str) -> dict[str, Any]:
    """Extract key metrics from a validation YAML file."""
    with open(validation_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Normalise to forward slashes so the regex works on Windows too
    validation_path = str(validation_path).replace("\\", "/")

    # Extract scenario and rep from path: results/{folder}/{scenario}/rep{rep}/validation.yaml
    match = re.search(r"results/([^/]+)/([^/]+)/rep(\d+)/validation\.yaml", validation_path)
    if match:
        scenario = match.group(2)  # group(1) is folder
        rep = int(match.group(3))
    else:
        scenario = "unknown"
        rep = 1

    # Parse simulate benchmark time from corresponding TSV
    simulate_seconds = None
    simulate_max_rss_mb = None
    bench_path = Path(re.sub(
        r"results/([^/]+)/([^/]+)/rep(\d+)/validation\.yaml",
        r"benchmarks/\1/\2/rep\3/simulate.tsv",
        validation_path,
    ))
    if bench_path.exists():
        with open(bench_path, encoding="utf-8", newline="") as bf:
            reader = csv.DictReader(bf, delimiter="\t")
            for row_b in reader:
                simulate_seconds = float(row_b["s"])

                if platform.system() == "Windows":
                    # Windows non supporta max_rss → metto NaN
                    simulate_max_rss_mb = float(1)
                else:
                    # Linux/macOS → converto normalmente
                    simulate_max_rss_mb = float(row_b["max_rss"])

                break

    params = data["parameters"]
    summary = data["summary"]

    row = {
        "scenario": scenario,
        "rep": rep,
        "N": params.get("N"),
        "G_ped": params.get("G_ped"),
        "G_sim": params.get("G_sim"),
        # Trait 1 parameters
        "A1": params.get("A1"),
        "C1": params.get("C1"),
        "E1": params.get("E1"),
        # Trait 2 parameters
        "A2": params.get("A2"),
        "C2": params.get("C2"),
        "E2": params.get("E2"),
        # Cross-trait correlations
        "rA": params.get("rA"),
        "rC": params.get("rC"),
        # Population parameters
        "p_mztwin": params.get("p_mztwin"),
        "p_nonsocial_father": params.get("p_nonsocial_father"),
        "fam_size": params.get("fam_size"),
        "seed": params.get("seed"),
        "checks_failed": summary.get("checks_failed"),
        # Twin rate
        "observed_twin_rate": get_nested(data, "twins", "twin_rate", "observed_rate"),
        # Trait 1 variances
        "variance_A1": get_nested(data, "statistical", "variance_A1", "observed"),
        "variance_C1": get_nested(data, "statistical", "variance_C1", "observed"),
        "variance_E1": get_nested(data, "statistical", "variance_E1", "observed"),
        # Trait 2 variances
        "variance_A2": get_nested(data, "statistical", "variance_A2", "observed"),
        "variance_C2": get_nested(data, "statistical", "variance_C2", "observed"),
        "variance_E2": get_nested(data, "statistical", "variance_E2", "observed"),
        # Cross-trait correlations (observed)
        "observed_rA": get_nested(data, "statistical", "cross_trait_rA", "observed"),
        "observed_rC": get_nested(data, "statistical", "cross_trait_rC", "observed"),
        "observed_rE": get_nested(data, "statistical", "cross_trait_rE", "observed"),
        # MZ twin correlations (trait 1)
        "mz_twin_A1_corr": get_nested(
            data, "heritability", "mz_twin_A1_correlation", "observed"
        ),
        "mz_twin_liability1_corr": get_nested(
            data, "heritability", "mz_twin_liability1_correlation", "observed"
        ),
        # MZ twin correlations (trait 2)
        "mz_twin_A2_corr": get_nested(
            data, "heritability", "mz_twin_A2_correlation", "observed"
        ),
        "mz_twin_liability2_corr": get_nested(
            data, "heritability", "mz_twin_liability2_correlation", "observed"
        ),
        # DZ sibling correlations (trait 1)
        "dz_sibling_A1_corr": get_nested(
            data, "heritability", "dz_sibling_A1_correlation", "observed"
        ),
        "dz_sibling_liability1_corr": get_nested(
            data, "heritability", "dz_sibling_liability1_correlation", "observed"
        ),
        # DZ sibling correlations (trait 2)
        "dz_sibling_A2_corr": get_nested(
            data, "heritability", "dz_sibling_A2_correlation", "observed"
        ),
        "dz_sibling_liability2_corr": get_nested(
            data, "heritability", "dz_sibling_liability2_correlation", "observed"
        ),
        # Half-sib statistics
        "half_sib_prop_expected": get_nested(
            data, "half_sibs", "half_sib_pair_proportion", "expected"
        ),
        "half_sib_prop_observed": get_nested(
            data, "half_sibs", "half_sib_pair_proportion", "observed"
        ),
        "offspring_with_half_sib_expected": get_nested(
            data, "half_sibs", "offspring_with_half_sib", "expected"
        ),
        "offspring_with_half_sib_observed": get_nested(
            data, "half_sibs", "offspring_with_half_sib", "observed"
        ),
        "half_sib_A1_corr": get_nested(
            data, "half_sibs", "half_sib_A1_correlation", "observed"
        ),
        "half_sib_liability1_corr": get_nested(
            data, "half_sibs", "half_sib_liability1_correlation", "observed"
        ),
        "half_sib_shared_C1": get_nested(
            data, "half_sibs", "half_sib_shared_C1", "observed"
        ),
        # Benchmark timing and memory
        "simulate_seconds": simulate_seconds,
        "simulate_max_rss_mb": simulate_max_rss_mb,
        # Family size distribution
        "mother_mean_offspring": get_nested(
            data, "family_size_distribution", "mother", "mean"
        ),
        "father_mean_offspring": get_nested(
            data, "family_size_distribution", "father", "mean"
        ),
        # Falconer heritability estimates
        "falconer_h2_trait1": get_nested(
            data, "heritability", "falconer_estimate_trait1", "observed"
        ),
        "falconer_h2_trait2": get_nested(
            data, "heritability", "falconer_estimate_trait2", "observed"
        ),
        # Parent-offspring regressions (trait 1)
        "parent_offspring_A1_slope": get_nested(
            data, "heritability", "parent_offspring_A1_regression", "slope"
        ),
        "parent_offspring_A1_r2": get_nested(
            data, "heritability", "parent_offspring_A1_regression", "r_squared"
        ),
        "parent_offspring_liability1_slope": get_nested(
            data, "heritability", "parent_offspring_liability1_regression", "slope"
        ),
        "parent_offspring_liability1_r2": get_nested(
            data, "heritability", "parent_offspring_liability1_regression", "r_squared"
        ),
        # Parent-offspring regressions (trait 2)
        "parent_offspring_A2_slope": get_nested(
            data, "heritability", "parent_offspring_A2_regression", "slope"
        ),
        "parent_offspring_A2_r2": get_nested(
            data, "heritability", "parent_offspring_A2_regression", "r_squared"
        ),
        "parent_offspring_liability2_slope": get_nested(
            data, "heritability", "parent_offspring_liability2_regression", "slope"
        ),
        "parent_offspring_liability2_r2": get_nested(
            data, "heritability", "parent_offspring_liability2_regression", "r_squared"
        ),
    }

    return row


def main(validation_files: list[str], output_path: str) -> None:
    """Gather all validation results into a TSV file."""
    rows = []
    for validation_path in validation_files:
        row = extract_metrics(validation_path)
        rows.append(row)

    # Sort by scenario name, then by rep
    rows.sort(key=lambda x: (x["scenario"], x["rep"]))

    logger.info("Gathered %d validation results -> %s", len(rows), output_path)

    # Write TSV
    if rows:
        columns = list(rows[0].keys())
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("\t".join(columns) + "\n")
            for row in rows:
                values = []
                for col in columns:
                    val = row[col]
                    if val is None:
                        values.append("")
                    elif isinstance(val, float):
                        values.append(f"{val:.4g}")
                    else:
                        values.append(str(val))
                f.write("\t".join(values) + "\n")


def cli() -> None:
    """Command-line interface for gathering validation results."""
    from sim_ace.cli_base import add_logging_args, init_logging
    parser = argparse.ArgumentParser(description="Gather validation results into TSV")
    add_logging_args(parser)
    parser.add_argument("validations", nargs="+", help="Validation YAML paths")
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    init_logging(args)

    main(args.validations, args.output)
