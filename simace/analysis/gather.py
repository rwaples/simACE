"""Gather validation results from all scenarios into a single TSV file."""

__all__ = ["extract_metrics"]

import argparse
import csv
import logging
import platform
import re
from pathlib import Path
from typing import Any

from simace.analysis.validation_schema import METRIC_REGISTRY
from simace.core.yaml_io import load_yaml

logger = logging.getLogger(__name__)

_VALIDATION_PATH_RE = re.compile(r"results/([^/]+)/([^/]+)/rep(\d+)/report\.yaml")


def _get_nested(d: Any, *keys: str, default: Any = None) -> Any:
    """Traverse nested dicts by key path, returning default if any key is missing."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def extract_metrics(report_path: str) -> dict[str, Any]:
    """Extract key metrics from a curated v2 ``report.yaml`` file.

    METRIC_REGISTRY paths are relative to the report root and resolve into the
    ``truth`` / ``estimators`` groups (ADR 0008). Parameters and the quality
    summary are read inline from ``inputs`` / ``quality_checks``.
    """
    data = load_yaml(report_path)

    report_path = str(report_path).replace("\\", "/")

    match = _VALIDATION_PATH_RE.search(report_path)
    if match:
        folder, scenario, rep_str = match.group(1), match.group(2), match.group(3)
        rep = int(rep_str)
        bench_path = Path(f"benchmarks/{folder}/{scenario}/rep{rep_str}/simulate.tsv")
    else:
        scenario = "unknown"
        rep = 1
        bench_path = Path("")

    simulate_seconds = None
    simulate_max_rss_mb = None
    if bench_path.exists():
        with open(bench_path, encoding="utf-8", newline="") as bf:
            reader = csv.DictReader(bf, delimiter="\t")
            for row_b in reader:
                simulate_seconds = float(row_b["s"])

                if platform.system() == "Windows":
                    # Windows does not support max_rss
                    simulate_max_rss_mb = float(1)
                else:
                    # Linux/macOS → normal
                    simulate_max_rss_mb = float(row_b["max_rss"])

                break

    params = _get_nested(data, "inputs", "parameters", default={})
    summary = _get_nested(data, "quality_checks", "summary", default={})

    row: dict[str, Any] = {
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
        # Population parameters.  ``mating_model`` defaults to "standard" so
        # validation YAMLs predating this column still gather cleanly.
        # ``expected_twin_rate`` is sourced from the validation YAML via
        # METRIC_REGISTRY (see validation_schema.py) — validate_twins emits
        # it for both standard (= p_mztwin) and WF (= 0) branches.
        "mating_model": params.get("mating_model", "standard"),
        "p_mztwin": params.get("p_mztwin"),
        "mating_lambda": params.get("mating_lambda"),
        "assort1": params.get("assort1"),
        "assort2": params.get("assort2"),
        "seed": params.get("seed"),
        "checks_failed": summary.get("n_failed"),
    }
    for spec in METRIC_REGISTRY:
        row[spec.column] = _get_nested(data, *spec.path)
    # Benchmark timing and memory live alongside parameters, not in the YAML
    row["simulate_seconds"] = simulate_seconds
    row["simulate_max_rss_mb"] = simulate_max_rss_mb
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
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Gather validation results into TSV")
    add_logging_args(parser)
    parser.add_argument("validations", nargs="+", help="Validation YAML paths")
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    init_logging(args)

    main(args.validations, args.output)
