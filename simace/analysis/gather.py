"""Gather per-replicate report summaries into a single wide TSV file."""

__all__ = ["extract_metrics"]

import argparse
import csv
import logging
import math
import platform
import re
from pathlib import Path
from typing import Any

from simace.analysis.report_schema import REPORT_SUMMARY_REGISTRY
from simace.core.yaml_io import load_yaml

logger = logging.getLogger(__name__)

_REPORT_PATH_RE = re.compile(r"results/([^/]+)/([^/]+)/rep(\d+)/report\.yaml")


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

    REPORT_SUMMARY_REGISTRY paths are relative to the report root and resolve
    into the ``scopes`` / ``observed`` / ``truth`` / ``estimators`` groups (ADR
    0008). Identity, parameters, and the quality summary are read inline from
    the path / ``inputs`` / ``quality_checks``.
    """
    data = load_yaml(report_path)

    report_path = str(report_path).replace("\\", "/")

    match = _REPORT_PATH_RE.search(report_path)
    if match:
        folder, scenario, rep_str = match.group(1), match.group(2), match.group(3)
        rep = int(rep_str)
        bench_path: Path | None = Path(f"benchmarks/{folder}/{scenario}/rep{rep_str}/simulate.tsv")
    else:
        folder = "unknown"
        scenario = "unknown"
        rep = 1
        bench_path = None

    simulate_seconds = None
    simulate_max_rss_mb = None
    if bench_path is not None and bench_path.exists():
        with open(bench_path, encoding="utf-8", newline="") as bf:
            first_row = next(csv.DictReader(bf, delimiter="\t"), None)
        if first_row is not None:
            simulate_seconds = float(first_row["s"])
            # Windows benchmarks have no max_rss column; fall back to a sentinel.
            simulate_max_rss_mb = 1.0 if platform.system() == "Windows" else float(first_row["max_rss"])

    params = _get_nested(data, "inputs", "parameters", default={})
    summary = _get_nested(data, "quality_checks", "summary", default={})

    row: dict[str, Any] = {
        "folder": folder,
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
        # reports predating this column still gather cleanly.
        # ``expected_twin_rate`` is sourced from the report via
        # REPORT_SUMMARY_REGISTRY (see report_schema.py) — validate_twins emits
        # it for both standard (= p_mztwin) and WF (= 0) branches.
        "mating_model": params.get("mating_model", "standard"),
        "p_mztwin": params.get("p_mztwin"),
        "mating_lambda": params.get("mating_lambda"),
        "assort1": params.get("assort1"),
        "assort2": params.get("assort2"),
        "seed": params.get("seed"),
        "quality_passed": summary.get("passed"),
        "checks_failed": summary.get("n_failed"),
        "quality_n_warn": summary.get("n_warn"),
    }
    for spec in REPORT_SUMMARY_REGISTRY:
        row[spec.column] = _get_nested(data, *spec.path)
    # Benchmark timing and memory live alongside parameters, not in the YAML
    row["simulate_seconds"] = simulate_seconds
    row["simulate_max_rss_mb"] = simulate_max_rss_mb
    return row


def main(report_files: list[str], output_path: str) -> None:
    """Gather report summaries from many replicates into a wide TSV file."""
    rows = []
    for report_path in report_files:
        row = extract_metrics(report_path)
        rows.append(row)

    # Sort by scenario name, then by rep
    rows.sort(key=lambda x: (x["scenario"], x["rep"]))

    logger.info("Gathered %d report summaries -> %s", len(rows), output_path)

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
                        # NaN has no TSV spelling readers agree on: pandas took the
                        # literal "nan" as missing, polars infers the whole column as
                        # String (ADR 0015). Emit the same empty field used for None;
                        # ±inf round-trips as a float either way.
                        values.append("" if math.isnan(val) else f"{val:.4g}")
                    else:
                        values.append(str(val))
                f.write("\t".join(values) + "\n")


def cli() -> None:
    """Command-line interface for gathering report summaries."""
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging

    parser = argparse.ArgumentParser(description="Gather report summaries into TSV")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("reports", nargs="+", help="report.yaml paths")
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    init_logging(args)

    main(args.reports, args.output)
