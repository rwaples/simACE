"""Gather per-replicate report summaries into a single wide TSV file."""

__all__ = ["extract_metrics"]

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from simace.analysis.report_schema import REPORT_SUMMARY_REGISTRY
from simace.analysis.stats.onset_tau import CELL_CATEGORIES
from simace.analysis.validate.on_demand import load_validation
from simace.analysis.validate.schema_file import one_warning_for_refused
from simace.core.relationships import C_TRANSMISSION_KEYS, RELATIONSHIP_TYPES
from simace.core.yaml_io import load_yaml

logger = logging.getLogger(__name__)

_REPORT_PATH_RE = re.compile(r"([^/]+)/([^/]+)/rep(\d+)/report\.yaml$")


def _simulate_timing(timing_path: Path) -> tuple[float | None, float | None]:
    """Return the simulate stage's wall seconds and peak RSS (MB) from a rep's ``timing.tsv``."""
    if not timing_path.exists():
        return None, None
    with open(timing_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["stage"] == "simulate":
                return float(row["wall_s"]), float(row["max_rss_mb"])
    return None, None


def _get_nested(d: Any, *keys: str, default: Any = None) -> Any:
    """Traverse nested dicts by key path, returning default if any key is missing."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def _truth_columns(data: dict[str, Any]) -> dict[str, Any]:
    """Pooled recorded-pedigree truth per trait (ADR 0025): ``truth_r_{code}_trait{t}`` and the midparent slope.

    None (an empty TSV cell) for reports without C transmission, which carry no truth block.
    """
    traits = _get_nested(data, "truth", "recorded_pedigree", "traits", default={}) or {}
    row: dict[str, Any] = {}
    for t in (1, 2):
        trait = traits.get(f"trait{t}") or {}
        relationships = trait.get("relationships") or {}
        for code in RELATIONSHIP_TYPES:
            row[f"truth_r_{code}_trait{t}"] = (relationships.get(code) or {}).get("r")
        row[f"truth_midparent_slope_trait{t}"] = (trait.get("midparent_slope") or {}).get("slope")
    return row


def extract_metrics(report_path: str) -> dict[str, Any]:
    """Extract key metrics from a curated v2 ``report.yaml`` file.

    REPORT_SUMMARY_REGISTRY paths are relative to the report root and resolve
    into the ``scopes`` / ``observed`` / ``truth`` / ``estimators`` groups (ADR
    0008). Identity, parameters, and the quality summary are read inline from
    the path / ``inputs`` / ``quality_checks``.
    """
    data = load_yaml(report_path)

    match = _REPORT_PATH_RE.search(str(report_path).replace("\\", "/"))
    if match:
        folder, scenario, rep = match.group(1), match.group(2), int(match.group(3))
    else:
        folder, scenario, rep = "unknown", "unknown", 1

    simulate_seconds, simulate_max_rss_mb = _simulate_timing(Path(report_path).with_name("timing.tsv"))

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
        # Cross-trait correlations.  ``rE`` resolves to None for reports
        # predating the key, which the expected-value marker skips.
        "rA": params.get("rA"),
        "rC": params.get("rC"),
        "rE": params.get("rE"),
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
        # The 2x2 mate-correlation matrix (empty when unset): with only
        # cross-trait terms, assort1/assort2 stay 0 although mating assorts, and
        # the reference-line gate needs to see it. JSON is also a Python literal,
        # which plot_validation parses back like the per-generation dict cells.
        "assort_matrix": None if params.get("assort_matrix") is None else json.dumps(params["assort_matrix"]),
        # C transmission coefficients (ADR 0025); reports predating them were built with 0.
        **{key: params.get(key, 0.0) for key in C_TRANSMISSION_KEYS},
        "seed": params.get("seed"),
        "quality_passed": summary.get("passed"),
        "checks_failed": summary.get("n_failed"),
        "quality_n_warn": summary.get("n_warn"),
    }
    for spec in REPORT_SUMMARY_REGISTRY:
        row[spec.column] = _get_nested(data, *spec.path)
    row.update(_truth_columns(data))
    # Stage timing lives in the rep's timing.tsv, not in the report
    row["simulate_seconds"] = simulate_seconds
    row["simulate_max_rss_mb"] = simulate_max_rss_mb
    # Like timing.tsv, `simace validate`'s output is read from beside the report (a rep not validated has none).
    row.update(_on_demand_columns(Path(report_path).with_name("validation.yaml")))
    return row


#: Onset-τ cells with a z column, in report order (ADR 0027).
_ONSET_TAU_CELLS = (*(f"trait{t}_{code}" for t in (1, 2) for code in CELL_CATEGORIES), "within_person")


def _on_demand_columns(path: Path) -> dict[str, Any]:
    """Columns from ``simace validate``'s ``validation.yaml``: empty for a rep it has not validated.

    A file without the current schema counts as not validated (ADR 0028).
    Every row carries every column, since the TSV's columns are the first
    row's keys. Gather copies per-rep values and decides no verdict; a cell's
    z is present for scored cells only.
    """
    data = load_validation(path) or {}
    onset_tau = (data.get("checks") or {}).get("onset_tau") or {}
    summary = onset_tau.get("summary") or {}
    cells = onset_tau.get("cells") or {}
    row: dict[str, Any] = {
        "onset_tau_n_scored": summary.get("n_scored"),
        "onset_tau_max_abs_z": summary.get("max_abs_z"),
        "onset_tau_rep_bound": summary.get("z_bound"),
    }
    for name in _ONSET_TAU_CELLS:
        row[f"onset_tau_z_{name}"] = (cells.get(name) or {}).get("z")
    return row


def main(report_files: list[str], output_path: str) -> None:
    """Gather report summaries from many replicates into a wide TSV file."""
    with one_warning_for_refused():
        rows = [extract_metrics(report_path) for report_path in report_files]

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


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line interface for gathering report summaries."""
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging

    parser = argparse.ArgumentParser(prog=prog, description="Gather report summaries into TSV")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("reports", nargs="+", help="report.yaml paths")
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args(argv)

    init_logging(args)

    main(args.reports, args.output)
