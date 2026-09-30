"""Versioned benchmark documents and pure summary helpers."""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_SCHEMA = "simace_pipeline_benchmark_manifest"
RESULTS_SCHEMA = "simace_pipeline_benchmark_results"
SCHEMA_VERSION = 1


class BenchmarkError(RuntimeError):
    """Raised when a benchmark command cannot produce a trustworthy result."""


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage: its stable summary key, its ``simace`` subcommand, and a short label.

    ``key`` predates ``simace run`` and is kept so summaries stay comparable
    with benchmarks recorded before it.
    """

    key: str
    command: str
    label: str


STAGES: tuple[StageSpec, ...] = (
    StageSpec("simulate", "simulate", "simulate"),
    StageSpec("phenotype", "phenotype", "phenotype"),
    StageSpec("censor", "censor", "censor"),
    StageSpec("ascertainment", "ascertain", "ascertain"),
    StageSpec("analyze", "analyze", "analyze"),
    StageSpec("plot_phenotype", "plot", "plots"),
    StageSpec("assemble_atlas", "atlas", "atlas"),
)

COMMAND_TO_STAGE = {stage.command: stage.key for stage in STAGES}


@dataclass(frozen=True)
class BenchmarkRun:
    """Parsed manifest and result documents from one invocation."""

    directory: Path
    manifest: dict[str, Any]
    results: dict[str, Any]


def _check_document(data: object, *, schema: str, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise BenchmarkError(f"{path} must contain a JSON object")
    marker = data.get("schema")
    expected = {"name": schema, "version": SCHEMA_VERSION}
    if marker != expected:
        raise BenchmarkError(f"{path} has unsupported schema {marker!r}; expected {expected!r}")
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise BenchmarkError(f"{path} has no run_id")
    return data


def read_run(directory: Path) -> BenchmarkRun:
    """Read and validate the two documents for a benchmark invocation."""
    directory = directory.resolve()
    try:
        manifest_raw = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        results_raw = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"missing benchmark document: {exc.filename}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid JSON in {exc.doc[:40]!r}: {exc}") from exc

    manifest = _check_document(manifest_raw, schema=MANIFEST_SCHEMA, path=directory / "manifest.json")
    results = _check_document(results_raw, schema=RESULTS_SCHEMA, path=directory / "results.json")
    if manifest["run_id"] != results["run_id"]:
        raise BenchmarkError(f"run_id differs between documents under {directory}")
    if results.get("status") not in {"running", "complete", "failed", "interrupted"}:
        raise BenchmarkError(f"{directory / 'results.json'} has invalid status {results.get('status')!r}")
    if not isinstance(results.get("executions"), list) or not isinstance(results.get("summaries"), list):
        raise BenchmarkError(f"{directory / 'results.json'} has invalid executions or summaries")
    return BenchmarkRun(directory, manifest, results)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write one JSON document atomically."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stats(values: list[float]) -> dict[str, float | int] | None:
    """Return a median and observed range, or None for no samples."""
    if not values:
        return None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def build_summaries(executions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize measured executions by scenario and canonical rule."""
    measured = [item for item in executions if item.get("phase") == "measured" and item.get("status") == "complete"]
    scenarios = sorted({str(item["scenario"]) for item in measured})
    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        selected = [item for item in measured if item["scenario"] == scenario]
        summaries.append(
            {
                "kind": "pipeline",
                "scenario": scenario,
                "rule": None,
                "wall_seconds": stats([float(item["wall_seconds"]) for item in selected]),
                "peak_rss_kb": stats([float(item["peak_summed_rss_kb"]) for item in selected]),
                "max_individual_rss_kb": stats([float(item["max_individual_rss_kb"]) for item in selected]),
                "gnu_time_max_rss_kb": stats([float(item["gnu_time_max_rss_kb"]) for item in selected]),
            }
        )

        rules = sorted({rule for item in selected for rule in item.get("rules", {})})
        for rule in rules:
            wall = [
                float(sample["wall_seconds"])
                for item in selected
                for sample in item.get("rules", {}).get(rule, {}).get("stage", [])
            ]
            rss = [
                float(item["rules"][rule]["peak_rss_kb"])
                for item in selected
                if item.get("rules", {}).get(rule, {}).get("peak_rss_kb") is not None
            ]
            summaries.append(
                {
                    "kind": "rule",
                    "scenario": scenario,
                    "rule": rule,
                    "wall_seconds": stats(wall),
                    "peak_rss_kb": stats(rss),
                }
            )
    return summaries


def summary_key(summary: dict[str, Any]) -> tuple[str, str, str | None]:
    """Return the stable comparison key for one summary row."""
    return str(summary["kind"]), str(summary["scenario"]), summary.get("rule")
