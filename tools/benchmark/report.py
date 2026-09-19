"""Benchmark summaries and baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tools.benchmark.model import BenchmarkError, summary_key

if TYPE_CHECKING:
    from tools.benchmark.model import BenchmarkRun


@dataclass(frozen=True)
class Comparison:
    """One baseline-to-candidate comparison."""

    rows: tuple[dict[str, Any], ...]
    incompatibilities: tuple[str, ...]
    regressions: int


_COMPATIBILITY_PATHS: tuple[tuple[str, ...], ...] = (
    ("schema",),
    ("inputs", "pixi_lock_sha256"),
    ("inputs", "config_files_sha256"),
    ("inputs", "scenarios"),
    ("host", "system"),
    ("host", "machine"),
    ("host", "cpu_model"),
    ("host", "physical_cores"),
    ("host", "logical_cores"),
    ("tools", "pixi"),
    ("tools", "python"),
    ("tools", "snakemake"),
    ("execution", "folder"),
    ("execution", "scenarios"),
    ("execution", "cores"),
    ("execution", "cache_mode"),
    ("execution", "sample_interval_seconds"),
    ("execution", "thread_environment"),
    ("execution", "targets"),
)


def _get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        value = value[key]
    return value


def compatibility_mismatches(baseline: BenchmarkRun, candidate: BenchmarkRun) -> list[str]:
    """Describe critical provenance fields that differ."""
    mismatches: list[str] = []
    for path in _COMPATIBILITY_PATHS:
        left = _get(baseline.manifest, path)
        right = _get(candidate.manifest, path)
        if left != right:
            mismatches.append(f"{'.'.join(path)}: baseline={left!r}, candidate={right!r}")
    return mismatches


def _require_complete(run: BenchmarkRun) -> None:
    if run.results["status"] != "complete":
        raise BenchmarkError(f"benchmark run {run.directory} has status {run.results['status']!r}")
    pipeline = [row for row in run.results["summaries"] if row.get("kind") == "pipeline"]
    if not pipeline or any((row.get("wall_seconds") or {}).get("n", 0) < 3 for row in pipeline):
        raise BenchmarkError(f"benchmark run {run.directory} has fewer than three measured repetitions")


def compare_runs(
    baseline: BenchmarkRun,
    candidate: BenchmarkRun,
    *,
    time_threshold_percent: float,
    memory_threshold_percent: float,
    allow_incompatible: bool,
) -> Comparison:
    """Compare matching summary medians and count threshold regressions."""
    _require_complete(baseline)
    _require_complete(candidate)
    mismatches = compatibility_mismatches(baseline, candidate)
    if mismatches and not allow_incompatible:
        raise BenchmarkError("incompatible benchmark provenance:\n  " + "\n  ".join(mismatches))

    baseline_rows = {summary_key(row): row for row in baseline.results["summaries"]}
    candidate_rows = {summary_key(row): row for row in candidate.results["summaries"]}
    if baseline_rows.keys() != candidate_rows.keys():
        missing_candidate = sorted(set(baseline_rows) - set(candidate_rows), key=str)
        missing_baseline = sorted(set(candidate_rows) - set(baseline_rows), key=str)
        raise BenchmarkError(
            f"summary keys differ; missing from candidate={missing_candidate}, missing from baseline={missing_baseline}"
        )

    rows: list[dict[str, Any]] = []
    regressions = 0
    for key in sorted(baseline_rows, key=str):
        for metric, threshold in (
            ("wall_seconds", time_threshold_percent),
            ("peak_rss_kb", memory_threshold_percent),
        ):
            left = baseline_rows[key].get(metric)
            right = candidate_rows[key].get(metric)
            if left is None or right is None:
                continue
            baseline_median = float(left["median"])
            candidate_median = float(right["median"])
            absolute = candidate_median - baseline_median
            percent = (absolute / baseline_median * 100.0) if baseline_median else float("inf")
            regressed = percent > threshold
            regressions += int(regressed)
            rows.append(
                {
                    "kind": key[0],
                    "scenario": key[1],
                    "rule": key[2],
                    "metric": metric,
                    "baseline": left,
                    "candidate": right,
                    "absolute_change": absolute,
                    "percent_change": percent,
                    "threshold_percent": threshold,
                    "regression": regressed,
                }
            )
    return Comparison(tuple(rows), tuple(mismatches), regressions)


def print_summary(run: BenchmarkRun) -> None:
    """Print the stored medians and observed ranges."""
    print(f"run: {run.results['run_id']} ({run.results['status']})")
    print(f"{'scope':<10}{'scenario':<22}{'rule':<18}{'metric':<18}{'n':>4}{'median':>12}{'range':>24}")
    for row in run.results["summaries"]:
        for metric in ("wall_seconds", "peak_rss_kb"):
            value = row.get(metric)
            if value is None:
                continue
            label = row.get("rule") or "all"
            observed = f"{value['min']:.2f}..{value['max']:.2f}"
            print(
                f"{row['kind']:<10}{row['scenario']:<22}{label:<18}{metric:<18}"
                f"{value['n']:>4}{value['median']:>12.2f}{observed:>24}"
            )


def print_comparison(comparison: Comparison) -> None:
    """Print a baseline comparison table."""
    for mismatch in comparison.incompatibilities:
        print(f"WARNING incompatible: {mismatch}")
    print(f"{'scenario':<22}{'rule':<18}{'metric':<18}{'baseline':>12}{'candidate':>12}{'change':>11}{'gate':>9}")
    for row in comparison.rows:
        rule = row["rule"] or "all"
        marker = "FAIL" if row["regression"] else "ok"
        print(
            f"{row['scenario']:<22}{rule:<18}{row['metric']:<18}"
            f"{row['baseline']['median']:>12.2f}{row['candidate']['median']:>12.2f}"
            f"{row['percent_change']:>10.2f}%{marker:>9}"
        )
