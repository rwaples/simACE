"""Aggregate the per-rule Snakemake benchmark TSVs written by ``bench_pipeline.sh``.

Reports a median and range per rule for both wall time and sampled peak RSS,
across every sweep and replicate. Sweep 1 is reported separately because it is
cold: numba compiles on first call, so mixing it into the medians understates
steady-state performance.

Run over the default output directory::

    python tools/bench_aggregate.py [BENCH_LOGS_DIR]

Read the RSS column with care. Snakemake samples peak RSS on an interval and
misses transient spikes -- see ``bench_memprobe.sh`` for figures you can size a
job against.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

#: Pipeline order, so the table reads as the DAG runs rather than alphabetically.
RULE_ORDER = [
    "emit_params",
    "simulate",
    "phenotype",
    "censor_weibull",
    "ascertainment",
    "analyze",
    "plot_phenotype",
    "assemble_atlas",
]


def read_tsv(path: Path) -> tuple[float, float] | None:
    """Return ``(seconds, max_rss_mb)`` from one Snakemake benchmark TSV."""
    rows = path.read_text().splitlines()
    if len(rows) < 2:
        return None
    fields = rows[1].split("\t")
    return float(fields[0]), float(fields[2])


def collect(logs: Path, scenario: str, *, warm_only: bool) -> dict[str, list[tuple[float, float]]]:
    """Map each rule to its ``(seconds, rss_mb)`` samples across sweeps and replicates."""
    samples: dict[str, list[tuple[float, float]]] = {}
    for sweep_dir in sorted(logs.glob("tsv/sweep*")):
        if warm_only and sweep_dir.name == "sweep1":
            continue
        for tsv in sorted((sweep_dir / scenario).rglob("*.tsv")):
            value = read_tsv(tsv)
            if value is not None:
                samples.setdefault(tsv.stem, []).append(value)
    return samples


def summarize(values: list[float]) -> str:
    """Format a median with its min-max range, or the bare value for a single sample."""
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.median(values):.2f} ({min(values):.2f}-{max(values):.2f})"


def scenarios_in(logs: Path) -> list[str]:
    """List the scenarios present, in the order the sweeps ran them.

    Run order comes from ``wallclock.tsv`` so the tables read smallest-first as
    configured, rather than alphabetically (which would put 100K before 10K).
    """
    order: list[str] = []
    wallclock = logs / "wallclock.tsv"
    if wallclock.exists():
        for line in wallclock.read_text().splitlines()[1:]:
            name = line.split("\t")[1]
            if name not in order:
                order.append(name)
    present = {d.name for sweep in logs.glob("tsv/sweep*") for d in sweep.iterdir() if d.is_dir()}
    return [s for s in order if s in present] + sorted(present - set(order))


def report(logs: Path, *, warm_only: bool) -> None:
    """Print one table per scenario for either the warm sweeps or all of them."""
    label = "WARM (sweep 2 onward)" if warm_only else "ALL SWEEPS (sweep 1 is cold)"
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    for scenario in scenarios_in(logs):
        data = collect(logs, scenario, warm_only=warm_only)
        if not data:
            continue
        print(f"\n--- {scenario} ---")
        print(f"{'rule':<22}{'n':>3}  {'seconds  median (min-max)':<30}{'peak RSS MB  median':>24}")
        ordered = RULE_ORDER + sorted(set(data) - set(RULE_ORDER))
        for rule in ordered:
            if rule not in data:
                continue
            secs = [s for s, _ in data[rule]]
            rss = [r for _, r in data[rule]]
            print(f"{rule:<22}{len(secs):>3}  {summarize(secs):<30}{summarize(rss):>24}")


def main(argv: list[str]) -> int:
    """Aggregate the benchmark TSVs under the given directory."""
    logs = Path(argv[1]) if len(argv) > 1 else Path("bench-logs")
    if not (logs / "tsv").is_dir():
        print(f"no benchmark snapshots under {logs}/tsv -- run tools/bench_pipeline.sh first", file=sys.stderr)
        return 1
    wallclock = logs / "wallclock.tsv"
    if wallclock.exists():
        print("=== whole-scenario wall clock ===")
        print(wallclock.read_text().rstrip())
    report(logs, warm_only=False)
    report(logs, warm_only=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
