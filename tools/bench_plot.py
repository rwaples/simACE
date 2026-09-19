"""Plot how simACE's runtime and peak memory scale with pedigree size.

Reads one run written by ``python -m tools.benchmark run`` and renders
one figure: wall-clock time on the left, peak RSS on the right, two bars per
benchmark point.

    pixi run python -m tools.bench_plot --bench-dir bench-logs/<run-id> --out notes/bench/scaling.png

The two bars are ``simulate`` and the pipeline total. The gap between them is
almost entirely ``analyze``, which carries the cost: 144 s of the 157 s core
pipeline at 10M. Drawing it as a third bar said the same thing twice, since it
tracked the total so closely the two were hard to tell apart. The printed table
still lists every stage.

The two totals are computed differently on purpose. The pipeline is a chain, so
stage times **add**. Peak memory does not: only one stage is resident at a time,
so the total is the **maximum** over stages. Summing it would overstate the
requirement by roughly the number of stages.

Time is read from Snakemake's own benchmark TSVs, which are stopwatches: the
``s`` column, elapsed wall seconds, not the ``cpu_time`` column beside it. Runs
are on 4 cores, so the two differ wherever a stage threads, and wall time is the
one a reader can plan against.
Memory is read from the result document's 4 Hz process-group samples because
Snakemake's sampled RSS misses transient spikes; ``--show-gap`` draws both so
the difference is visible.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tools.benchmark.model import STAGES, BenchmarkError, read_run

if TYPE_CHECKING:
    from tools.benchmark.model import StageSpec

REPO = Path(__file__).resolve().parents[1]

KB_PER_GB = 1048576.0


@dataclass(frozen=True)
class Group:
    """A set of stages drawn as one bar.

    Attributes:
        key: Identifier.
        label: Legend text.
        color: Bar colour.
        members: Stage keys in the group, or None for every stage that survived
            ``--exclude``.
    """

    key: str
    label: str
    color: str
    members: tuple[str, ...] | None


#: The plot shows the simulation stage and the pipeline total, and nothing else.
#: ``analyze`` is the whole gap between them (144 s of 157 s at 10M), so its own
#: bar sat a hair under "all stages" at every point and cost a legend entry to
#: say what the gap already says. The small stages (params, phenotype, censor,
#: ascertain) are together under 8 s even at 10M. Every stage is still in the
#: printed table and still inside "all".
GROUPS: tuple[Group, ...] = (
    Group("simulate", "simulate", "#0B5377", ("simulate",)),
    Group("all", "all stages", "#8FA3B0", None),
)


@dataclass(frozen=True)
class Point:
    """One benchmark point: a scenario at a given population size.

    Attributes:
        scenario: Scenario name.
        individuals: Total pedigree individuals, ``N * G_ped``.
        wall_s: Median wall seconds per stage key across measured repetitions.
        peak_gb: Median sampled peak RSS in GB per measured repetition.
        snake_gb: What Snakemake itself reported, per stage key, for ``--show-gap``.
        tree_peak_gb: Median whole-run summed-tree peak, the machine footprint.
    """

    scenario: str
    individuals: int
    wall_s: dict[str, float]
    peak_gb: dict[str, float]
    snake_gb: dict[str, float]
    tree_peak_gb: float


def build_points(bench_dir: Path) -> list[Point]:
    """Assemble one Point per scenario from the benchmark logs.

    Args:
        bench_dir: One immutable benchmark run directory.

    Returns:
        Points sorted by population size.

    Raises:
        BenchmarkError: If the run is incomplete or has no usable summaries.
    """
    run = read_run(bench_dir)
    if run.results["status"] != "complete":
        raise BenchmarkError(f"benchmark run has status {run.results['status']!r}")
    summaries = {
        (row["scenario"], row.get("rule")): row
        for row in run.results["summaries"]
        if row["kind"] in {"pipeline", "rule"}
    }
    executions = [
        item for item in run.results["executions"] if item["phase"] == "measured" and item["status"] == "complete"
    ]

    points: list[Point] = []
    scenario_inputs = run.manifest["inputs"]["scenarios"]
    for scenario in run.manifest["execution"]["scenarios"]:
        params = scenario_inputs[scenario]
        individuals = int(params["N"]) * int(params["G_ped"])
        wall: dict[str, float] = {}
        peak: dict[str, float] = {}
        for stage in STAGES:
            row = summaries.get((scenario, stage.key))
            if row is None:
                continue
            if row["wall_seconds"] is not None:
                wall[stage.key] = float(row["wall_seconds"]["median"])
            if row["peak_rss_kb"] is not None:
                peak[stage.key] = float(row["peak_rss_kb"]["median"]) / KB_PER_GB
        snake_samples: dict[str, list[float]] = {}
        for execution in executions:
            if execution["scenario"] != scenario:
                continue
            for key, rule in execution["rules"].items():
                for sample in rule["snakemake"]:
                    if sample["max_rss_mb"] is not None:
                        snake_samples.setdefault(key, []).append(float(sample["max_rss_mb"]) / 1024.0)
        pipeline = summaries.get((scenario, None))
        if pipeline is None or pipeline["peak_rss_kb"] is None:
            raise BenchmarkError(f"scenario {scenario!r} has no pipeline memory summary")
        points.append(
            Point(
                scenario=scenario,
                individuals=individuals,
                wall_s=wall,
                peak_gb=peak,
                snake_gb={key: statistics.median(values) for key, values in snake_samples.items()},
                tree_peak_gb=float(pipeline["peak_rss_kb"]["median"]) / KB_PER_GB,
            )
        )
    return sorted(points, key=lambda p: p.individuals)


def human_count(n: int) -> str:
    """Render an individual count as 200K / 2M / 20M.

    Args:
        n: Number of individuals.

    Returns:
        Short label.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    return f"{n / 1_000:g}K"


def group_stages(group: Group, included: list[StageSpec]) -> list[StageSpec]:
    """Resolve a group to the stages it covers.

    Args:
        group: The group.
        included: Stages left after ``--exclude``.

    Returns:
        The stages in the group, in pipeline order.
    """
    if group.members is None:
        return included
    return [stage for stage in included if stage.key in group.members]


def group_wall(point: Point, group: Group, included: list[StageSpec]) -> float:
    """Sum a group's wall time, because the pipeline is a chain and times add.

    Args:
        point: Benchmark point.
        group: The group.
        included: Stages left after ``--exclude``.

    Returns:
        Seconds.
    """
    return sum(point.wall_s.get(stage.key, 0.0) for stage in group_stages(group, included))


def group_peak(point: Point, group: Group, included: list[StageSpec], reported: bool = False) -> float:
    """Take a group's peak memory as the maximum, not the sum.

    Only one stage is resident at a time, so the machine's requirement is the
    largest stage, not their total.

    Args:
        point: Benchmark point.
        group: The group.
        included: Stages left after ``--exclude``.
        reported: Read Snakemake's own figure instead of the sampled one.

    Returns:
        Gigabytes.
    """
    source = point.snake_gb if reported else point.peak_gb
    return max((source.get(stage.key, 0.0) for stage in group_stages(group, included)), default=0.0)


def apply_log_axis(ax: object, values: list[float]) -> None:
    """Put an axis on a log scale with ticks a reader can actually read off it.

    Decade-only ticks are too sparse for the range these benchmarks cover, so
    ticks land on the 1-2-5 series and are labelled as plain numbers rather than
    powers. Gridlines matter more here than on a linear axis, where bar height
    alone carries the value.

    Args:
        ax: The axis to configure.
        values: Every value plotted on it, used to pick the limits.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    from simace.plotting.plot_style import enable_value_gridlines

    positive = [v for v in values if v > 0]
    low = min(positive) if positive else 0.1
    high = max(positive) if positive else 1.0
    ax.set_yscale("log")
    # Bars are drawn from zero, which is off a log axis entirely; the floor is
    # what they visually start from, so it is set below the smallest bar rather
    # than left to autoscale onto it.
    ax.set_ylim(low / 2.5, high * 2.2)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=15))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    enable_value_gridlines(ax)


def render(
    points: list[Point],
    out_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    show_gap: bool,
    exclude: frozenset[str] = frozenset(),
) -> None:
    """Draw the two-panel scaling figure.

    Args:
        points: Benchmark points, smallest first.
        out_path: PNG destination; parent directories are created.
        figsize: Figure size in inches.
        dpi: Output resolution.
        show_gap: Also draw what Snakemake reported, to expose its undercount.
        exclude: Stage keys to leave out. Passing ``plots`` and ``atlas`` gives
            the simulation pipeline alone; atlas rendering is a fixed cost that
            does not scale with N and otherwise dominates the small points.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from simace.plotting.plot_style import apply_nature_style

    apply_nature_style()
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=figsize)
    x = list(range(len(points)))
    labels = [human_count(p.individuals) for p in points]
    included = [
        stage
        for stage in STAGES
        if not ({stage.key, stage.label} & exclude)
        and any(stage.key in p.wall_s or stage.key in p.peak_gb for p in points)
    ]

    width = 0.8 / len(GROUPS)
    every_time: list[float] = []
    every_peak: list[float] = []
    for index, group in enumerate(GROUPS):
        offsets = [xi - 0.4 + width * (index + 0.5) for xi in x]
        times = [group_wall(p, group, included) for p in points]
        peaks = [group_peak(p, group, included) for p in points]
        every_time.extend(times)
        every_peak.extend(peaks)
        ax_time.bar(offsets, times, width=width * 0.86, color=group.color, zorder=3)
        ax_mem.bar(offsets, peaks, width=width * 0.86, color=group.color, zorder=3)
        if show_gap:
            reported = [group_peak(p, group, included, reported=True) for p in points]
            every_peak.extend(reported)
            ax_mem.plot(offsets, reported, linestyle="none", marker="_", markersize=7, color="#CC4444", zorder=4)
        if group.members is None:
            for xi, seconds, gigabytes in zip(offsets, times, peaks, strict=True):
                ax_time.text(xi, seconds, f"{seconds:.0f}s", ha="center", va="bottom", fontsize=9)
                ax_mem.text(xi, gigabytes, f"{gigabytes:.1f}", ha="center", va="bottom", fontsize=9)

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels)
    ax_time.set_xlabel("Pedigree individuals")
    ax_time.set_ylabel("Wall-clock time (s)")
    ax_time.set_title("Runtime")
    apply_log_axis(ax_time, every_time)

    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(labels)
    ax_mem.set_xlabel("Pedigree individuals")
    ax_mem.set_ylabel("Peak resident memory (GB)")
    ax_mem.set_title("Peak RAM")
    apply_log_axis(ax_mem, every_peak)

    handles = [Patch(facecolor=group.color, label=group.label) for group in GROUPS]
    if show_gap:
        handles.append(plt.Line2D([], [], marker="_", color="#CC4444", linestyle="none", label="Snakemake reported"))
    ax_time.legend(handles=handles, loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def report(points: list[Point], exclude: frozenset[str] = frozenset()) -> None:
    """Print the numbers behind the figure, including Snakemake's undercount.

    Every stage is listed even when the figure groups them, since the per-stage
    table is what you diagnose a regression from.

    Args:
        points: Benchmark points.
        exclude: Stages omitted from the figure, so the group rows below match it.
    """
    for point in points:
        included = [s for s in STAGES if not ({s.key, s.label} & exclude) and s.key in point.wall_s]
        print(f"\n=== {point.scenario}: {point.individuals:,} individuals ===")
        print(f"{'stage':<16}{'wall_s':>9}{'peak_GB':>10}{'snake_GB':>10}{'ratio':>8}")
        for stage in STAGES:
            if stage.key not in point.wall_s and stage.key not in point.peak_gb:
                continue
            measured = point.peak_gb.get(stage.key, 0.0)
            reported = point.snake_gb.get(stage.key, 0.0)
            ratio = f"{measured / reported:.2f}x" if reported > 0 else "-"
            marker = " " if stage in included else "-"
            print(
                f"{marker}{stage.label:<15}{point.wall_s.get(stage.key, 0.0):>9.1f}"
                f"{measured:>10.2f}{reported:>10.2f}{ratio:>8}"
            )
        print(f"{'-' * 53}")
        for group in GROUPS:
            seconds = group_wall(point, group, included)
            gigabytes = group_peak(point, group, included)
            snake = group_peak(point, group, included, reported=True)
            ratio = f"{gigabytes / snake:.2f}x" if snake > 0 else "-"
            print(f"{group.label:<16}{seconds:>9.1f}{gigabytes:>10.2f}{snake:>10.2f}{ratio:>8}")
        print(f"{'whole run':<16}{'':>9}{point.tree_peak_gb:>10.2f}{'':>10}{'':>8}")


def main() -> int:
    """Parse arguments, build the figure, print the table.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-dir", type=Path, required=True, help="one tools.benchmark run directory")
    parser.add_argument("--out", type=Path, default=REPO / "notes" / "bench" / "scaling.png", help="output PNG")
    parser.add_argument("--figsize", type=float, nargs=2, default=(9.0, 3.6), metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--show-gap", action="store_true", help="mark what Snakemake reported next to the measured peak"
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], metavar="STAGE", help="stages to omit by key or label, e.g. plots atlas"
    )
    args = parser.parse_args()

    try:
        points = build_points(args.bench_dir)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report(points, frozenset(args.exclude))
    render(points, args.out, tuple(args.figsize), args.dpi, args.show_gap, frozenset(args.exclude))
    print(f"\nwrote {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
