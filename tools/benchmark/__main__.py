"""Run, summarize, and compare reproducible simACE pipeline benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.benchmark.model import BenchmarkError, read_run
from tools.benchmark.report import compare_runs, print_comparison, print_summary
from tools.benchmark.runner import CommandFailed, RunConfig, run_benchmark

PROFILES = {
    "smoke": ("test", ("small_test",)),
    "release": ("base", ("baseline10K", "baseline100K", "baseline1M")),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.benchmark", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a new benchmark into an immutable directory")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", choices=sorted(PROFILES))
    source.add_argument("--folder")
    run.add_argument("--scenarios", nargs="+", help="required with --folder")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--jobs", type=int, default=1, help="reps computed concurrently by simace run")
    run.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    run.add_argument("--order-seed", type=int, default=0)
    run.add_argument("--sample-interval", type=float, default=0.25, metavar="SECONDS")
    run.add_argument("--out", type=Path)

    summarize = commands.add_parser("summarize", help="print stored medians and observed ranges")
    summarize.add_argument("run_directory", type=Path)

    compare = commands.add_parser("compare", help="compare a candidate run with a compatible baseline")
    compare.add_argument("baseline_directory", type=Path)
    compare.add_argument("candidate_directory", type=Path)
    compare.add_argument("--time-threshold-percent", type=float, default=5.0)
    compare.add_argument("--memory-threshold-percent", type=float, default=5.0)
    compare.add_argument("--allow-incompatible", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.profile:
        if args.scenarios:
            raise BenchmarkError("--scenarios cannot be combined with --profile")
        folder, scenarios = PROFILES[args.profile]
    else:
        if not args.scenarios:
            raise BenchmarkError("--scenarios is required with --folder")
        folder, scenarios = args.folder, tuple(args.scenarios)
    config = RunConfig(
        folder=folder,
        scenarios=tuple(scenarios),
        profile=args.profile,
        repeats=args.repeats,
        jobs=args.jobs,
        cache_mode=args.cache_mode,
        order_seed=args.order_seed,
        sample_interval_seconds=args.sample_interval,
        output=args.out,
        cli_argv=tuple(sys.argv),
    )
    output = run_benchmark(config)
    print(f"benchmark complete: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the selected benchmark subcommand."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "summarize":
            print_summary(read_run(args.run_directory))
            return 0
        if args.time_threshold_percent < 0 or args.memory_threshold_percent < 0:
            raise BenchmarkError("regression thresholds must be nonnegative")
        baseline = read_run(args.baseline_directory)
        candidate = read_run(args.candidate_directory)
        comparison = compare_runs(
            baseline,
            candidate,
            time_threshold_percent=args.time_threshold_percent,
            memory_threshold_percent=args.memory_threshold_percent,
            allow_incompatible=args.allow_incompatible,
        )
        print_comparison(comparison)
        return 1 if comparison.regressions else 0
    except CommandFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.returncode if 0 < exc.returncode < 126 else 1
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("benchmark interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
