"""The ``simace`` command line.

``simace <command> [args...]`` imports only the module that owns
``<command>`` and hands it the remaining arguments, so each stage starts
with the same imports it would have as a standalone script. Flags are
defined once, in that module's ``cli(argv, prog)``.
"""

from __future__ import annotations

__all__ = ["COMMANDS", "main"]

import argparse
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """Where a subcommand's ``cli(argv, prog)`` lives, and its one-line help."""

    module: str
    help: str
    func: str = "cli"


COMMANDS: dict[str, Command] = {
    "run": Command("simace.cli.run", "Run every replicate of a scenario, then its plots and atlas"),
    "gather": Command("simace.cli.gather", "Summarize a folder's reports and render its validation atlas"),
    "show": Command("simace.cli.inspect", "Print a scenario's resolved parameters, seeds, and paths", "show_cli"),
    "ls": Command("simace.cli.inspect", "List scenarios and the state of each replicate", "ls_cli"),
    "simulate": Command("simace.simulation.simulate", "Simulate a pedigree with A/C/E liabilities"),
    "phenotype": Command("simace.phenotype.runner", "Draw event times for two traits from a pedigree"),
    "censor": Command("simace.censoring.censor", "Apply age-window and competing-death censoring"),
    "ascertain": Command("simace.ascertainment.runner", "Dropout plus case-weighted N_sample selection"),
    "analyze": Command("simace.analysis.analyze", "Write a replicate's report, plot payload, and plot sample"),
    "plot": Command("simace.plotting.plot_phenotype", "Render a scenario's phenotype plots"),
    "atlas": Command("simace.plotting.scenario_atlas", "Assemble a scenario's phenotype atlas"),
    "gather-reports": Command("simace.analysis.gather", "Collect report.yaml files into one summary TSV"),
    "plot-validation": Command("simace.plotting.plot_validation", "Render validation plots from a summary TSV"),
    "validate": Command("simace.analysis.validate.runner", "Debug: run validation checks on a pedigree"),
    "stats": Command("simace.analysis.stats.runner", "Debug: build a stats report for one replicate"),
    "effective-size": Command("simace.analysis.stats.effective_size", "Estimate Ne for one replicate"),
    "plot-effective-size": Command("simace.plotting.plot_effective_size", "Render the Ne atlas"),
    "parquet-to-tsv": Command("simace.core.parquet_to_tsv", "Convert parquet files to TSV"),
}


def _epilog() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {command.help}" for name, command in COMMANDS.items()]
    return "commands:\n" + "\n".join(lines) + "\n\nRun 'simace <command> --help' for a command's flags."


def main(argv: list[str] | None = None) -> None:
    """Dispatch ``simace <command> [args...]`` to the command's module."""
    from simace.core.cli_base import add_version_arg

    parser = argparse.ArgumentParser(
        prog="simace",
        description="Simulate multi-generational pedigrees with A/C/E variance components.",
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_version_arg(parser, "simace")
    parser.add_argument("command", choices=COMMANDS, metavar="command")
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    ns = parser.parse_args(argv)

    command = COMMANDS[ns.command]
    entry = getattr(importlib.import_module(command.module), command.func)
    entry(ns.args, prog=f"simace {ns.command}")
