"""Assemble a scenario's phenotype atlas from its plots and per-rep reports."""

from __future__ import annotations

__all__ = ["cli", "scenario_atlas"]

import argparse
from pathlib import Path
from typing import Any

from simace.core.yaml_io import load_yaml


def scenario_atlas(
    *,
    plot_dir: Path,
    params_path: Path,
    meta: dict[str, Any],
    report_paths: list[Path],
    plot_payload_paths: list[Path],
    output: Path,
) -> None:
    """Render ``output`` (``.html`` or ``.pdf``) for one scenario.

    Args:
        plot_dir: directory holding the scenario's phenotype plots.
        params_path: rep 1 ``params.yaml``; its keys head the title page.
        meta: scenario config keys absent from ``params.yaml`` (phenotype,
            censoring, ascertainment, ``scenario``, ``plot_format``). Non-null
            values override ``params.yaml``.
        report_paths: one ``report.yaml`` per replicate.
        plot_payload_paths: one ``plot_payload.yaml`` per replicate.
        output: atlas path; the extension selects the renderer.
    """
    from simace.plotting.atlas_manifest import build_phenotype_atlas
    from simace.plotting.render_atlas import render_atlas
    from simace.plotting.stats_report import plotting_report_views

    scenario_params = load_yaml(params_path)
    scenario_params.update({key: val for key, val in meta.items() if val is not None})

    reports = [load_yaml(p) for p in report_paths]
    payloads = [load_yaml(p) for p in plot_payload_paths]
    render_atlas(
        build_phenotype_atlas(scenario_params),
        plot_dir,
        output,
        plot_ext=scenario_params.get("plot_format", "png"),
        scenario_params=scenario_params,
        stats_data=plotting_report_views(reports, payloads),
    )


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line entry point for the scenario atlas."""
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging, yaml_mapping

    parser = argparse.ArgumentParser(prog=prog, description="Assemble a scenario's phenotype atlas")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--plot-dir", required=True, type=Path, help="Directory holding the phenotype plots")
    parser.add_argument("--params", required=True, type=Path, help="Rep 1 params.yaml")
    parser.add_argument(
        "--meta",
        type=yaml_mapping,
        default={},
        help="YAML flow mapping of scenario keys merged over params.yaml (non-null values win)",
    )
    parser.add_argument("--report", nargs="+", required=True, type=Path, help="report.yaml paths, one per rep")
    parser.add_argument(
        "--plot-payload", nargs="+", required=True, type=Path, help="plot_payload.yaml paths, one per rep"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Atlas path; the extension (.html or .pdf) selects the renderer"
    )
    args = parser.parse_args(argv)
    init_logging(args)

    scenario_atlas(
        plot_dir=args.plot_dir,
        params_path=args.params,
        meta=args.meta,
        report_paths=args.report,
        plot_payload_paths=args.plot_payload,
        output=args.output,
    )
