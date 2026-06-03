"""Assemble scenario plot atlas - Snakemake wrapper with CLI fallback."""

import logging
from pathlib import Path

from simace import _snakemake_tag, setup_logging
from simace.core.yaml_io import load_yaml
from simace.plotting.atlas_manifest import build_phenotype_atlas
from simace.plotting.plot_atlas import assemble_atlas
from simace.plotting.stats_report import plotting_report_views

logger = logging.getLogger(__name__)


def _load_cli_scenario_params(params_yaml: str | None, scenario: str, config_dir: Path) -> dict | None:
    """Load CLI scenario metadata, enriching old params.yaml files with config defaults when possible."""
    if params_yaml is None and scenario == "unknown":
        return None

    params = load_yaml(params_yaml) if params_yaml else {}
    if params is None:
        params = {}

    merged = dict(params)
    if scenario != "unknown" and config_dir.exists():
        try:
            from simace.config import resolve_defaults, resolve_scenarios

            defaults = resolve_defaults(config_dir)
            scenarios = resolve_scenarios(config_dir, defaults)
            if scenario in scenarios:
                merged = {**defaults, **scenarios[scenario], **params}
        except Exception as exc:  # pragma: no cover - fallback should not fail atlas assembly
            logger.warning("Could not merge CLI atlas metadata from %s: %s", config_dir, exc)

    merged["scenario"] = scenario
    return merged


def _discover_cli_stats(plot_dir: Path) -> tuple[list[Path], list[Path | None]]:
    """Find per-replicate stats beside a CLI plot directory when not passed explicitly."""
    scenario_dir = plot_dir.parent
    report_paths = sorted(scenario_dir.glob("rep*/report.yaml"))
    if report_paths:
        payload_paths = []
        for report_path in report_paths:
            payload_path = report_path.parent / "plot_payload.yaml"
            payload_paths.append(payload_path if payload_path.exists() else None)
        return report_paths, payload_paths

    flat_stats_paths = sorted(scenario_dir.glob("rep*/phenotype_stats.yaml"))
    return flat_stats_paths, [None] * len(flat_stats_paths)


def _is_curated_report(report: dict | None) -> bool:
    """Return true for current v2 report.yaml payloads that need adaptation."""
    return isinstance(report, dict) and any(key in report for key in ("observed", "scopes", "inputs"))


def _stats_for_atlas(reports: list[dict | None], payloads: list[dict | None]) -> list[dict]:
    """Return flat plotting stats for current report.yaml or legacy phenotype_stats.yaml inputs."""
    if not reports:
        return []
    if any(_is_curated_report(report) for report in reports):
        return plotting_report_views(reports, payloads)
    return [report for report in reports if report]


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    p = snakemake.params

    phenotype_paths = [Path(x) for x in snakemake.input.phenotype]
    output_path = Path(snakemake.output[0])
    plot_dir = phenotype_paths[0].parent

    scenario_params = load_yaml(snakemake.input.params_yaml)

    # Merge in config-level parameters not present in params.yaml
    extra_keys = [
        "scenario",
        "replicates",
        "folder",
        "beta1",
        "beta_sex1",
        "phenotype_model1",
        "phenotype_params1",
        "beta2",
        "beta_sex2",
        "phenotype_model2",
        "phenotype_params2",
        "standardize",
        "censor_age",
        "gen_censoring",
        "death_scale",
        "death_rho",
        "G_pheno",
        "N_sample",
        "dropout_rate",
        "case_ascertainment_ratio",
        "max_degree",
        "plot_format",
    ]
    for key in extra_keys:
        val = getattr(p, key, None)
        if val is not None:
            scenario_params[key] = val

    plot_ext = scenario_params.get("plot_format", "png")
    items = build_phenotype_atlas(scenario_params)

    # Build per-replicate plotting views for Table 1 from the curated v2 report
    # plus its plot_payload (dense arrays merged back so Table 1 can derive
    # onset quartiles).
    reports = [load_yaml(p) for p in snakemake.input.report]
    payloads = [load_yaml(p) for p in snakemake.input.plot_payload]
    all_stats = plotting_report_views(reports, payloads)

    assemble_atlas(
        items,
        plot_dir,
        output_path,
        plot_ext=plot_ext,
        scenario_params=scenario_params,
        stats_data=all_stats,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        import argparse

        from simace.core.cli_base import add_logging_args, init_logging

        parser = argparse.ArgumentParser(description="Assemble scenario plot atlas")
        add_logging_args(parser)
        parser.add_argument("--plot-dir", required=True, help="Directory containing the plot PNGs")
        parser.add_argument("--params-yaml", default=None, help="Scenario params.yaml for title page")
        parser.add_argument(
            "--config-dir",
            default="config",
            help="Config directory used to enrich CLI atlas metadata with defaults (default: config)",
        )
        parser.add_argument("--report", nargs="*", default=[], help="report.yaml paths (one per replicate)")
        parser.add_argument("--plot-payload", nargs="*", default=[], help="plot_payload.yaml paths (one per replicate)")
        parser.add_argument("--scenario", default="unknown", help="Scenario name")
        parser.add_argument("--output", required=True, help="Output PDF path")
        parser.add_argument(
            "--html-output",
            default=None,
            help="Optional output HTML atlas path (CLI prototype; Snakemake does not pass this flag)",
        )
        parser.add_argument("--plot-ext", default="png", help="Plot file extension (default: png)")
        args = parser.parse_args()
        init_logging(args)

        scenario_params = _load_cli_scenario_params(args.params_yaml, args.scenario, Path(args.config_dir))

        report_paths = [Path(rp) for rp in args.report]
        payload_paths: list[Path | None] = [Path(pp) for pp in args.plot_payload]
        if not report_paths:
            report_paths, payload_paths = _discover_cli_stats(Path(args.plot_dir))
        elif not payload_paths:
            payload_paths = []
            for report_path in report_paths:
                sibling_payload = report_path.parent / "plot_payload.yaml"
                payload_paths.append(sibling_payload if sibling_payload.exists() else None)

        reports = [load_yaml(rp) for rp in report_paths]
        payloads = [load_yaml(pp) if pp is not None else None for pp in payload_paths]
        all_stats = _stats_for_atlas(reports, payloads)

        items = build_phenotype_atlas(scenario_params)
        assemble_atlas(
            items,
            Path(args.plot_dir),
            Path(args.output),
            plot_ext=args.plot_ext,
            scenario_params=scenario_params,
            stats_data=all_stats or None,
        )
        if args.html_output:
            from simace.plotting.plot_atlas_html import assemble_html_atlas

            assemble_html_atlas(
                items,
                Path(args.plot_dir),
                Path(args.html_output),
                plot_ext=args.plot_ext,
                scenario_params=scenario_params,
                stats_data=all_stats or None,
            )
    else:
        _run_snakemake()
