"""The results-directory convention that ``simace run`` imposes.

Every path under ``results/`` and ``logs/`` that the pipeline reads or writes
is built here. Stage subcommands never see this module: they take explicit
paths, and ``simace run`` fills those paths from a :class:`Layout`.

``pedigree.parquet`` (the recorded pedigree), ``cohort.parquet``,
``report.yaml`` and ``params.yaml`` under
``results/{folder}/{scenario}/rep{rep}/`` are read by fitACE and are a frozen
contract (results layout 2, ADR 0021).
"""

from __future__ import annotations

__all__ = ["Layout", "RepArtifact", "add_root_args", "project_root", "require_config", "resolve_roots"]

import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

_ROOT_MARKER = Path("config") / "_default.yaml"


def _is_simace_root(candidate: Path) -> bool:
    """A simACE checkout: its ``config/_default.yaml`` beside a ``pyproject.toml`` naming ``simace``.

    ``config/_default.yaml`` alone also matches the fitACE checkout nested
    inside simACE, whose scenarios simace cannot resolve.
    """
    if not (candidate / _ROOT_MARKER).is_file():
        return False
    try:
        return 'name = "simace"' in (candidate / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return False


def project_root(start: Path | None = None) -> Path | None:
    """Return the nearest ancestor of ``start`` (default: cwd) that is a simACE checkout, or None."""
    here = (start or Path.cwd()).resolve()
    return next((c for c in (here, *here.parents) if _is_simace_root(c)), None)


def add_root_args(parser: argparse.ArgumentParser, *, logs: bool = False) -> None:
    """Add ``--config-dir``, ``--results`` and (optionally) ``--logs``, defaulting to the project root's."""
    tail = " (default: found from the current directory)"
    parser.add_argument("--config-dir", type=Path, default=None, help="Config directory" + tail)
    parser.add_argument("--results", type=Path, default=None, help="Results root" + tail)
    if logs:
        parser.add_argument("--logs", type=Path, default=None, help="Log root" + tail)


def resolve_roots(args: argparse.Namespace) -> tuple[Path, Layout]:
    """Return the config directory and :class:`Layout` from the root flags.

    A flag left unset resolves under the project root found by walking up
    from the current directory, or under the current directory when no
    ``config/_default.yaml`` is found above it. Paths stay relative when the
    current directory is the root, so commands run from it print the same
    paths as before.
    """
    root = project_root()
    base = Path() if root in (None, Path.cwd().resolve()) else root
    config_dir = args.config_dir if args.config_dir is not None else base / "config"
    results = args.results if args.results is not None else base / "results"
    logs = getattr(args, "logs", None)
    return config_dir, Layout(root=results, logs=logs if logs is not None else base / "logs")


def require_config(config_dir: Path, command: str) -> None:
    """Exit 2 with a hint when ``config_dir`` has no ``_default.yaml``, as outside a checkout."""
    if (config_dir / _ROOT_MARKER.name).is_file():
        return
    print(
        f"simace {command}: no scenario config at {config_dir / _ROOT_MARKER.name}. Run it inside a simACE "
        "checkout (git clone https://github.com/rwaples/simACE) or pass --config-dir. The stage subcommands "
        "(simace simulate, cohort, analyze, ...) take explicit paths and need no checkout.",
        file=sys.stderr,
    )
    raise SystemExit(2)


class RepArtifact(StrEnum):
    """Basenames of the files in one replicate directory."""

    PEDIGREE = "pedigree.parquet"
    PEDIGREE_FULL_TSTRAIT = "pedigree.full.tstrait.parquet"
    PARAMS = "params.yaml"
    COHORT = "cohort.parquet"
    PHENOTYPED_POPULATION = "phenotyped_population.yaml"
    REPORT = "report.yaml"
    PLOT_PAYLOAD = "plot_payload.yaml"
    PLOTTING_SAMPLE = "plotting_sample.parquet"
    EFFECTIVE_SIZE = "effective_size.yaml"
    RUN_MANIFEST = "run.yaml"
    TIMING = "timing.tsv"


@dataclass(frozen=True)
class Layout:
    """Paths for scenarios, replicates, and folders under a results root."""

    root: Path = Path("results")
    logs: Path = Path("logs")

    def scenario_dir(self, folder: str, scenario: str) -> Path:
        """Return ``{root}/{folder}/{scenario}``."""
        return self.root / folder / scenario

    def rep_dir(self, folder: str, scenario: str, rep: int) -> Path:
        """Return ``{root}/{folder}/{scenario}/rep{rep}``."""
        return self.scenario_dir(folder, scenario) / f"rep{rep}"

    def rep(self, folder: str, scenario: str, rep: int, artifact: RepArtifact) -> Path:
        """Return the path of one replicate artifact."""
        return self.rep_dir(folder, scenario, rep) / artifact

    def scenario_lock(self, folder: str, scenario: str) -> Path:
        """Return ``{root}/{folder}/{scenario}/.run.lock``, held by a running ``simace run``."""
        return self.scenario_dir(folder, scenario) / ".run.lock"

    def scenario_plots(self, folder: str, scenario: str) -> Path:
        """Return the directory holding a scenario's phenotype plots, atlas, and plot timing."""
        return self.scenario_dir(folder, scenario) / "plots"

    def scenario_plots_manifest(self, folder: str, scenario: str) -> Path:
        """Return ``plots/plots.yaml``: which reps the scenario's plots and atlas were built from."""
        return self.scenario_plots(folder, scenario) / "plots.yaml"

    def folder_summary(self, folder: str) -> Path:
        """Return the folder-level ``report_summary.tsv`` path."""
        return self.root / folder / "report_summary.tsv"

    def folder_plots(self, folder: str) -> Path:
        """Return the directory holding a folder's validation plots and atlas."""
        return self.root / folder / "plots"

    def log(self, folder: str, scenario: str, rep: int, stage: str) -> Path:
        """Return the log path for one stage of one replicate."""
        return self.logs / folder / scenario / f"rep{rep}" / f"{stage}.log"

    def scenario_log(self, folder: str, scenario: str, stage: str) -> Path:
        """Return the log path for a scenario-level stage (plot, atlas)."""
        return self.logs / folder / scenario / f"{stage}.log"

    def folder_log(self, folder: str, stage: str) -> Path:
        """Return the log path for a folder-level stage (gather, plot-validation)."""
        return self.logs / folder / f"{stage}.log"
