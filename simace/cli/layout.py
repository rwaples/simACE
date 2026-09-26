"""The results-directory convention that ``simace run`` imposes.

Every path under ``results/`` and ``logs/`` that the pipeline reads or writes
is built here. Stage subcommands never see this module: they take explicit
paths, and ``simace run`` fills those paths from a :class:`Layout`.

``pedigree.parquet``, ``trait.parquet``, ``report.yaml`` and ``params.yaml``
under ``results/{folder}/{scenario}/rep{rep}/`` are read by fitACE and are a
frozen contract.
"""

from __future__ import annotations

__all__ = ["Layout", "RepArtifact"]

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class RepArtifact(StrEnum):
    """Basenames of the files in one replicate directory."""

    PEDIGREE_FULL = "pedigree.full.parquet"
    PEDIGREE_FULL_TSTRAIT = "pedigree.full.tstrait.parquet"
    PARAMS = "params.yaml"
    TRAIT_RAW = "trait.raw.parquet"
    TRAIT_FULL = "trait.full.parquet"
    PEDIGREE = "pedigree.parquet"
    TRAIT = "trait.parquet"
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
