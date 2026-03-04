"""sim_ace - ACE pedigree simulation with A/C/E variance components."""

__version__ = "0.1.0"

import logging


def setup_logging(level=logging.INFO, log_file=None):
    """Configure the ``sim_ace`` package logger.

    Call this once from CLI entry points or Snakemake wrappers.
    Subsequent calls are no-ops (handlers already attached).

    Args:
        level: logging level (e.g. logging.DEBUG, logging.INFO).
        log_file: optional path; if given a FileHandler is added.
    """
    pkg = logging.getLogger("sim_ace")
    pkg.setLevel(level)
    if pkg.handlers:
        return
    fmt_console = logging.Formatter("%(levelname)s: %(message)s")
    fmt_file = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt_console)
    pkg.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt_file)
        pkg.addHandler(fh)

from sim_ace.simulate import (
    run_simulation,
    generate_correlated_components,
    mating,
    reproduce,
    add_to_pedigree,
)
from sim_ace.phenotype import simulate_phenotype
from sim_ace.censor import age_censor, death_censor
from sim_ace.threshold import apply_threshold
from sim_ace.validate import run_validation
from sim_ace.stats import tetrachoric_corr_se
from sim_ace.pedigree_graph import extract_relationship_pairs
from sim_ace.sample import run_sample


def __getattr__(name: str):
    """Lazy import for heavy optional modules."""
    if name == "pairwise_weibull_corr_se":
        from sim_ace.survival_corr import pairwise_weibull_corr_se
        return pairwise_weibull_corr_se
    raise AttributeError(f"module 'sim_ace' has no attribute {name!r}")
