"""simace - ACE pedigree simulation with A/C/E variance components."""

__all__ = ["setup_logging"]

from importlib.metadata import version

__version__ = version("simace")

import logging


def setup_logging(level=logging.INFO, log_file=None, tag=None):
    """Configure the ``simace`` package logger.

    Call this once from CLI entry points or Snakemake wrappers.
    Subsequent calls are no-ops (handlers already attached).

    Args:
        level: logging level (e.g. logging.DEBUG, logging.INFO).
        log_file: optional path; if given a FileHandler is added.
        tag: optional prefix for console messages (e.g. "beck_adhd/rep1").
    """
    pkg = logging.getLogger("simace")
    pkg.setLevel(level)
    if pkg.handlers:
        return
    prefix = f"[{tag}] " if tag else ""
    fmt_console = logging.Formatter(f"{prefix}%(levelname)s: %(message)s")
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


def _snakemake_tag(wildcards) -> str:
    """Build a console log tag from Snakemake wildcards."""
    parts = []
    scenario = getattr(wildcards, "scenario", None)
    if scenario:
        parts.append(scenario)
    rep = getattr(wildcards, "rep", None)
    if rep:
        parts.append(f"rep{rep}")
    return "/".join(parts) if parts else ""
