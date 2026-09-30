"""Unified ascertainment stage: random dropout + case-weighted N_sample draw.

Replaces the legacy two-stage design (pre-phenotype pedigree dropout +
post-censor subsampling), per ADR 0001. Under ``simace run`` the ``cohort``
stage calls :func:`run_ascertainment` in process and stores its two outputs as
one ``cohort.parquet`` (ADR 0021); the standalone ``simace ascertain`` command
writes them as an explicit pedigree and trait file pair.

The implementation and CLI live in :mod:`simace.ascertainment.runner`; the
names below are re-exported for the public API and the ``simace ascertain``
command.
"""

from .runner import (
    cli,
    copy_passthrough_if_possible,
    run_ascertainment,
)

__all__ = ["cli", "copy_passthrough_if_possible", "run_ascertainment"]
