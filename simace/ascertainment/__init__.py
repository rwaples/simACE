"""Unified ascertainment stage: random dropout + case-weighted N_sample draw.

Replaces the legacy two-stage design (pre-phenotype pedigree dropout +
post-censor subsampling). Per ADR 0001: ascertainment writes the canonical
post-stage ``pedigree.parquet`` and ``trait.parquet`` outputs that both
simACE-stats and fitACE consume.

The implementation and CLI live in :mod:`simace.ascertainment.runner`; the
names below are re-exported for the public API and the ``simace-ascertain``
entry point.
"""

from .runner import (
    _sever_dangling_links as _sever_dangling_links,  # re-export: tests import it
)
from .runner import (
    cli,
    copy_passthrough_if_possible,
    run_ascertainment,
)

__all__ = ["cli", "copy_passthrough_if_possible", "run_ascertainment"]
