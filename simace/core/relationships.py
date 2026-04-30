"""Relationship-pair vocabulary used across stats and plotting.

Currently exposes ``PAIR_TYPES`` — the canonical 7-element subset of
``REL_REGISTRY`` (defined in ``simace.core.pedigree_graph``) used for
tetrachoric / liability correlation analyses. Future home for additional
relationship-pair constants when ``pedigree_graph.py`` is restructured.
"""

__all__ = ["PAIR_TYPES"]

# Subset of REL_REGISTRY codes used for correlation analyses.
PAIR_TYPES: list[str] = [
    "MZ",
    "FS",
    "MO",
    "FO",
    "MHS",
    "PHS",
    "1C",
]
