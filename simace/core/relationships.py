"""Relationship-pair and sex vocabulary used across stats and plotting."""

__all__ = ["PAIR_TYPES", "SEX_LEVELS"]

# Canonical 7-element subset of REL_REGISTRY (defined in
# ``simace.core.pedigree_graph``) used for tetrachoric / liability correlation
# analyses.
PAIR_TYPES: list[str] = [
    "MZ",
    "FS",
    "MO",
    "FO",
    "MHS",
    "PHS",
    "1C",
]

# Encoding of the binary ``sex`` column used throughout the pipeline.
SEX_LEVELS: list[tuple[int, str]] = [(0, "female"), (1, "male")]
