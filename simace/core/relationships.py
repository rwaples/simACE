"""Relationship-pair and sex vocabulary used across stats and plotting.

Canonical home for relationship-type *properties*: the C-sharing rule and
expected liability correlations. Kinship is sourced from
``pedigree_graph.PAIR_KINSHIP`` (never re-declared here). Pooling and
presentation groupings deliberately live at the call sites, not here.
See ``docs/adr/0009-relationship-semantics-home.md``.
"""

from pedigree_graph import PAIR_KINSHIP

__all__ = [
    "RELATIONSHIP_TYPES",
    "SEX_LEVELS",
    "expected_liability_corr",
    "shared_environment_coefficient",
]

# Canonical 7-element subset of REL_REGISTRY (defined in
# ``pedigree_graph``) used for tetrachoric / liability correlation
# analyses.
RELATIONSHIP_TYPES: list[str] = [
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

# Whether a relationship type shares a household — and therefore the common
# environment C. Households are assigned by mother (CONTEXT.md), so same-mother
# types (MZ, FS, MHS) share C; paternal half-sibs, parent-offspring, and cousin
# pairs do not. Keys mirror ``RELATIONSHIP_TYPES``.
_SHARED_C: dict[str, float] = {
    "MZ": 1.0,
    "FS": 1.0,
    "MHS": 1.0,
    "PHS": 0.0,
    "MO": 0.0,
    "FO": 0.0,
    "1C": 0.0,
}


def shared_environment_coefficient(relationship_type: str) -> float:
    """Return the C-sharing coefficient for ``relationship_type``.

    ``1.0`` if the pair shares a household (and thus the common-environment
    component C), else ``0.0``.

    Raises:
        ValueError: if ``relationship_type`` is not one of
            :data:`RELATIONSHIP_TYPES`.
    """
    try:
        return _SHARED_C[relationship_type]
    except KeyError:
        raise ValueError(
            f"unknown relationship type {relationship_type!r}; expected one of {RELATIONSHIP_TYPES}"
        ) from None


def expected_liability_corr(relationship_type: str, A: float, C: float) -> float:
    """Expected liability correlation ``2 * kinship * A + C_shared * C``.

    Kinship is read from ``pedigree_graph.PAIR_KINSHIP`` (never a literal);
    ``C_shared`` comes from :func:`shared_environment_coefficient`. ``A`` and
    ``C`` are the additive-genetic and common-environment variances.

    Raises:
        ValueError: if ``relationship_type`` is not one of
            :data:`RELATIONSHIP_TYPES`.
    """
    c_coef = shared_environment_coefficient(relationship_type)  # validates the type
    return 2.0 * PAIR_KINSHIP[relationship_type] * A + c_coef * C
