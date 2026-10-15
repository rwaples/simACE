"""Relationship-pair and sex vocabulary used across stats and plotting.

Canonical home for relationship-type *properties*: the C-sharing rule,
expected liability correlations, and the reference line plots draw (analytic
where exact, else the recorded-pedigree truth; ADR 0025). Kinship is sourced from
``pedigree_graph.RELATIONSHIPS[code].nominal_kinship`` (never re-declared
here). Pooling and presentation groupings deliberately live at the call
sites, not here. See ``docs/adr/0009-relationship-semantics-home.md``.
``pedigree_graph`` is imported only where kinship is read, so importing the
constants here (as ``simace run`` does) does not load it.
"""

__all__ = [
    "C_TRANSMISSION_KEYS",
    "DEFAULT_MAX_DEGREE",
    "RELATIONSHIP_TYPES",
    "SEX_LEVELS",
    "analytic_applies",
    "expected_liability_corr",
    "reference_corr",
    "reference_midparent_slope",
    "shared_environment_coefficient",
    "transmission_active",
    "transmission_coefficients",
    "transmission_key",
]

from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# Canonical 7-element subset of ``pedigree_graph.RELATIONSHIPS`` used for
# tetrachoric / liability correlation analyses.
RELATIONSHIP_TYPES: list[str] = [
    "MZ",
    "FS",
    "MO",
    "FO",
    "MHS",
    "PHS",
    "1C",
]

# Default relationship-extraction depth for the analysis and plotting paths.
# Follows ``pedigree_graph``'s registry cutoff exactly: degree 3 is the depth
# that first includes 1st cousins (``1C`` above); degree 2 stops at half-sibs,
# grandparents, and avuncular pairs. Single source of truth for the many
# ``relationship_pairs(max_degree=...)`` call sites — keep ``config/_default.yaml``
# (``analysis.max_degree``) in sync with this value.
DEFAULT_MAX_DEGREE: int = 3

# Encoding of the binary ``sex`` column used throughout the pipeline.
SEX_LEVELS: list[tuple[int, str]] = [(0, "female"), (1, "male")]

# The fraction of the common environment C a relationship type shares under
# random mating with constant C, as a function of the C transmission
# coefficients (ADR 0025). Households are assigned by mother (CONTEXT.md), so
# without transmission same-mother types (MZ, FS, MHS) share all of C and every
# other type none. With transmission, parent-offspring pairs share b_m or b_f,
# maternal half-sibs with different fathers lose b_f**2, and paternal half-sibs
# gain it. Lineage-mixed types (1C) have no single value. Keys mirror
# ``RELATIONSHIP_TYPES``.
_SHARED_C: dict[str, Callable[[float, float], float | None]] = {
    "MZ": lambda b_m, b_f: 1.0,
    "FS": lambda b_m, b_f: 1.0,
    "MHS": lambda b_m, b_f: 1.0 - b_f**2,
    "PHS": lambda b_m, b_f: b_f**2,
    "MO": lambda b_m, b_f: b_m,
    "FO": lambda b_m, b_f: b_f,
    "1C": lambda b_m, b_f: 0.0 if b_m == b_f == 0.0 else None,
}


def shared_environment_coefficient(relationship_type: str, b_m: float = 0.0, b_f: float = 0.0) -> float | None:
    """Return the fraction of C that ``relationship_type`` pairs share.

    ``b_m`` and ``b_f`` are the trait's maternal and paternal C transmission
    coefficients. With both 0 the value is ``1.0`` for householdmates (MZ,
    FS, MHS) and ``0.0`` otherwise. Otherwise it is the random-mating,
    constant-C value, or None for a lineage-mixed type, which has none.

    Raises:
        ValueError: if ``relationship_type`` is not one of
            :data:`RELATIONSHIP_TYPES`.
    """
    try:
        rule = _SHARED_C[relationship_type]
    except KeyError:
        raise ValueError(
            f"unknown relationship type {relationship_type!r}; expected one of {RELATIONSHIP_TYPES}"
        ) from None
    return rule(b_m, b_f)


def expected_liability_corr(
    relationship_type: str, A: float, C: float, b_m: float = 0.0, b_f: float = 0.0, V_P: float = 1.0
) -> float | None:
    """Expected liability correlation ``(2 * kinship * A + C_shared * C) / V_P``.

    Kinship is read from ``pedigree_graph.RELATIONSHIPS`` (never a literal);
    ``C_shared`` comes from :func:`shared_environment_coefficient`, and the
    result is None where that is, and where ``V_P`` is not positive (a
    zero-variance trait has no correlation). ``A`` and ``C`` are the
    additive-genetic and common-environment variances and ``V_P`` the
    liability variance.

    Raises:
        ValueError: if ``relationship_type`` is not one of
            :data:`RELATIONSHIP_TYPES`.
    """
    from pedigree_graph import RELATIONSHIPS

    c_coef = shared_environment_coefficient(relationship_type, b_m, b_f)  # validates the type
    if c_coef is None or V_P <= 0.0:
        return None
    return (2.0 * RELATIONSHIPS[relationship_type].nominal_kinship * A + c_coef * C) / V_P


def transmission_key(side: str, trait: int) -> str:
    """The flat config key of a C transmission coefficient: ``Ctm_maternal1`` and so on (ADR 0025)."""
    return f"Ctm_{side}{trait}"


#: Every C transmission coefficient key, in stage-flag order.
C_TRANSMISSION_KEYS: tuple[str, ...] = tuple(
    transmission_key(side, trait) for trait in (1, 2) for side in ("maternal", "paternal")
)


def transmission_coefficients(params: Mapping[str, Any], trait: int) -> tuple[float, float]:
    """Return trait ``trait``'s ``(b_m, b_f)``; absent keys (pre-ADR-0025 params) are 0."""
    return (
        float(params.get(transmission_key("maternal", trait), 0.0)),
        float(params.get(transmission_key("paternal", trait), 0.0)),
    )


def transmission_active(params: Mapping[str, Any]) -> bool:
    """Whether any trait has a nonzero C transmission coefficient."""
    return any(transmission_coefficients(params, t) != (0.0, 0.0) for t in (1, 2))


def _is_zero(value: Any) -> bool:
    """Whether a mating setting is a no-op; an unparsed value (a gathered TSV's string) counts as active."""
    if value is None:
        return True
    if isinstance(value, str):
        return False
    if isinstance(value, dict):
        return all(float(v) == 0.0 for v in value.values())
    if isinstance(value, list | tuple):
        return all(_is_zero(v) for v in value)
    return float(value) == 0.0


def analytic_applies(relationship_type: str, params: Mapping[str, Any], trait: int) -> bool:
    """Whether the analytic expected correlation is exact for ``relationship_type`` on ``trait``.

    Always with no C transmission on the trait (the pre-ADR-0025 line).
    Otherwise only under random mating with scalar ``C{trait}`` and
    ``E{trait}`` (so parent and offspring share one ``V_P``) and for a
    single-path type, which has a shared-C fraction.
    """
    b_m, b_f = transmission_coefficients(params, trait)
    if b_m == b_f == 0.0:
        return True
    return _closed_form_holds(params, trait) and shared_environment_coefficient(relationship_type, b_m, b_f) is not None


def _closed_form_holds(params: Mapping[str, Any], trait: int) -> bool:
    """Random mating with scalar ``C{trait}`` and ``E{trait}``, where the transmission formulas are exact.

    Wright-Fisher mating is random whatever assortment keys it inherits (they are ignored at runtime).
    """
    random_mating = params.get("mating_model", "standard") != "standard" or all(
        _is_zero(params.get(key)) for key in ("assort1", "assort2", "assort_matrix")
    )
    return random_mating and all(isinstance(params.get(f"{k}{trait}"), Real) for k in ("C", "E"))


def _liability_variance(params: Mapping[str, Any], trait: int) -> float:
    """``V_P = A + C + E`` for the trait; 1 when C or E is per-generation, which has no single ``V_P``.

    The gate admits per-generation C or E only without C transmission, where the
    line stays the undivided ``2 * kinship * A + C_shared * C`` it always was.
    """
    if not all(isinstance(params.get(f"{k}{trait}"), Real) for k in ("C", "E")):
        return 1.0
    return float(params[f"A{trait}"]) + float(params[f"C{trait}"]) + float(params[f"E{trait}"])


def reference_corr(
    relationship_type: str,
    params: Mapping[str, Any],
    trait: int,
    truth: Mapping[str, Mapping[str, Any]] | None = None,
) -> float | None:
    """The liability-correlation reference line for ``relationship_type`` on ``trait``.

    The analytic ``(2 * kinship * A + C_shared * C) / V_P`` when
    :func:`analytic_applies`, else the recorded-pedigree truth
    ``truth[relationship_type]["r"]``, else None. None also where ``A`` or
    ``C`` is absent or per-generation (no single value) or the trait has no
    liability variance.
    """
    if analytic_applies(relationship_type, params, trait):
        A, C = params.get(f"A{trait}"), params.get(f"C{trait}")
        if not (isinstance(A, Real) and isinstance(C, Real)):
            return None
        b_m, b_f = transmission_coefficients(params, trait)
        return expected_liability_corr(
            relationship_type, float(A), float(C), b_m, b_f, _liability_variance(params, trait)
        )
    entry = (truth or {}).get(relationship_type)
    return None if entry is None or entry.get("r") is None else float(entry["r"])


def reference_midparent_slope(params: Mapping[str, Any], trait: int, truth: float | None = None) -> float | None:
    """The midparent-offspring liability slope reference line for ``trait``.

    ``A / V_P`` without C transmission and ``(A + (b_m + b_f) * C) / V_P``
    under the closed-form gate; otherwise the recorded-pedigree truth slope,
    else None. None where ``A`` is absent and, on the formula paths, for a
    trait with no liability variance (whose recorded slope is None too).
    """
    A = params.get(f"A{trait}")
    if not isinstance(A, Real):
        return None
    b_m, b_f = transmission_coefficients(params, trait)
    if b_m == b_f == 0.0:
        V_P = _liability_variance(params, trait)
        return float(A) / V_P if V_P > 0.0 else None
    if _closed_form_holds(params, trait):
        V_P = _liability_variance(params, trait)
        return (float(A) + (b_m + b_f) * float(params[f"C{trait}"])) / V_P if V_P > 0.0 else None
    return truth
