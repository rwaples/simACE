"""AM-aware expected additive-genetic relative correlations.

Under random mating the additive-genetic (A-component) correlation between
relatives equals the relatedness ``2·kinship`` (full sibs 0.5, half sibs 0.25,
parent-offspring 0.5). Phenotypic assortative mating correlates the mates'
breeding values — the genetic mate correlation ``mu_A`` — which inflates these
relative correlations. This module supplies the AM-aware expectations and the
data-anchored mate-correlation measurement they use.

Notation (per trait): ``mu_A = Corr(A_mother, A_father)`` and ``r_ho =
Corr(liability_mother, liability_father)`` over the unique mating pairs. Derived
from simACE's midparent transmission model (``reproduce`` in
:mod:`simace.simulation.simulate`) and verified against simulation:

* Full sibs / parent-offspring: ``(1 + mu_A) / 2`` — both parents (FS) or the
  one parent (PO) contribute, with the mate cross-term ``mu_A``.
* Half sibs: ``(1 + 2·mu_A + mu_A·r_ho) / 4``. Half sibs share one parent; in
  the multiple-mating model the two *distinct* co-parents are themselves
  genetically correlated through the shared mate (co-mate correlation
  ``mu_A·r_ho``, via the A→P→shared-P→P→A path), which the naive
  ``(1 + 2·mu_A)/4`` omits.
* MZ twins: ``1`` (identical breeding values — unaffected by AM).

All forms reduce to ``2·kinship`` when ``mu_A = 0``. This is single-trait
(direct phenotypic) assortment only — the regime of Herzig et al. (2026,
doi:10.1016/j.tpb.2026.06.003) and the companion variance equilibrium in
:mod:`simace.simulation.am_equilibrium`. Cross-trait / multivariate (bivariate)
AM (Border et al. 2024) is out of scope and signalled via :func:`am_relatedness_mode`
so callers can skip rather than assert a single-trait formula.
"""

from typing import Any

import numpy as np
import pandas as pd

from simace.core.numerics import safe_corrcoef
from simace.core.pedigree_arrays import PedigreeArrays

from ._common import _MIN_PAIRS_FOR_CORR

__all__ = [
    "am_expected_a_correlation",
    "am_relatedness_mode",
    "observed_mate_correlations",
    "resolve_expected_a_corr",
]


def _active(value: Any) -> bool:
    """True if an assortment parameter is non-zero (scalar) or any-gen non-zero (dict)."""
    if isinstance(value, dict):
        return any(value.values())
    return bool(value)


def am_relatedness_mode(params: dict[str, Any], t: int) -> str:
    """Classify trait ``t``'s assortment for relatedness expectations.

    Returns ``"none"`` (no AM → random-mating expectation applies),
    ``"single"`` (single-trait direct AM → AM formula applies), or
    ``"bivariate"`` (both traits assort → cross-trait paths; single-trait
    formula does not apply, caller should skip).
    """
    if params.get("mating_model", "standard") != "standard":
        return "none"
    this = params.get(f"assort{t}", 0.0)
    other = params.get("assort2" if t == 1 else "assort1", 0.0)
    if not _active(this):
        return "none"
    return "bivariate" if _active(other) else "single"


def observed_mate_correlations(df: pd.DataFrame, ped: PedigreeArrays, t: int) -> tuple[float, float, int]:
    """Measure ``(mu_A, r_ho, n_pairs)`` for trait ``t`` over unique mating pairs.

    ``mu_A`` is the genetic (A-component) mate correlation; ``r_ho`` the
    liability (phenotypic) mate correlation. Both are pooled across the recorded
    non-founder generations. Returns ``(0.0, 0.0, n)`` when there are too few
    pairs or a correlation is undefined.
    """
    non_founders = df[df["mother"] != -1]
    pairs = non_founders[["mother", "father"]].drop_duplicates()
    # Restrict to pairs whose parents are both present in the recorded pedigree.
    in_idx = ped.contains(pairs["mother"].to_numpy()) & ped.contains(pairs["father"].to_numpy())
    pairs = pairs[in_idx]
    n_pairs = len(pairs)
    if n_pairs < _MIN_PAIRS_FOR_CORR:
        return 0.0, 0.0, n_pairs

    m = pairs["mother"].to_numpy()
    f = pairs["father"].to_numpy()
    a_m = ped.gather(f"A{t}", m)
    a_f = ped.gather(f"A{t}", f)
    l_m = a_m + ped.gather(f"C{t}", m) + ped.gather(f"E{t}", m)
    l_f = a_f + ped.gather(f"C{t}", f) + ped.gather(f"E{t}", f)

    mu_a = safe_corrcoef(a_m, a_f)
    r_ho = safe_corrcoef(l_m, l_f)
    mu_a = 0.0 if np.isnan(mu_a) else float(mu_a)
    r_ho = 0.0 if np.isnan(r_ho) else float(r_ho)
    return mu_a, r_ho, n_pairs


def am_expected_a_correlation(kind: str, mu_a: float, r_ho: float) -> float:
    """AM-aware expected A-component correlation for a relationship ``kind``.

    Args:
        kind: One of ``"MZ"``, ``"FS"``, ``"PO"``, ``"HS"``.
        mu_a: Genetic mate correlation ``Corr(A_mother, A_father)``.
        r_ho: Liability mate correlation (used only by ``"HS"``).

    Returns:
        Expected correlation; reduces to ``2·kinship`` at ``mu_a = 0``.
    """
    if kind in ("FS", "PO"):
        return (1.0 + mu_a) / 2.0
    if kind == "HS":
        return (1.0 + 2.0 * mu_a + mu_a * r_ho) / 4.0
    if kind == "MZ":
        return 1.0
    raise ValueError(f"unknown relationship kind: {kind!r}")


def resolve_expected_a_corr(
    df: pd.DataFrame,
    ped: PedigreeArrays,
    params: dict[str, Any],
    t: int,
    kind: str,
    default: float,
) -> tuple[float | None, str | None, dict[str, Any]]:
    """Resolve the expected A-correlation for trait ``t``, AM-aware.

    Returns ``(expected, skip_reason, info)``:

    * no AM → ``(default, None, {})`` (random-mating ``2·kinship`` unchanged);
    * single-trait AM → ``(am_expected, None, {mu_A, r_ho, mate_pairs})``;
    * bivariate AM → ``(None, reason, {})`` so the caller can skip the scored
      check (cross-trait paths are not modelled by the single-trait formula).
    """
    mode = am_relatedness_mode(params, t)
    if mode == "none":
        return default, None, {}
    if mode == "bivariate":
        return None, "both-trait AM active (cross-trait paths not in single-trait formula)", {}
    mu_a, r_ho, n = observed_mate_correlations(df, ped, t)
    expected = am_expected_a_correlation(kind, mu_a, r_ho)
    return expected, None, {"mu_A": mu_a, "r_ho": r_ho, "mate_pairs": n}
