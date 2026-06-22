"""Prevalence resolution helpers shared by the threshold-based phenotype models.

Prevalence is encoded in the type system: only the threshold-based phenotype
models (``adult``, ``cure_frailty``, ``simple_ltm``) accept a prevalence
parameter. Frailty and first_passage do not — case fraction emerges from the
hazard for those families.

Prevalence may be:
  * a scalar float (uniform across the population);
  * a per-generation dict (int generation key → float prevalence);
  * a sex-specific dict (``{"female": ..., "male": ...}``), where each side
    may itself be a scalar or a per-generation dict.
"""

import numpy as np
from scipy.special import ndtri

from simace.phenotype.hazards import StandardizeMode, standardize_liability

__all__ = [
    "case_status_from_liability",
    "liability_threshold_mask",
    "prevalence_to_array",
    "resolve_prevalence",
]


def _validate_prevalence(value) -> None:
    """Raise unless every case fraction ``K`` in ``value`` lies in the open (0, 1).

    Accepts a scalar or an already-expanded per-individual array.  A ``K`` outside
    ``(0, 1)`` would otherwise pass silently into ``ndtri(1 - K)`` (see
    :func:`liability_threshold_mask`), yielding a ``±inf``/``nan`` threshold — no
    cases or all cases — with no error.
    """
    arr = np.asarray(value, dtype=float)
    in_range = (arr > 0.0) & (arr < 1.0)
    if not np.all(in_range):
        bad = np.unique(arr[~in_range]).tolist()
        raise ValueError(f"prevalence must be in the open interval (0, 1), got out-of-range value(s) {bad}")


def prevalence_to_array(prev, generation: np.ndarray) -> float | np.ndarray:
    """Expand a scalar or per-generation dict to a per-individual array.

    A general expander: returns ``prev`` unchanged if it is not a dict, otherwise
    broadcasts a per-generation dict over ``generation``.  It does **not**
    range-check values — it is reused for non-prevalence weights too (e.g. the
    blend weight ``alpha`` in ``blended_post``, where 0 and 1 are valid).
    Prevalence range-checking is :func:`resolve_prevalence`'s job.

    Raises:
        ValueError: per-generation dict missing a generation present in ``generation``.
    """
    if isinstance(prev, dict):
        arr = np.empty(len(generation))
        for gen in np.unique(generation):
            mask = generation == gen
            gen_key = int(gen)
            if gen_key not in prev:
                raise ValueError(f"prevalence dict missing generation {gen_key}; dict has keys {sorted(prev.keys())}")
            arr[mask] = prev[gen_key]
        return arr
    return prev


def resolve_prevalence(
    prev,
    sex: np.ndarray | None,
    generation: np.ndarray,
) -> float | np.ndarray:
    """Resolve prevalence to a scalar or per-individual array, validated to (0, 1).

    ``sex`` is required only when ``prev`` is a sex-specific dict; pass any
    array of matching length otherwise.  The resolved case fraction is validated
    to lie in the open interval ``(0, 1)`` — the single home for that check, so
    all three threshold models (adult ``ltm`` + ``cox``, ``cure_frailty``,
    ``simple_ltm``) inherit it regardless of their downstream threshold/rank step.

    Raises:
        ValueError: a resolved prevalence value is outside ``(0, 1)``.
    """
    if isinstance(prev, dict) and "female" in prev and "male" in prev:
        f_prev = prevalence_to_array(prev["female"], generation)
        m_prev = prevalence_to_array(prev["male"], generation)
        resolved = np.where(sex == 1, m_prev, f_prev)
    else:
        resolved = prevalence_to_array(prev, generation)
    _validate_prevalence(resolved)
    return resolved


def liability_threshold_mask(L: np.ndarray, prevalence) -> np.ndarray:
    """Case mask from already-standardized liability ``L`` and resolved prevalence.

    Cases are individuals above the probit threshold ``ndtri(1 - K)``.  This is
    the single home for the threshold convention shared by ``adult.ltm``,
    ``cure_frailty``, and ``simple_ltm`` — changing the ``ndtri(1 - K)`` mapping
    or the tie-breaking touches exactly one place.

    The comparison is strict (``threshold < L``): on continuous liability the tie
    set is measure-zero, so the realised case fraction matches ``K`` and the
    ``<`` / ``<=`` choice is immaterial in practice.

    Args:
        L: per-individual *standardized* liability.
        prevalence: resolved case fraction ``K`` — scalar or per-individual array
            (already passed through :func:`resolve_prevalence`).

    Returns:
        Boolean array, ``True`` for cases.
    """
    return ndtri(1.0 - np.asarray(prevalence)) < L


def case_status_from_liability(
    liability: np.ndarray,
    prevalence,
    sex: np.ndarray | None,
    generation: np.ndarray,
    mode: StandardizeMode,
) -> np.ndarray:
    """Probit liability-threshold case status from *raw* liability.

    Standardizes ``liability`` under ``mode``, resolves ``prevalence``, then
    applies :func:`liability_threshold_mask`.  Convenience bundle for callers
    that need only case status from raw inputs: ``cure_frailty`` and
    ``simple_ltm``.  ``adult.ltm`` instead calls :func:`liability_threshold_mask`
    directly, because it reuses the standardized liability and the resolved
    prevalence array for its onset CIF (and resolves prevalence once at the
    ``simulate`` level to share with its ``cox`` sub-method).

    Args:
        liability: per-individual liability values.
        prevalence: case fraction ``K`` — scalar, per-generation dict, or
            ``{"female": ..., "male": ...}`` dict (resolved via
            :func:`resolve_prevalence`).
        sex: per-individual sex codes; required only for sex-specific prevalence.
        generation: per-individual generation labels (for standardization and
            per-generation prevalence).
        mode: liability standardization mode (``"none" | "global" | "per_generation"``).

    Returns:
        Boolean array, ``True`` for cases.
    """
    L = standardize_liability(liability, mode, generation)
    prev = resolve_prevalence(prevalence, sex, generation)
    return liability_threshold_mask(L, prev)
