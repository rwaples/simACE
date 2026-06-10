"""Prevalence resolution helpers shared by AdultModel and CureFrailtyModel.

Prevalence is encoded in the type system: only the threshold-based phenotype
models (``adult``, ``cure_frailty``) accept a prevalence parameter. Frailty
and first_passage do not — case fraction emerges from the hazard for those
families.

Prevalence may be:
  * a scalar float (uniform across the population);
  * a per-generation dict (int generation key → float prevalence);
  * a sex-specific dict (``{"female": ..., "male": ...}``), where each side
    may itself be a scalar or a per-generation dict.
"""

import numpy as np
from scipy.special import ndtri

from simace.phenotype.hazards import StandardizeMode, standardize_liability

__all__ = ["case_status_from_liability", "prevalence_to_array", "resolve_prevalence"]


def prevalence_to_array(prev, generation: np.ndarray) -> float | np.ndarray:
    """Expand a scalar or per-generation dict prevalence to a per-individual array.

    Returns ``prev`` unchanged if it is not a dict.

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
    sex: np.ndarray,
    generation: np.ndarray,
) -> float | np.ndarray:
    """Resolve prevalence to a scalar or per-individual array.

    ``sex`` is required only when ``prev`` is a sex-specific dict; pass any
    array of matching length otherwise.
    """
    if isinstance(prev, dict) and "female" in prev and "male" in prev:
        f_prev = prevalence_to_array(prev["female"], generation)
        m_prev = prevalence_to_array(prev["male"], generation)
        return np.where(sex == 1, m_prev, f_prev)
    return prevalence_to_array(prev, generation)


def case_status_from_liability(
    liability: np.ndarray,
    prevalence,
    sex: np.ndarray | None,
    generation: np.ndarray,
    mode: StandardizeMode,
) -> np.ndarray:
    """Probit liability-threshold case status.

    Standardizes ``liability`` under ``mode`` then flags individuals above the
    probit threshold ``ndtri(1 - K)`` as cases.  This is the single home for the
    standardize-then-threshold idiom shared by ``adult.ltm``, ``cure_frailty``,
    and ``simple_ltm``.

    The comparison is strict (``threshold < L``): on continuous liability the
    tie set is measure-zero, so the realised case fraction matches ``K`` and the
    ``<`` / ``<=`` choice is immaterial in practice.

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
    return ndtri(1.0 - np.asarray(prev)) < L
