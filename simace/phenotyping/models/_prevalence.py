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

__all__ = ["prevalence_to_array", "resolve_prevalence"]


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
