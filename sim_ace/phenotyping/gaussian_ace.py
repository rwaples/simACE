"""Gaussian A+C+E phenotype by rescaling simulator columns.

Repurposes the A1, C1, E1 columns the ACE simulator already writes
(``sim_ace.simulation.simulate``) as a clean Gaussian phenotype for REML
benchmarking. Each component is multiplied by a single scalar chosen so its
empirical variance over the supplied rows matches the requested target.

The rescaling preserves MZ-twin identity (A1 is already copied across twins
at ``simulate.py:855``) and per-household C identity (``simulate.py:844``).
Truth variance components are returned so the bench script can compare them
against REML estimates.
"""

from __future__ import annotations

__all__ = [
    "simulate_gaussian_ace",
    "write_truth_json",
]

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_gaussian_ace(
    pedigree: pd.DataFrame,
    var_a: float = 0.5,
    var_c: float = 0.2,
    var_e: float = 0.3,
    seed: int = 0,
    fresh_e: bool = False,
    fresh_c: bool = False,
) -> pd.DataFrame:
    """Build a Gaussian y = A + C + E with requested variance components.

    Args:
        pedigree: DataFrame with at least ``id``, ``household_id``, ``A1``,
            ``C1``, ``E1`` columns.
        var_a: target additive genetic variance.
        var_c: target common-environment variance.
        var_e: target unique-environment variance.
        seed: RNG seed used when *fresh_e* or *fresh_c* is True.
        fresh_e: if True, redraw E as i.i.d. Gaussian N(0, var_e) instead of
            rescaling the simulator's E1 column. Useful for running multiple
            phenotype reps on the same pedigree without re-simulating.
        fresh_c: if True, draw a fresh per-household C ~ N(0, var_c); each
            individual inherits its household's value. Useful when the
            simulator was run with C1=0 (and the stored C1 column is zero)
            but you still want a nonzero common-environment component.

    Returns:
        DataFrame with columns ``id``, ``fid``, ``iid`` (both = ``id``),
        ``A``, ``C``, ``E``, ``y``. Truth variance components and h2/c2
        are attached under ``.attrs["truth"]``.

    Raises:
        ValueError: if required columns are missing or a component has
            near-zero variance (cannot be rescaled).
    """
    required = {"id", "household_id", "A1", "C1", "E1"}
    missing = required - set(pedigree.columns)
    if missing:
        raise ValueError(f"pedigree missing required columns: {sorted(missing)}")
    if var_a < 0 or var_c < 0 or var_e < 0:
        raise ValueError(f"variance components must be non-negative; got {var_a=}, {var_c=}, {var_e=}")

    n = len(pedigree)
    rng = np.random.default_rng(seed)

    a_raw = pedigree["A1"].to_numpy(dtype=np.float64)
    if fresh_c:
        hh = pedigree["household_id"].to_numpy()
        # Map household ids (may include -1 for singletons) to contiguous 0-based.
        uniq, inv = np.unique(hh, return_inverse=True)
        per_hh = rng.standard_normal(len(uniq))
        c_raw = per_hh[inv]
    else:
        c_raw = pedigree["C1"].to_numpy(dtype=np.float64)
    e_raw = rng.standard_normal(n) if fresh_e else pedigree["E1"].to_numpy(dtype=np.float64)

    A = _rescale_to_variance(a_raw, var_a, "A1")
    C = _rescale_to_variance(c_raw, var_c, "C" if fresh_c else "C1")
    E = _rescale_to_variance(e_raw, var_e, "E" if fresh_e else "E1")

    y = A + C + E

    ids = pedigree["id"].to_numpy()
    out = pd.DataFrame(
        {
            "id": ids,
            "fid": ids,
            "iid": ids,
            "A": A,
            "C": C,
            "E": E,
            "y": y,
        }
    )
    total = var_a + var_c + var_e
    out.attrs["truth"] = {
        "var_a": float(var_a),
        "var_c": float(var_c),
        "var_e": float(var_e),
        "var_total": float(total),
        "h2": float(var_a / total) if total > 0 else float("nan"),
        "c2": float(var_c / total) if total > 0 else float("nan"),
    }

    logger.info(
        "simulate_gaussian_ace: n=%d target=(%.4f,%.4f,%.4f) empirical_var_y=%.4f",
        n,
        var_a,
        var_c,
        var_e,
        float(y.var(ddof=1)),
    )
    return out


def _rescale_to_variance(x: np.ndarray, target: float, name: str) -> np.ndarray:
    """Multiply *x* by a scalar so ``var(x*s) == target``.

    Uses ddof=1 sample variance. When *target* is 0, returns zeros.
    """
    if target == 0.0:
        return np.zeros_like(x, dtype=np.float64)
    s = float(x.std(ddof=1))
    if s < 1e-12:
        raise ValueError(f"{name} has near-zero variance ({s:.3e}); cannot rescale to {target}")
    return x * (np.sqrt(target) / s)


def write_truth_json(truth: dict[str, Any], path: str | Path) -> Path:
    """Write *truth* dict as JSON; used as a sidecar to the phenotype output."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(truth, indent=2, sort_keys=True))
    return out
