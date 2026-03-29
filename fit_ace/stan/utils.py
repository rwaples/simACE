"""Shared utilities for Stan fitting scripts."""

import numpy as np
import pandas as pd


def compute_dii(ped: pd.DataFrame) -> np.ndarray:
    """Compute Henderson's D-inverse diagonal (non-inbred pedigree).

    dii[i] = 1 / (1 - 0.25 * n_known_parents[i])
    Founders: 1.0, one parent known: 4/3, both parents known: 2.0
    """
    n_parents = (ped.mother.values != -1).astype(int) + (ped.father.values != -1).astype(int)
    return 1.0 / (1.0 - 0.25 * n_parents)
