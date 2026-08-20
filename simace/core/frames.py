"""Frame-library compatibility helpers (ADR 0015).

``pedigree_graph`` grew a structural ``FrameLike`` protocol that accepts polars
directly, but the family is pinned to a release predating it, and that pin is
deliberate: pair-extraction changes upstream can silently bias heritability
(CLAUDE.md gotcha #4). :func:`pedigree_graph_input` is the NumPy dict escape
hatch that spans the gap, so polars callers never exercise pandas
compatibility.

Retire this once the pin moves to a release carrying ``FrameLike``: polars
frames then pass natively and the helper reduces to the identity.
"""

from __future__ import annotations

__all__ = ["pedigree_graph_input"]

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def pedigree_graph_input(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | dict[str, np.ndarray]:
    """Return ``df`` in a form every ``PedigreeGraph`` constructor accepts.

    Polars frames become a dict of NumPy column arrays (zero-copy for the
    null-free numeric dtypes the pedigree carries); pandas frames pass
    through unchanged.
    """
    if isinstance(df, pl.DataFrame):
        return {c: df[c].to_numpy() for c in df.columns}
    return df
