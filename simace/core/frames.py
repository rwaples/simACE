"""Frame-library compatibility helpers (transitional, ADR 0015).

The external ``pedigree_graph`` package accepts ``dict[str, np.ndarray]`` and
pandas frames but not (yet) polars. :func:`pedigree_graph_input` is the NumPy
dict escape hatch the family uses so polars callers never exercise pandas
compatibility. Once pedigree-graph's structural frame protocol lands, polars
frames pass natively and this helper reduces to the identity for them too.
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
