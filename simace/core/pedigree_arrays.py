"""Pedigree columns as numpy arrays, addressable by ``id``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import polars as pl

__all__ = ["PedigreeArrays"]

_ID = "id"

# The id->position map is a direct-address table sized by ``max_id``, not by row
# count, so a sparse id space costs memory out of proportion to the pedigree.
# Real simACE ids are dense (see the class docstring), leaving this a guard
# against pathological input rather than a limit anyone should meet.
_MAX_MAP_BYTES = 512 * 1024**2


class PedigreeArrays:
    """A pedigree's columns as numpy arrays, addressable by ``id``.

    Replaces the ``df.set_index("id")`` idiom used throughout
    :mod:`simace.analysis`. That frame was doing double duty — an id->row
    lookup *and* a plain column container — and had accumulated several
    mutually inconsistent access idioms (``.loc`` gather, an explicit
    ``pd.Series`` position map, ``.reindex``, ``.isin`` on the index, and
    modules building their own local index). This type answers all of them:

    ==============================  =========================================
    Operation                       Call
    ==============================  =========================================
    values for a set of ids         ``ped.gather("A1", ids)``
    row positions for a set of ids  ``ped.positions(ids)``
    which ids are present           ``ped.contains(ids)``
    a whole column                  ``ped["A1"]``
    ==============================  =========================================

    **Lookup is a direct-address table**, ``pos[id] -> row``, not a hash map or
    a binary search. Measured on a 6M-row pedigree gathering 2M ids (best of 5):

    ===========================  ===========  ==============
    Approach                     dense ids    gapped ids
    ===========================  ===========  ==============
    ``pandas .loc[ids, col]``    110 ms       129 ms
    ``pandas get_indexer``       64 ms        71 ms
    ``np.searchsorted``          1118 ms      1130 ms
    **direct-address (build)**   **81 ms**    **75 ms**
    **direct-address (reused)**  **43 ms**    **44 ms**
    ===========================  ===========  ==============

    ``np.searchsorted`` loses badly even though simACE ids are always sorted —
    the cost is cache behaviour in the binary search, not a missing sort, so
    pre-sorting does not rescue it. Direct addressing wins because simACE ids
    are dense: :func:`simace.simulation.simulate.run_simulation` assigns them
    as ``np.arange(n) + offset``, so the full pedigree is exactly ``0..N-1``,
    and ascertainment only drops rows (leaving gaps, never renumbering). The
    map therefore costs ``4 * (max_id + 1)`` bytes — 24 MB at 6M rows.

    **Misses are strict, and negative ids are rejected.** ``-1`` is simACE's
    universal missing sentinel (``mother``, ``father``, ``twin``), and a bare
    ``pos[-1]`` would silently return the *last* row where ``.loc[-1]`` raised.
    :meth:`positions` and :meth:`gather` therefore raise on any unknown id and
    raise a distinct :class:`ValueError` on a negative one. :meth:`contains` is
    the sanctioned way to ask first: it treats a negative or out-of-range id as
    simply absent, matching what ``.isin(index)`` did — which is load-bearing,
    since ascertainment severs ``mother``/``father`` independently and a
    surviving row may carry one real parent and one ``-1``.

    Column arrays are views into the source frame, not copies (verified against
    pandas 3 for the numpy-backed dtypes this pipeline writes), so construction
    costs only the map.
    """

    __slots__ = ("_cols", "_max_id", "_n", "_pos")

    def __init__(self, columns: Mapping[str, np.ndarray]) -> None:
        """Build from a mapping of column name to array.

        Args:
            columns: Column arrays, all of equal length, including ``id``.

        Raises:
            ValueError: If ``id`` is absent, columns differ in length, ids are
                negative or duplicated, or the id space is too sparse for a
                direct-address map (see ``_MAX_MAP_BYTES``).
        """
        if _ID not in columns:
            raise ValueError(f"PedigreeArrays requires an {_ID!r} column; got {sorted(columns)}")

        ids = np.asarray(columns[_ID])
        n = len(ids)
        bad_len = {name: len(arr) for name, arr in columns.items() if len(arr) != n}
        if bad_len:
            raise ValueError(f"PedigreeArrays column length mismatch against {_ID} (n={n}): {bad_len}")

        if n and int(ids.min()) < 0:
            neg = ids[ids < 0]
            raise ValueError(
                f"PedigreeArrays: {len(neg)} negative id(s) in the {_ID} column "
                f"(first {min(len(neg), 10)}: {neg[:10].tolist()}). "
                f"{_ID} is a real identifier; -1 is the missing-parent sentinel and never a row."
            )

        max_id = int(ids.max()) if n else -1
        n_slots = max_id + 1
        if n_slots * np.dtype(np.int32).itemsize > _MAX_MAP_BYTES:
            raise ValueError(
                f"PedigreeArrays: id space too sparse for a direct-address map "
                f"(n={n}, max_id={max_id}, would need {n_slots * 4 / 1024**2:.0f} MB "
                f"> {_MAX_MAP_BYTES / 1024**2:.0f} MB)."
            )

        pos = np.full(n_slots, -1, dtype=np.int32)
        pos[ids] = np.arange(n, dtype=np.int32)
        # A duplicate id overwrites its earlier slot, so fewer than n slots end
        # up filled. Cheaper than np.unique, which dominated profiles here.
        if int((pos >= 0).sum()) != n:
            counts = np.bincount(ids, minlength=n_slots)
            dup = np.flatnonzero(counts > 1)
            raise ValueError(
                f"PedigreeArrays: {len(dup)} duplicated id(s) "
                f"(first {min(len(dup), 10)}: {dup[:10].tolist()}); {_ID} must be unique."
            )

        self._cols = dict(columns)
        self._pos = pos
        self._n = n
        self._max_id = max_id

    @classmethod
    def from_frame(cls, df: pd.DataFrame | pl.DataFrame) -> PedigreeArrays:
        """Build from a pedigree DataFrame, taking every column as an array.

        Row *i* of ``df`` is position *i* here — the id order is preserved
        exactly, so positions returned by :meth:`positions` index any array
        derived from ``df`` in its original row order. Accepts pandas and
        polars frames alike — the columns come out through ``.to_numpy()``
        either way (zero-copy for the null-free numeric dtypes this pipeline
        carries).

        Args:
            df: Pedigree with an ``id`` column.

        Returns:
            A ``PedigreeArrays`` over ``df``'s columns.
        """
        return cls({name: df[name].to_numpy() for name in df.columns})

    def __len__(self) -> int:
        """Return the number of rows."""
        return self._n

    def __getitem__(self, col: str) -> np.ndarray:
        """Return a whole column as an array, in row order."""
        try:
            return self._cols[col]
        except KeyError:
            raise KeyError(f"PedigreeArrays has no column {col!r}; available: {sorted(self._cols)}") from None

    def __contains__(self, col: str) -> bool:
        """Return whether ``col`` is present, so ``"sex" in ped`` works."""
        return col in self._cols

    @property
    def columns(self) -> list[str]:
        """Return the available column names."""
        return sorted(self._cols)

    @property
    def ids(self) -> np.ndarray:
        """Return the ``id`` column, in row order."""
        return self._cols[_ID]

    def contains(self, ids: np.ndarray) -> np.ndarray:
        """Return a boolean mask of which ``ids`` are present in this pedigree.

        Negative and out-of-range ids are reported as absent rather than
        raising — this is the sanctioned pre-check, and the missing-parent
        sentinel ``-1`` is a legitimate thing to ask about. Replaces
        ``ids.isin(df_indexed.index)``.

        Args:
            ids: Ids to test. Not required to be unique or sorted.

        Returns:
            Boolean array of the same shape as ``ids``.
        """
        arr = np.asarray(ids)
        present = np.zeros(arr.shape, dtype=bool)
        if arr.size == 0:
            return present
        in_range = (arr >= 0) & (arr <= self._max_id)
        present[in_range] = self._pos[arr[in_range]] >= 0
        return present

    def positions(self, ids: np.ndarray) -> np.ndarray:
        """Return the row position of each id.

        Args:
            ids: Ids to locate. Not required to be unique or sorted.

        Returns:
            ``int32`` row positions, same shape as ``ids``, suitable for
            fancy-indexing any array in this pedigree's row order.

        Raises:
            ValueError: If any id is negative. ``-1`` is the missing-parent
                sentinel; filter it out, or use :meth:`contains` first.
            KeyError: If any id is absent from this pedigree, matching what
                ``df_indexed.loc[ids]`` did.
        """
        arr = np.asarray(ids)
        if arr.size == 0:
            return np.empty(arr.shape, dtype=np.int32)

        if int(arr.min()) < 0:
            neg = arr[arr < 0]
            raise ValueError(
                f"PedigreeArrays.positions: {len(neg)} negative id(s) "
                f"(first {min(len(neg), 10)}: {neg[:10].tolist()}). "
                f"-1 is the missing-parent sentinel, not a row; filter it out or use contains()."
            )

        if int(arr.max()) > self._max_id:
            over = arr[arr > self._max_id]
            raise KeyError(
                f"PedigreeArrays.positions: {len(over)} id(s) above max_id={self._max_id} "
                f"(first {min(len(over), 10)}: {over[:10].tolist()})."
            )

        found = self._pos[arr]
        if bool((found < 0).any()):
            missing = arr[found < 0]
            raise KeyError(
                f"PedigreeArrays.positions: {len(missing)} id(s) not in this pedigree "
                f"(first {min(len(missing), 10)}: {missing[:10].tolist()})."
            )
        return found

    def gather(self, col: str, ids: np.ndarray) -> np.ndarray:
        """Return ``col``'s values for each id, in the order given.

        Equivalent to ``self[col][self.positions(ids)]``, and replaces
        ``df_indexed.loc[ids, col].values``.

        Args:
            col: Column name.
            ids: Ids to gather. Not required to be unique or sorted.

        Returns:
            Values of ``col``, same shape as ``ids``.

        Raises:
            ValueError: If any id is negative (see :meth:`positions`).
            KeyError: If ``col`` is absent, or any id is absent.
        """
        return self[col][self.positions(ids)]
