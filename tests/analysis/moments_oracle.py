"""Build a :class:`~pedigree_graph.RelationshipMoments` from explicit pair index arrays.

The stats functions consume the moments table the engine produces
(:func:`simace.analysis.stats.moments.relationship_moments_for`); tests that
plant arbitrary pair sets (swap, permutation, hand-computed phi) need the same
table without a pedigree. This helper accumulates the exact integers the
engine would from ``(first_rows, second_rows)`` arrays, over the same factor
layout: ``category``, then ``first_generation``, ``first_sex``,
``first_affected{t}`` and ``second_sex``, ``second_affected{t}`` for every
factor column the frame carries.
"""

from __future__ import annotations

import math

import numpy as np
from pedigree_graph.moments import MomentAxis, RelationshipMoments

from simace.core.relationships import RELATIONSHIP_TYPES

_FIRST_FACTORS = ("generation", "sex", "affected1", "affected2")
_SECOND_FACTORS = ("sex", "affected1", "affected2")
_QUANTIZED_BITS = 43


def _exponent(column: np.ndarray) -> int:
    magnitude = float(np.max(np.abs(column))) if len(column) else 0.0
    if magnitude == 0.0:
        return 0
    mantissa, exponent = math.frexp(magnitude)
    return _QUANTIZED_BITS - exponent + (1 if mantissa == 0.5 else 0)


def moments_from_pairs(frame, pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> RelationshipMoments:
    """The exact moments table of *pairs* (row positions into *frame*) per category."""
    categories = tuple(RELATIONSHIP_TYPES)
    n = len(frame)
    first = {
        name: np.asarray(frame[name].to_numpy()).astype(np.int64) for name in _FIRST_FACTORS if name in frame.columns
    }
    second = {
        name: np.asarray(frame[name].to_numpy()).astype(np.int64) for name in _SECOND_FACTORS if name in frame.columns
    }
    if "generation" not in first:
        first = {"generation": np.zeros(n, dtype=np.int64), **first}
    columns = tuple(f"liability{t}" for t in (1, 2) if f"liability{t}" in frame.columns)
    values = {c: np.asarray(frame[c].to_numpy(), dtype=np.float64) for c in columns}
    exponents = np.array([_exponent(values[c]) for c in columns], dtype=np.int64)
    quantized = {c: [int(v) for v in np.rint(np.ldexp(values[c], int(exponents[i])))] for i, c in enumerate(columns)}

    def axes_of(role: str, factors: dict[str, np.ndarray]) -> tuple[list[MomentAxis], list[np.ndarray]]:
        axes, indices = [], []
        for name, column in factors.items():
            levels, index = np.unique(column, return_inverse=True)
            axes.append(MomentAxis(f"{role}_{name}", levels))
            indices.append(index)
        return axes, indices

    first_axes, first_index = axes_of("first", first)
    second_axes, second_index = axes_of("second", second)
    axes = (MomentAxis("category", np.array(categories, dtype=object)), *first_axes, *second_axes)
    shape = tuple(len(axis.levels) for axis in axes)
    k = len(columns)
    counts = np.zeros(shape, dtype=np.int64)
    sum_first = np.zeros((*shape, k), dtype=object)
    sum_second = np.zeros((*shape, k), dtype=object)
    sumsq_first = np.zeros((*shape, k), dtype=object)
    sumsq_second = np.zeros((*shape, k), dtype=object)
    cross = np.zeros((*shape, k), dtype=object)
    for ci, code in enumerate(categories):
        idx1, idx2 = pairs.get(code, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
        for a, b in zip(idx1.tolist(), idx2.tolist(), strict=True):
            cell = (ci, *(int(ix[a]) for ix in first_index), *(int(ix[b]) for ix in second_index))
            counts[cell] += 1
            for j, c in enumerate(columns):
                qa, qb = quantized[c][a], quantized[c][b]
                sum_first[(*cell, j)] += qa
                sum_second[(*cell, j)] += qb
                sumsq_first[(*cell, j)] += qa * qa
                sumsq_second[(*cell, j)] += qb * qb
                cross[(*cell, j)] += qa * qb
    return RelationshipMoments(
        axes=axes,
        columns=columns,
        products=tuple((f"first.{c}", f"second.{c}") for c in columns),
        exponents=exponents,
        counts=counts,
        q_sum_first=sum_first,
        q_sum_second=sum_second,
        q_sumsq_first=sumsq_first,
        q_sumsq_second=sumsq_second,
        q_cross=cross,
        symmetric="canonical",
    )
