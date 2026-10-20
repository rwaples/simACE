"""``unique_sorted`` matches ``np.unique`` on values and inverse indices."""

from __future__ import annotations

import numpy as np
import pytest

from simace.core.numerics import unique_sorted


@pytest.mark.parametrize(
    "x",
    [
        np.array([], dtype=np.int64),
        np.array([7], dtype=np.int64),
        np.array([3, 1, 3, 2, 1, 1], dtype=np.int64),
        np.array([-1, -1, 0, 5, -1], dtype=np.int32),
        np.random.default_rng(0).integers(0, 50, 1_000, dtype=np.int64),
        np.random.default_rng(1).integers(0, 2**40, 1_000, dtype=np.int64),
    ],
)
def test_matches_np_unique(x: np.ndarray) -> None:
    uniques, inverse = unique_sorted(x, return_inverse=True)
    expected, expected_inverse = np.unique(x, return_inverse=True)
    np.testing.assert_array_equal(unique_sorted(x), expected)
    np.testing.assert_array_equal(uniques, expected)
    np.testing.assert_array_equal(inverse, expected_inverse)
    np.testing.assert_array_equal(uniques[inverse], x)


def test_rejects_float_input() -> None:
    with pytest.raises(TypeError, match="integer"):
        unique_sorted(np.array([np.nan, np.nan, 1.0]))
