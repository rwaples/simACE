"""Property-based tests for the liability-threshold case mask.

``liability_threshold_mask`` is the single home of the ``ndtri(1 - K)``
convention shared by three phenotype models. Two properties pin it across the
whole open prevalence interval: the case set is monotone-nested in K (exact),
and on a standardized liability the realized case fraction matches K
(inverse-CDF round-trip) — where the existing test checks only one fixed K.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.phenotype.models._prevalence import liability_threshold_mask

_liability = hnp.arrays(
    np.float64,
    st.integers(min_value=1, max_value=200),
    elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
)
_prevalence = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)


@given(liability=_liability, k1=_prevalence, k2=_prevalence)
def test_case_set_monotone_in_prevalence(liability, k1, k2):
    lo, hi = sorted((k1, k2))
    mask_lo = liability_threshold_mask(liability, lo)
    mask_hi = liability_threshold_mask(liability, hi)
    # a larger target prevalence can only add cases, never remove them
    assert np.all(mask_lo <= mask_hi)


@settings(deadline=None, max_examples=40)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1), k=st.floats(min_value=0.02, max_value=0.5))
def test_realized_case_fraction_matches_prevalence(seed, k):
    n = 20_000
    liability = np.random.default_rng(seed).standard_normal(n)
    frac = liability_threshold_mask(liability, k).mean()
    tol = 5.0 * np.sqrt(k * (1.0 - k) / n)  # ~5 binomial standard errors
    assert abs(frac - k) <= tol
