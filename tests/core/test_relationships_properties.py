"""Property-based bound for the expected liability correlation.

On the valid variance simplex (A, C >= 0 with A + C <= 1), the expected
liability correlation ``2*kinship*A + C_shared*C`` must itself be a valid
correlation in [0, 1]. This is a compact tripwire for the documented
cross-package coupling risk (CLAUDE.md gotcha #4): if a kinship value drifted
above 0.5 in the external ``pedigree_graph`` registry, this bound would break
(corr > 1) where the exact-value unit tests would silently track the wrong
number.
"""

from hypothesis import assume, given
from hypothesis import strategies as st

from simace.core.relationships import RELATIONSHIP_TYPES, expected_liability_corr

_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(rt=st.sampled_from(RELATIONSHIP_TYPES), a=_unit, c=_unit)
def test_liability_corr_in_unit_interval_on_simplex(rt, a, c):
    assume(a + c <= 1.0)
    r = expected_liability_corr(rt, A=a, C=c)
    assert 0.0 <= r <= 1.0 + 1e-12


@given(
    rt=st.sampled_from(RELATIONSHIP_TYPES),
    a1=_unit,
    a2=_unit,
    c=_unit,
)
def test_liability_corr_monotone_nondecreasing_in_a(rt, a1, a2, c):
    lo, hi = sorted((a1, a2))
    assume(hi + c <= 1.0)
    assert expected_liability_corr(rt, A=lo, C=c) <= expected_liability_corr(rt, A=hi, C=c) + 1e-12
