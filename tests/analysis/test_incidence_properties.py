"""Property-based competing-risks invariants for the Aalen-Johansen estimator.

For any cohort, at every age on the grid the cause-specific cumulative
incidences and the survival function partition probability:
``F_disease + F_death + S == 1``; the CIFs are non-decreasing, S is
non-increasing, and all three stay in [0, 1]. These must hold across tied event
times, all-censored / all-death cohorts, and delayed entry — the edge-case
cross-product a single hand-built table cannot cover.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.analysis.stats.incidence import _aalen_johansen

_AGES = np.linspace(0.0, 120.0, 50)


@st.composite
def _aj_inputs(draw):
    n = draw(st.integers(min_value=1, max_value=60))
    exit_time = draw(
        hnp.arrays(np.float64, n, elements=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False))
    )
    event_type = draw(hnp.arrays(np.int8, n, elements=st.integers(0, 2)))
    if draw(st.booleans()):
        entry = draw(hnp.arrays(np.float64, n, elements=st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False)))
    else:
        entry = np.zeros(n)
    return entry, exit_time, event_type


@settings(deadline=None, max_examples=150)
@given(inp=_aj_inputs(), greenwood=st.booleans())
def test_aj_partition_bounds_monotonicity(inp, greenwood):
    entry, exit_time, event_type = inp
    out = _aalen_johansen(entry, exit_time, event_type, _AGES, greenwood=greenwood)
    f_disease = out["aj_disease"]
    f_death = out["aj_death"]
    survival = out["aj_survival"]

    # competing-risks partition holds at every grid age
    assert np.allclose(f_disease + f_death + survival, 1.0, atol=1e-9)

    # all three are probabilities
    for arr in (f_disease, f_death, survival):
        assert np.all(arr >= -1e-12)
        assert np.all(arr <= 1.0 + 1e-9)

    # CIFs non-decreasing, survival non-increasing
    assert np.all(np.diff(f_disease) >= -1e-12)
    assert np.all(np.diff(f_death) >= -1e-12)
    assert np.all(np.diff(survival) <= 1e-12)

    if greenwood:
        se = out["aj_se"]
        assert np.all(np.isfinite(se))
        assert np.all(se >= -1e-12)
