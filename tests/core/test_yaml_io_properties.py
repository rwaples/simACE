"""Property-based round-trip for the numpy->builtin YAML normalizer.

``to_native`` is explicitly recursive over arbitrary nesting of numpy and
python scalars/arrays. For any such structure it must (a) leave no numpy type
behind, and (b) survive a ``safe_dump`` / ``safe_load`` round-trip — the
canonical shape a single fixture cannot probe across mixed dtypes and depths.
"""

import numpy as np
import yaml
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.core.yaml_io import to_native

_finite_float = st.floats(allow_nan=False, allow_infinity=False, width=64)

# Leaf values: python builtins, numpy scalars, and small numpy arrays. numpy
# string/complex scalars are intentionally excluded — to_native passes those
# through unchanged (a documented, out-of-scope gap), so they are not part of
# the round-trip contract.
_base = st.one_of(
    st.integers(min_value=-(2**40), max_value=2**40),
    _finite_float,
    st.booleans(),
    st.text(),
    st.builds(np.int32, st.integers(min_value=-(2**31), max_value=2**31 - 1)),
    st.builds(np.int64, st.integers(min_value=-(2**40), max_value=2**40)),
    st.builds(np.float64, _finite_float),
    st.builds(np.bool_, st.booleans()),
    hnp.arrays(np.int64, hnp.array_shapes(max_dims=2, max_side=4), elements=st.integers(-1000, 1000)),
    hnp.arrays(np.float64, hnp.array_shapes(max_dims=2, max_side=4), elements=_finite_float),
)

_nested = st.recursive(
    _base,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1), children, max_size=4),
    ),
    max_leaves=15,
)


def _assert_no_numpy(obj):
    assert not isinstance(obj, (np.generic, np.ndarray))
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_numpy(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_numpy(v)


@given(_nested)
def test_to_native_leaves_no_numpy_residue(obj):
    _assert_no_numpy(to_native(obj))


@given(_nested)
def test_to_native_yaml_roundtrips(obj):
    native = to_native(obj)
    reloaded = yaml.safe_load(yaml.safe_dump(native))
    assert reloaded == native
