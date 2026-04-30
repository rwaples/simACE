"""Tests for simace.core.yaml_io: numpy → native conversion + loader factory."""

import numpy as np
import yaml

from simace.core.yaml_io import to_native, yaml_loader


class TestToNative:
    def test_passthrough_python_scalars(self):
        assert to_native(1) == 1
        assert to_native(1.5) == 1.5
        assert to_native("x") == "x"
        assert to_native(True) is True

    def test_numpy_int(self):
        v = to_native(np.int64(7))
        assert v == 7
        assert type(v) is int

    def test_numpy_float(self):
        v = to_native(np.float32(0.25))
        assert v == 0.25
        assert type(v) is float

    def test_numpy_bool(self):
        v = to_native(np.bool_(True))
        assert v is True
        assert type(v) is bool

    def test_numpy_array(self):
        v = to_native(np.array([1, 2, 3], dtype=np.int32))
        assert v == [1, 2, 3]
        assert all(type(x) is int for x in v)

    def test_nested_dict_and_list(self):
        obj = {
            "a": np.int64(3),
            "b": [np.float64(0.1), np.float32(0.2)],
            "c": {"d": np.array([1.0, 2.0])},
        }
        out = to_native(obj)
        assert out == {"a": 3, "b": [0.1, pytest_approx_float(0.2)], "c": {"d": [1.0, 2.0]}}

    def test_yaml_dump_round_trip(self):
        """to_native output must serialize cleanly via yaml.dump."""
        obj = {"x": np.int32(2), "arr": np.array([1.5, 2.5])}
        text = yaml.safe_dump(to_native(obj))
        loaded = yaml.safe_load(text)
        assert loaded == {"x": 2, "arr": [1.5, 2.5]}


class TestYamlLoader:
    def test_returns_a_loader_class(self):
        cls = yaml_loader()
        # Must be a SafeLoader (or its C variant)
        assert issubclass(cls, yaml.SafeLoader) or cls is getattr(yaml, "CSafeLoader", yaml.SafeLoader)

    def test_loader_round_trips(self):
        cls = yaml_loader()
        loaded = yaml.load("a: 1\nb: [2, 3]\n", Loader=cls)
        assert loaded == {"a": 1, "b": [2, 3]}


def pytest_approx_float(v: float, tol: float = 1e-6) -> float:
    """Round-trip float32 → float promotes precision; loosen exact equality."""
    return float(np.float32(v))
