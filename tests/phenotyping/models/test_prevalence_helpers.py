"""Direct tests for the prevalence-resolution helpers shared by adult/cure_frailty."""

import numpy as np
import pytest

from simace.phenotyping.models._prevalence import prevalence_to_array, resolve_prevalence


class TestPrevalenceToArray:
    def test_scalar_passthrough(self):
        gen = np.array([0, 0, 1, 1])
        out = prevalence_to_array(0.1, gen)
        assert out == 0.1

    def test_dict_expands_per_generation(self):
        gen = np.array([0, 0, 1, 1, 2, 2])
        out = prevalence_to_array({0: 0.05, 1: 0.10, 2: 0.20}, gen)
        np.testing.assert_array_equal(out, [0.05, 0.05, 0.10, 0.10, 0.20, 0.20])

    def test_dict_missing_key_raises(self):
        gen = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="missing generation 2"):
            prevalence_to_array({0: 0.1, 1: 0.2}, gen)


class TestResolvePrevalence:
    def test_scalar_passthrough(self):
        sex = np.zeros(5, dtype=int)
        gen = np.zeros(5, dtype=int)
        assert resolve_prevalence(0.15, sex, gen) == 0.15

    def test_per_generation_dict(self):
        sex = np.zeros(4, dtype=int)
        gen = np.array([0, 0, 1, 1])
        out = resolve_prevalence({0: 0.05, 1: 0.10}, sex, gen)
        np.testing.assert_array_equal(out, [0.05, 0.05, 0.10, 0.10])

    def test_sex_specific_scalar(self):
        sex = np.array([0, 1, 0, 1])  # 0 = female, 1 = male
        gen = np.zeros(4, dtype=int)
        out = resolve_prevalence({"female": 0.05, "male": 0.10}, sex, gen)
        np.testing.assert_array_equal(out, [0.05, 0.10, 0.05, 0.10])

    def test_sex_specific_with_per_gen_dict(self):
        sex = np.array([0, 1, 0, 1, 0, 1])
        gen = np.array([0, 0, 1, 1, 2, 2])
        out = resolve_prevalence(
            {
                "female": {0: 0.01, 1: 0.02, 2: 0.03},
                "male": {0: 0.04, 1: 0.05, 2: 0.06},
            },
            sex,
            gen,
        )
        np.testing.assert_array_equal(out, [0.01, 0.04, 0.02, 0.05, 0.03, 0.06])

    def test_sex_specific_missing_gen_raises(self):
        sex = np.array([0, 1])
        gen = np.array([0, 1])
        with pytest.raises(ValueError, match="missing generation"):
            resolve_prevalence(
                {"female": {0: 0.05}, "male": {0: 0.10}},  # missing gen 1
                sex,
                gen,
            )

    def test_only_one_sex_key_falls_through_to_scalar_path(self):
        # A dict without both 'female' AND 'male' is treated as a per-gen dict.
        sex = np.array([0, 1])
        gen = np.array([0, 1])
        out = resolve_prevalence({0: 0.05, 1: 0.10}, sex, gen)
        np.testing.assert_array_equal(out, [0.05, 0.10])
