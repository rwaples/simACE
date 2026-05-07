"""Tests for the ``@stage`` decorator in ``simace.core.stage``."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from simace.core.schema import CENSORED, PEDIGREE, PHENOTYPE
from simace.core.stage import stage


def _make_pedigree_df() -> pd.DataFrame:
    n = 4
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int32),
            "generation": np.zeros(n, dtype=np.int32),
            "sex": np.zeros(n, dtype=np.int8),
            "mother": -np.ones(n, dtype=np.int32),
            "father": -np.ones(n, dtype=np.int32),
            "twin": -np.ones(n, dtype=np.int32),
            "household_id": np.arange(n, dtype=np.int32),
            "A1": np.zeros(n, dtype=np.float32),
            "C1": np.zeros(n, dtype=np.float32),
            "E1": np.zeros(n, dtype=np.float32),
            "liability1": np.zeros(n, dtype=np.float64),
            "A2": np.zeros(n, dtype=np.float32),
            "C2": np.zeros(n, dtype=np.float32),
            "E2": np.zeros(n, dtype=np.float32),
            "liability2": np.zeros(n, dtype=np.float64),
        }
    )


def _make_phenotype_df() -> pd.DataFrame:
    df = _make_pedigree_df()
    df["t1"] = np.ones(len(df), dtype=np.float32)
    df["t2"] = np.ones(len(df), dtype=np.float32)
    return df


class TestValidation:
    def test_input_validation_passes(self):
        @stage(reads=PEDIGREE, writes=PHENOTYPE)
        def fn(pedigree, *, k):
            return _make_phenotype_df()

        result = fn(_make_pedigree_df(), k=1)
        assert "t1" in result.columns

    def test_input_validation_fails_with_stage_name(self):
        @stage(reads=PEDIGREE, writes=PHENOTYPE)
        def run_phenotype(pedigree, *, k):
            return _make_phenotype_df()

        bad = _make_pedigree_df().drop(columns=["liability1"])
        with pytest.raises(ValueError, match=r"phenotype input.*liability1"):
            run_phenotype(bad, k=1)

    def test_output_validation_fails_with_stage_name(self):
        @stage(reads=PEDIGREE, writes=PHENOTYPE)
        def run_phenotype(pedigree, *, k):
            return pedigree  # missing t1, t2

        with pytest.raises(ValueError, match=r"phenotype output.*t1"):
            run_phenotype(_make_pedigree_df(), k=1)

    def test_reads_none_skips_input_check(self):
        @stage(reads=None, writes=PEDIGREE)
        def run_simulation(*, n):
            return _make_pedigree_df()

        result = run_simulation(n=4)
        assert len(result) == 4

    def test_writes_none_skips_output_check(self):
        @stage(reads=PEDIGREE, writes=None)
        def fn(pedigree):
            return "not a dataframe"

        assert fn(_make_pedigree_df()) == "not a dataframe"

    def test_input_via_kwarg_resolves_to_first_param(self):
        """run_wrapper passes the DataFrame as a kwarg keyed by parameter name."""

        @stage(reads=PEDIGREE, writes=PHENOTYPE)
        def run_phenotype(pedigree, *, k):
            return _make_phenotype_df()

        result = run_phenotype(pedigree=_make_pedigree_df(), k=1)
        assert "t1" in result.columns

    def test_input_via_kwarg_validation_fails_with_stage_name(self):
        @stage(reads=PEDIGREE, writes=PHENOTYPE)
        def run_phenotype(pedigree, *, k):
            return _make_phenotype_df()

        bad = _make_pedigree_df().drop(columns=["liability1"])
        with pytest.raises(ValueError, match=r"phenotype input.*liability1"):
            run_phenotype(pedigree=bad, k=1)

    def test_no_input_dataframe_raises(self):
        @stage(reads=PEDIGREE, writes=PEDIGREE)
        def fn(pedigree=None):
            return pedigree if pedigree is not None else _make_pedigree_df()

        with pytest.raises(TypeError, match=r"called with no input DataFrame"):
            fn()


class TestSignaturePreservation:
    def test_inspect_signature_returns_original(self):
        @stage(reads=PEDIGREE, writes=PHENOTYPE)
        def run_phenotype(pedigree, *, G_pheno: int, seed: int = 42):
            return _make_phenotype_df()

        sig = inspect.signature(run_phenotype)
        params = list(sig.parameters.values())
        assert params[0].name == "pedigree"
        assert params[1].name == "G_pheno"
        assert params[1].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[2].name == "seed"
        assert params[2].default == 42

    def test_wrapper_carries_original_name(self):
        @stage(reads=PEDIGREE, writes=PHENOTYPE)
        def run_phenotype(pedigree):
            return _make_phenotype_df()

        assert run_phenotype.__name__ == "run_phenotype"


class TestMetadata:
    def test_attributes_present_and_correct(self):
        @stage(reads=PHENOTYPE, writes=CENSORED)
        def run_censor(phenotype, *, seed):
            return phenotype

        assert run_censor.reads is PHENOTYPE
        assert run_censor.writes is CENSORED
        assert run_censor.stage_name == "censor"

    def test_default_stage_name_strips_run_prefix(self):
        @stage(reads=PEDIGREE, writes=PEDIGREE)
        def run_dropout(pedigree):
            return pedigree

        assert run_dropout.stage_name == "dropout"

    def test_name_override_honored(self):
        @stage(reads=PEDIGREE, writes=PEDIGREE, name="custom")
        def run_dropout(pedigree):
            return pedigree

        assert run_dropout.stage_name == "custom"

    def test_name_used_in_error_messages(self):
        @stage(reads=PEDIGREE, writes=PEDIGREE, name="custom")
        def run_dropout(pedigree):
            return pedigree

        bad = _make_pedigree_df().drop(columns=["liability1"])
        with pytest.raises(ValueError, match=r"custom input"):
            run_dropout(bad)


class TestPassthrough:
    def test_early_return_passthrough_satisfies_output_check(self):
        """run_sample-style early return: no transform, output equals input."""

        @stage(reads=CENSORED, writes=CENSORED)
        def run_sample(phenotype, *, n_sample):
            if n_sample <= 0:
                return phenotype  # early return passthrough
            return phenotype.iloc[:n_sample].reset_index(drop=True)

        df = _make_phenotype_df()
        for trait in (1, 2):
            df[f"age_censored{trait}"] = np.zeros(len(df), dtype=bool)
            df[f"t_observed{trait}"] = np.ones(len(df), dtype=np.float32)
            df[f"death_censored{trait}"] = np.zeros(len(df), dtype=bool)
            df[f"affected{trait}"] = np.ones(len(df), dtype=bool)
        df["death_age"] = np.full(len(df), 80.0, dtype=np.float32)

        out = run_sample(df, n_sample=0)
        assert len(out) == len(df)

    def test_pedigree_passthrough(self):
        """run_dropout-style early return: same shape in and out."""

        @stage(reads=PEDIGREE, writes=PEDIGREE)
        def run_dropout(pedigree, *, rate):
            if rate <= 0:
                return pedigree
            return pedigree.iloc[1:].reset_index(drop=True)

        df = _make_pedigree_df()
        out = run_dropout(df, rate=0)
        assert len(out) == len(df)
