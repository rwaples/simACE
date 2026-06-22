"""Unit tests for the public ``simace.config`` API.

Covers the resolver surface (``resolve_defaults`` / ``resolve_scenarios``),
``KNOWN_SIM_KEYS`` stability, the disjoint-key invariant that fitACE relies
on, and the cross-domain rejection of unknown keys.
"""

from pathlib import Path

import pytest

from simace.config import (
    _HIERARCHICAL_TO_FLAT,
    _SIM_FLAT_GLOBALS,
    KNOWN_SIM_KEYS,
    _is_zero_assort,
    _validate_pedigree_config,
    _validate_phenotype_config,
    flatten_hierarchical,
    get_all_folders,
    get_folder,
    get_param,
    get_scenarios_for_folder,
    resolve_defaults,
    resolve_scenarios,
)

REPO_ROOT = Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "config"


class TestKnownSimKeys:
    def test_known_sim_keys_is_union_of_mapping_and_globals(self):
        assert frozenset(_HIERARCHICAL_TO_FLAT.values()) | _SIM_FLAT_GLOBALS == KNOWN_SIM_KEYS

    def test_known_sim_keys_includes_globals(self):
        for g in ("seed", "replicates", "folder", "N", "G_ped", "plot_format"):
            assert g in KNOWN_SIM_KEYS

    def test_known_sim_keys_includes_flattened(self):
        for f in ("A1", "C2", "phenotype_model1", "censor_age", "N_sample"):
            assert f in KNOWN_SIM_KEYS

    def test_known_sim_keys_excludes_fit_keys(self):
        """Guard against drift: fit-domain flat keys must NEVER appear here."""
        fit_keys = {
            "epimight_mi_K",
            "epimight_rubin_level",
            "epimight_kinds",
            "pafgrs_ndegree",
            "grm_n_pcs",
            "grm_min_kinship",
            "export_grm_threshold",
            "iter_reml_trait",
            "iter_reml_prevalence",
        }
        assert fit_keys.isdisjoint(KNOWN_SIM_KEYS)


class TestResolveDefaults:
    def test_returns_flat_dict(self):
        defaults = resolve_defaults(CONFIG_DIR)
        # No hierarchical section keys should remain.
        assert "pedigree" not in defaults
        assert "phenotype" not in defaults
        assert "censoring" not in defaults

    def test_keys_match_known_sim_keys(self):
        defaults = resolve_defaults(CONFIG_DIR)
        assert set(defaults.keys()) == set(KNOWN_SIM_KEYS)

    def test_scalar_values_preserved(self):
        defaults = resolve_defaults(CONFIG_DIR)
        assert defaults["seed"] == 42
        assert defaults["A1"] == 0.5
        assert defaults["censor_age"] == 80
        assert defaults["phenotype_model1"] == "frailty"


class TestResolveScenarios:
    def test_every_scenario_file_loads(self):
        scenarios = resolve_scenarios(CONFIG_DIR)
        # At minimum the default folders must be present.
        folders = {s["folder"] for s in scenarios.values()}
        assert "base" in folders
        assert "test" in folders

    def test_folder_inferred_from_filename(self):
        scenarios = resolve_scenarios(CONFIG_DIR)
        for name, params in scenarios.items():
            # The folder key is always populated.
            assert "folder" in params, f"{name} missing folder"

    def test_unknown_scenario_key_rejected(self, tmp_path):
        (tmp_path / "_default.yaml").write_text(
            "defaults:\n  seed: 1\n  replicates: 1\n  folder: base\n"
            "  N: 100\n  G_ped: 2\n  G_pheno: 1\n  G_sim: 3\n"
            "  standardize: true\n  plot_format: png\n"
        )
        (tmp_path / "base.yaml").write_text("bad_scenario:\n  bogus_key: 42\n")
        with pytest.raises(ValueError, match="unknown keys"):
            resolve_scenarios(tmp_path)

    def test_underscore_prefixed_files_skipped(self, tmp_path):
        (tmp_path / "_default.yaml").write_text(
            "defaults:\n  seed: 1\n  replicates: 1\n  folder: base\n"
            "  N: 100\n  G_ped: 2\n  G_pheno: 1\n  G_sim: 3\n"
            "  standardize: true\n  plot_format: png\n"
        )
        (tmp_path / "_ignored.yaml").write_text("ignored_scenario:\n  seed: 99\n")
        scenarios = resolve_scenarios(tmp_path)
        assert "ignored_scenario" not in scenarios


class TestResolveScenariosEdgeCases:
    """Focused resolver edge cases."""

    @staticmethod
    def _flat_defaults() -> dict:
        return {
            "seed": 1,
            "replicates": 1,
            "folder": "base",
            "N": 10,
            "G_ped": 2,
            "G_pheno": 2,
            "G_sim": 2,
            "standardize": "global",
            "plot_format": "png",
            "E1": 0.5,
            "E2": 0.5,
            "phenotype_model1": "frailty",
            "phenotype_params1": {"distribution": "weibull"},
            "phenotype_model2": "frailty",
            "phenotype_params2": {"distribution": "weibull"},
            "mating_model": "standard",
        }

    def test_defaults_argument_skips_default_yaml_load(self, tmp_path):
        (tmp_path / "base.yaml").write_text("ok:\n  seed: 2\n")
        scenarios = resolve_scenarios(tmp_path, defaults=self._flat_defaults())
        assert scenarios["ok"]["seed"] == 2

    def test_invalid_folder_name_rejected(self, tmp_path):
        (tmp_path / "bad-name.yaml").write_text("scenario:\n  seed: 2\n")
        with pytest.raises(ValueError, match="Invalid folder name"):
            resolve_scenarios(tmp_path, defaults=self._flat_defaults())

    def test_empty_yaml_file_skipped(self, tmp_path):
        (tmp_path / "empty.yaml").write_text("")
        assert resolve_scenarios(tmp_path, defaults=self._flat_defaults()) == {}

    def test_duplicate_scenario_name_rejected(self, tmp_path):
        (tmp_path / "a.yaml").write_text("dup:\n  seed: 2\n")
        (tmp_path / "b.yaml").write_text("dup:\n  seed: 3\n")
        with pytest.raises(ValueError, match="Duplicate scenario"):
            resolve_scenarios(tmp_path, defaults=self._flat_defaults())

    def test_explicit_folder_is_preserved(self, tmp_path):
        (tmp_path / "base.yaml").write_text("ok:\n  folder: explicit\n")
        scenarios = resolve_scenarios(tmp_path, defaults=self._flat_defaults())
        assert scenarios["ok"]["folder"] == "explicit"


class TestPhenotypeConfigValidation:
    """Focused phenotype-model validation edge cases."""

    @staticmethod
    def _wrap(overrides: dict) -> dict:
        defaults = {
            "phenotype_model1": "frailty",
            "phenotype_params1": {"distribution": "weibull"},
            "phenotype_model2": "frailty",
            "phenotype_params2": {"distribution": "weibull"},
        }
        return {"defaults": defaults, "scenarios": {"sc": overrides}}

    def test_invalid_model_rejected(self):
        with pytest.raises(ValueError, match="not valid"):
            _validate_phenotype_config(self._wrap({"phenotype_model1": "bogus"}))

    def test_frailty_missing_distribution_rejected(self):
        with pytest.raises(ValueError, match="must include 'distribution'"):
            _validate_phenotype_config(self._wrap({"phenotype_model1": "frailty", "phenotype_params1": {}}))

    def test_frailty_invalid_distribution_rejected(self):
        with pytest.raises(ValueError, match="distribution='bad'"):
            _validate_phenotype_config(
                self._wrap({"phenotype_model1": "frailty", "phenotype_params1": {"distribution": "bad"}})
            )

    def test_adult_missing_method_rejected(self):
        with pytest.raises(ValueError, match="must include 'method'"):
            _validate_phenotype_config(
                self._wrap({"phenotype_model1": "adult", "phenotype_params1": {"prevalence": 0.1}})
            )

    def test_adult_invalid_method_rejected(self):
        with pytest.raises(ValueError, match="method='bad'"):
            _validate_phenotype_config(
                self._wrap({"phenotype_model1": "adult", "phenotype_params1": {"method": "bad", "prevalence": 0.1}})
            )

    def test_simple_ltm_missing_onset_rejected(self):
        with pytest.raises(ValueError, match="must include an 'onset' dict"):
            _validate_phenotype_config(
                self._wrap({"phenotype_model1": "simple_ltm", "phenotype_params1": {"prevalence": 0.1}})
            )

    def test_simple_ltm_invalid_onset_kind_rejected(self):
        with pytest.raises(ValueError, match=r"onset\.kind='bad'"):
            _validate_phenotype_config(
                self._wrap(
                    {
                        "phenotype_model1": "simple_ltm",
                        "phenotype_params1": {"onset": {"kind": "bad"}, "prevalence": 0.1},
                    }
                )
            )

    def test_simple_ltm_valid_onset_passes(self):
        _validate_phenotype_config(
            self._wrap(
                {
                    "phenotype_model1": "simple_ltm",
                    "phenotype_params1": {"onset": {"kind": "fixed", "age": 30}, "prevalence": 0.1},
                }
            )
        )

    def test_threshold_model_missing_prevalence_rejected(self):
        with pytest.raises(ValueError, match="must include 'prevalence'"):
            _validate_phenotype_config(
                self._wrap({"phenotype_model1": "adult", "phenotype_params1": {"method": "ltm"}})
            )

    def test_hazard_model_with_prevalence_rejected(self):
        with pytest.raises(ValueError, match="must NOT include 'prevalence'"):
            _validate_phenotype_config(
                self._wrap({"phenotype_model1": "first_passage", "phenotype_params1": {"prevalence": 0.1}})
            )


class TestAccessors:
    def test_get_param_scenario_override(self):
        config = {
            "defaults": {"seed": 42, "N": 100},
            "scenarios": {"s1": {"seed": 99, "folder": "x"}},
        }
        assert get_param(config, "s1", "seed") == 99
        assert get_param(config, "s1", "N") == 100

    def test_get_folder_and_listings(self):
        config = {
            "defaults": {"folder": "base"},
            "scenarios": {
                "a": {"folder": "f1"},
                "b": {"folder": "f1"},
                "c": {"folder": "f2"},
            },
        }
        assert get_folder(config, "a") == "f1"
        assert sorted(get_scenarios_for_folder(config, "f1")) == ["a", "b"]
        assert get_all_folders(config) == ["f1", "f2"]


class TestFlattenHierarchicalWithCustomMapping:
    """Regression: ``flatten_hierarchical`` must accept an external mapping."""

    def test_custom_mapping_flattens(self):
        mapping = {("foo", "bar"): "foo_bar"}
        d = {"foo": {"bar": 7}, "other": 1}
        result = flatten_hierarchical(d, mapping)
        assert result == {"foo_bar": 7, "other": 1}

    def test_custom_mapping_rejects_unknown(self):
        mapping = {("foo", "bar"): "foo_bar"}
        with pytest.raises(ValueError, match="Unknown hierarchical"):
            flatten_hierarchical({"foo": {"nope": 7}}, mapping)

    def test_mapping_none_uses_sim_mapping(self):
        d = {"pedigree": {"trait1": {"A": 0.3}}}
        assert flatten_hierarchical(d) == {"A1": 0.3}


class TestIsZeroAssort:
    """``_is_zero_assort`` distinguishes no-op AM settings from non-no-op."""

    def test_scalar_zero(self):
        assert _is_zero_assort(0)
        assert _is_zero_assort(0.0)

    def test_scalar_nonzero(self):
        assert not _is_zero_assort(0.3)
        assert not _is_zero_assort(-0.1)

    def test_dict_all_zero(self):
        assert _is_zero_assort({0: 0, 1: 0, 2: 0})
        assert _is_zero_assort({})  # vacuously zero

    def test_dict_any_nonzero(self):
        assert not _is_zero_assort({0: 0, 1: 0.3})
        assert not _is_zero_assort({0: -0.1})


class TestMatingModelValidation:
    """Strict-WF override validation in ``_validate_pedigree_config``.

    Operates on scenario-level overrides only — defaults from `_default.yaml`
    that the scenario doesn't explicitly touch are silently ignored at runtime.
    """

    @staticmethod
    def _wrap(scenario_overrides: dict) -> dict:
        """Build a minimal merged-config dict for the validator."""
        return {
            "defaults": {
                "E1": 0.5,
                "E2": 0.5,
                "mating_model": "standard",
            },
            "scenarios": {"sc": {**scenario_overrides, "E1": 0.5, "E2": 0.5}},
        }

    def test_default_standard_passes(self):
        _validate_pedigree_config(self._wrap({}))

    def test_explicit_wright_fisher_passes(self):
        _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher"}))

    def test_invalid_mating_model_raises(self):
        with pytest.raises(ValueError, match="mating_model='foobar'"):
            _validate_pedigree_config(self._wrap({"mating_model": "foobar"}))

    def test_wf_with_explicit_mating_lambda_raises(self):
        # Any explicit mating_lambda override is rejected under WF, even at default.
        with pytest.raises(ValueError, match="mating_lambda"):
            _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher", "mating_lambda": 0.5}))

    def test_wf_with_explicit_p_mztwin_zero_passes(self):
        _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher", "p_mztwin": 0}))

    def test_wf_with_explicit_p_mztwin_nonzero_raises(self):
        with pytest.raises(ValueError, match="p_mztwin"):
            _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher", "p_mztwin": 0.05}))

    def test_wf_with_inherited_default_p_mztwin_passes(self):
        # Default `p_mztwin: 0.02` lives in defaults; scenario doesn't touch it
        # → not flagged.
        config = {
            "defaults": {
                "E1": 0.5,
                "E2": 0.5,
                "mating_model": "standard",
                "p_mztwin": 0.02,
            },
            "scenarios": {"sc": {"mating_model": "wright_fisher", "E1": 0.5, "E2": 0.5}},
        }
        _validate_pedigree_config(config)

    def test_wf_with_assort1_zero_dict_passes(self):
        _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher", "assort1": {0: 0, 1: 0}}))

    def test_wf_with_assort1_nonzero_dict_raises(self):
        with pytest.raises(ValueError, match="assort1"):
            _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher", "assort1": {0: 0, 1: 0.3}}))

    def test_wf_with_assort1_nonzero_scalar_raises(self):
        with pytest.raises(ValueError, match="assort1"):
            _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher", "assort1": 0.3}))

    def test_wf_with_assort2_nonzero_raises(self):
        with pytest.raises(ValueError, match="assort2"):
            _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher", "assort2": 0.4}))

    def test_wf_with_assort_matrix_raises(self):
        with pytest.raises(ValueError, match="assort_matrix"):
            _validate_pedigree_config(
                self._wrap({"mating_model": "wright_fisher", "assort_matrix": [[0.3, 0.05], [0.05, 0.2]]})
            )

    def test_wf_with_assort_matrix_explicit_none_passes(self):
        _validate_pedigree_config(self._wrap({"mating_model": "wright_fisher", "assort_matrix": None}))
