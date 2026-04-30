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
