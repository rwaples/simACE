"""Unit tests for the hierarchical config flattening logic in simace.config."""

from pathlib import Path

import pytest
import yaml

from simace.config import (
    flatten_hierarchical as _flatten_hierarchical,
)
from simace.config import (
    resolve_defaults,
    resolve_scenarios,
)


class TestFlattenPassthrough:
    """Flat dicts with no section keys pass through unchanged."""

    def test_empty_dict(self):
        assert _flatten_hierarchical({}) == {}

    def test_globals_only(self):
        d = {"seed": 42, "N": 10000, "A1": 0.5}
        assert _flatten_hierarchical(d) == d

    def test_all_flat_keys_passthrough(self):
        d = {
            "seed": 42,
            "replicates": 1,
            "N": 20000,
            "A1": 0.5,
            "C1": 0.1,
            "E1": 0.4,
            "phenotype_model1": "frailty",
            "censor_age": 80,
        }
        assert _flatten_hierarchical(d) == d


class TestFlattenPedigree:
    """Pedigree section flattening."""

    def test_simple_pedigree_keys(self):
        d = {"pedigree": {"mating_lambda": 0.5, "p_mztwin": 0.02}}
        result = _flatten_hierarchical(d)
        assert result == {"mating_lambda": 0.5, "p_mztwin": 0.02}

    def test_trait_nesting(self):
        d = {"pedigree": {"trait1": {"A": 0.8, "C": 0.0, "E": 0.2}}}
        result = _flatten_hierarchical(d)
        assert result == {"A1": 0.8, "C1": 0.0, "E1": 0.2}

    def test_trait2(self):
        d = {"pedigree": {"trait2": {"A": 0.3, "C": 0.1, "E": 0.6}}}
        result = _flatten_hierarchical(d)
        assert result == {"A2": 0.3, "C2": 0.1, "E2": 0.6}

    def test_cross_trait_correlations(self):
        d = {"pedigree": {"rA": 0.3, "rC": 0.1, "rE": 0.05}}
        result = _flatten_hierarchical(d)
        assert result == {"rA": 0.3, "rC": 0.1, "rE": 0.05}

    def test_generation_varying_values_preserved(self):
        """Dict values (like per-generation E) are preserved opaque."""
        gen_E = {0: 0.30, 3: 0.35, 4: 0.40, 5: 0.45}
        d = {"pedigree": {"trait1": {"E": gen_E}}}
        result = _flatten_hierarchical(d)
        assert result == {"E1": gen_E}


class TestFlattenPhenotype:
    """Phenotype section flattening."""

    def test_trait1_model(self):
        d = {"phenotype": {"trait1": {"model": "frailty"}}}
        result = _flatten_hierarchical(d)
        assert result == {"phenotype_model1": "frailty"}

    def test_trait1_params_preserved(self):
        params = {"distribution": "weibull", "scale": 2160, "rho": 0.8}
        d = {"phenotype": {"trait1": {"params": params}}}
        result = _flatten_hierarchical(d)
        assert result == {"phenotype_params1": params}

    def test_full_trait(self):
        # PR3: prevalence lives inside per-model params (adult/cure_frailty only).
        d = {
            "phenotype": {
                "trait1": {
                    "model": "cure_frailty",
                    "params": {
                        "distribution": "lognormal",
                        "mu": 2.35,
                        "sigma": 0.71,
                        "prevalence": 0.05,
                    },
                    "beta": 0.8,
                    "beta_sex": 0.5,
                },
            }
        }
        result = _flatten_hierarchical(d)
        assert result["phenotype_model1"] == "cure_frailty"
        assert result["beta1"] == 0.8
        assert result["beta_sex1"] == 0.5
        assert result["phenotype_params1"]["prevalence"] == 0.05
        assert result["phenotype_params1"]["distribution"] == "lognormal"

    def test_sex_specific_prevalence_preserved(self):
        # PR3: prevalence is now a per-model param, not a top-level trait key.
        d = {
            "phenotype": {
                "trait1": {
                    "model": "adult",
                    "params": {
                        "method": "ltm",
                        "prevalence": {"female": 0.08, "male": 0.12},
                    },
                }
            }
        }
        result = _flatten_hierarchical(d)
        assert result["phenotype_model1"] == "adult"
        assert result["phenotype_params1"]["prevalence"] == {"female": 0.08, "male": 0.12}

    def test_top_level_prevalence_rejected(self):
        # PR3: top-level phenotype.traitN.prevalence is no longer accepted.
        d = {"phenotype": {"trait1": {"prevalence": 0.05}}}
        with pytest.raises(ValueError, match="no longer supported"):
            _flatten_hierarchical(d)


class TestFlattenOtherSections:
    """Censoring, sampling, analysis sections."""

    def test_censoring(self):
        d = {"censoring": {"max_age": 80, "death_scale": 164, "death_rho": 2.73}}
        result = _flatten_hierarchical(d)
        assert result == {"censor_age": 80, "death_scale": 164, "death_rho": 2.73}

    def test_sampling(self):
        d = {"sampling": {"N_sample": 5000, "case_ascertainment_ratio": 2.0}}
        result = _flatten_hierarchical(d)
        assert result == {"N_sample": 5000, "case_ascertainment_ratio": 2.0}

    def test_analysis(self):
        d = {"analysis": {"max_degree": 3, "estimate_inbreeding": True}}
        result = _flatten_hierarchical(d)
        assert result == {"max_degree": 3, "estimate_inbreeding": True}


class TestFlattenMixed:
    """Mixed globals + sections."""

    def test_globals_plus_sections(self):
        d = {
            "seed": 42,
            "N": 10000,
            "pedigree": {"rA": 0.7},
            "censoring": {"max_age": 90},
        }
        result = _flatten_hierarchical(d)
        assert result == {"seed": 42, "N": 10000, "rA": 0.7, "censor_age": 90}


class TestFlattenErrors:
    """Error cases."""

    def test_unknown_section_key_raises(self):
        d = {"pedigree": {"bogus_key": 42}}
        with pytest.raises(ValueError, match="Unknown hierarchical config key"):
            _flatten_hierarchical(d)

    def test_flat_hierarchical_collision_raises(self):
        d = {"A1": 0.5, "pedigree": {"trait1": {"A": 0.8}}}
        with pytest.raises(ValueError, match="both flat and hierarchical"):
            _flatten_hierarchical(d)

    def test_unknown_nested_key_raises(self):
        d = {"phenotype": {"trait1": {"bogus": 42}}}
        with pytest.raises(ValueError, match="Unknown hierarchical config key"):
            _flatten_hierarchical(d)


class TestRoundTrip:
    """Loading the actual _default.yaml and verifying it flattens to expected keys."""

    def test_default_yaml_flattens_to_expected_keys(self):
        with open("config/_default.yaml") as f:
            raw = yaml.safe_load(f)
        defaults = raw["defaults"]
        flat = _flatten_hierarchical(defaults)

        expected_keys = {
            "seed",
            "replicates",
            "folder",
            "N",
            "G_ped",
            "G_pheno",
            "G_sim",
            "standardize",
            "plot_format",
            "drop_from",
            "use_gene_drop",
            "mating_lambda",
            "p_mztwin",
            "assort1",
            "assort2",
            "assort_matrix",
            "A1",
            "C1",
            "E1",
            "A2",
            "C2",
            "E2",
            "rA",
            "rC",
            "rE",
            "phenotype_model1",
            "phenotype_params1",
            "beta1",
            "beta_sex1",
            "phenotype_model2",
            "phenotype_params2",
            "beta2",
            "beta_sex2",
            "censor_age",
            "gen_censoring",
            "death_scale",
            "death_rho",
            "N_sample",
            "case_ascertainment_ratio",
            "pedigree_dropout_rate",
            "max_degree",
            "estimate_inbreeding",
            "tstrait_num_causal",
            "tstrait_frac_causal",
            "tstrait_maf_threshold",
            "tstrait_alpha",
            "tstrait_effect_mean",
            "tstrait_effect_var",
            "tstrait_trait_id",
            "tstrait_share_architecture",
        }
        assert set(flat.keys()) == expected_keys

    def test_default_yaml_values_match(self):
        with open("config/_default.yaml") as f:
            raw = yaml.safe_load(f)
        flat = _flatten_hierarchical(raw["defaults"])

        assert flat["seed"] == 42
        assert flat["A1"] == 0.5
        assert flat["C2"] == 0.2
        assert flat["phenotype_model1"] == "frailty"
        assert flat["phenotype_params1"]["distribution"] == "weibull"
        assert flat["phenotype_params1"]["scale"] == 2160
        assert flat["beta2"] == 1.5
        # PR3: _default.yaml uses frailty for both traits, which carries no prevalence.
        assert "prevalence" not in flat["phenotype_params1"]
        assert flat["censor_age"] == 80
        assert flat["death_scale"] == 164


class TestGenCensoringCoercion:
    """resolve_defaults / resolve_scenarios coerce gen_censoring keys to int.

    YAML loads unquoted scalar keys per their parsed type, so a config that
    quotes its generation keys (``'0': [80, 80]``) would otherwise reach
    downstream code as ``str`` keys and break ``gen_censoring[gen_int]``
    lookups.  The coercion makes wrappers and rules robust to the YAML quirk.
    """

    def _write(self, tmp_path: Path, name: str, body: dict) -> Path:
        path = tmp_path / name
        with open(path, "w") as fh:
            yaml.safe_dump(body, fh)
        return path

    def test_resolve_defaults_string_keys_coerced(self, tmp_path):
        body = {
            "defaults": {
                "seed": 1,
                "censoring": {
                    "max_age": 80,
                    "gen_censoring": {"0": [80, 80], "1": [40, 80]},
                    "death_scale": 164,
                    "death_rho": 2.73,
                },
            }
        }
        self._write(tmp_path, "_default.yaml", body)
        flat = resolve_defaults(tmp_path)
        assert flat["gen_censoring"] == {0: [80, 80], 1: [40, 80]}
        assert all(isinstance(k, int) for k in flat["gen_censoring"])

    def test_resolve_defaults_int_keys_passthrough(self, tmp_path):
        body = {
            "defaults": {
                "censoring": {
                    "max_age": 80,
                    "gen_censoring": {0: [80, 80], 1: [40, 80]},
                    "death_scale": 164,
                    "death_rho": 2.73,
                },
            }
        }
        self._write(tmp_path, "_default.yaml", body)
        flat = resolve_defaults(tmp_path)
        assert flat["gen_censoring"] == {0: [80, 80], 1: [40, 80]}

    def test_resolve_scenarios_coerces_overrides(self, tmp_path):
        # Defaults supply a complete config so per-folder overrides are valid.
        defaults_body = {
            "defaults": {
                "seed": 42,
                "replicates": 1,
                "folder": "x",
                "N": 100,
                "G_ped": 4,
                "G_pheno": 4,
                "G_sim": 4,
                "standardize": True,
                "plot_format": "png",
                "drop_from": 0,
                "use_gene_drop": False,
                "pedigree": {
                    "mating_lambda": 0.5,
                    "p_mztwin": 0.0,
                    "assort1": 0.0,
                    "assort2": 0.0,
                    "assort_matrix": None,
                    "trait1": {"A": 0.5, "C": 0.0, "E": 0.5},
                    "trait2": {"A": 0.5, "C": 0.0, "E": 0.5},
                    "rA": 0.0,
                    "rC": 0.0,
                    "rE": 0.0,
                },
                "phenotype": {
                    "trait1": {
                        "model": "frailty",
                        "params": {"distribution": "weibull", "scale": 2000, "rho": 1.0},
                        "beta": 1.0,
                        "beta_sex": 0.0,
                    },
                    "trait2": {
                        "model": "frailty",
                        "params": {"distribution": "weibull", "scale": 2000, "rho": 1.0},
                        "beta": 1.0,
                        "beta_sex": 0.0,
                    },
                },
                "censoring": {
                    "max_age": 80,
                    "gen_censoring": {0: [80, 80]},
                    "death_scale": 164,
                    "death_rho": 2.73,
                },
                "sampling": {"N_sample": 100, "case_ascertainment_ratio": 1.0, "pedigree_dropout_rate": 0.0},
                "analysis": {"max_degree": 2, "estimate_inbreeding": False},
                "tstrait": {
                    "num_causal": 0,
                    "frac_causal": 0.0,
                    "maf_threshold": 0.0,
                    "alpha": 0.0,
                    "effect_mean": 0.0,
                    "effect_var": 0.0,
                    "trait_id": "t1",
                    "share_architecture": False,
                },
            }
        }
        self._write(tmp_path, "_default.yaml", defaults_body)
        scenario_body = {
            "scen_quoted": {
                "censoring": {
                    "max_age": 80,
                    "gen_censoring": {"2": [0, 80], "3": [0, 45]},
                    "death_scale": 164,
                    "death_rho": 2.73,
                },
            }
        }
        self._write(tmp_path, "x.yaml", scenario_body)
        scenarios = resolve_scenarios(tmp_path)
        assert scenarios["scen_quoted"]["gen_censoring"] == {2: [0, 80], 3: [0, 45]}

    def test_real_default_yaml_keys_are_ints(self):
        """The shipped _default.yaml round-trips to int keys via the loader."""
        flat = resolve_defaults("config")
        assert all(isinstance(k, int) for k in flat["gen_censoring"])
