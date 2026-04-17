"""Unit tests for the hierarchical config flattening logic in workflow/common.py."""

import pytest

from workflow.common import _flatten_hierarchical


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
        d = {
            "phenotype": {
                "trait1": {
                    "model": "cure_frailty",
                    "params": {"distribution": "lognormal", "mu": 2.35, "sigma": 0.71},
                    "beta": 0.8,
                    "beta_sex": 0.5,
                    "prevalence": 0.05,
                },
            }
        }
        result = _flatten_hierarchical(d)
        assert result["phenotype_model1"] == "cure_frailty"
        assert result["beta1"] == 0.8
        assert result["beta_sex1"] == 0.5
        assert result["prevalence1"] == 0.05
        assert result["phenotype_params1"]["distribution"] == "lognormal"

    def test_sex_specific_prevalence_preserved(self):
        d = {"phenotype": {"trait1": {"prevalence": {"female": 0.08, "male": 0.12}}}}
        result = _flatten_hierarchical(d)
        assert result == {"prevalence1": {"female": 0.08, "male": 0.12}}


class TestFlattenOtherSections:
    """Censoring, sampling, analysis, epimight, pafgrs sections."""

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

    def test_epimight(self):
        d = {"epimight": {"K": 10, "rubin_level": "meta", "kinds": ["PO", "FS"]}}
        result = _flatten_hierarchical(d)
        assert result == {"epimight_mi_K": 10, "epimight_rubin_level": "meta", "epimight_kinds": ["PO", "FS"]}

    def test_pafgrs(self):
        d = {"pafgrs": {"max_degree_pafgrs": 4}}
        result = _flatten_hierarchical(d)
        assert result == {"pafgrs_ndegree": 4}


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
        import yaml

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
            "prevalence1",
            "phenotype_model2",
            "phenotype_params2",
            "beta2",
            "beta_sex2",
            "prevalence2",
            "censor_age",
            "gen_censoring",
            "death_scale",
            "death_rho",
            "N_sample",
            "case_ascertainment_ratio",
            "pedigree_dropout_rate",
            "max_degree",
            "estimate_inbreeding",
            "epimight_mi_K",
            "epimight_rubin_level",
            "epimight_kinds",
            "pafgrs_ndegree",
            "iter_reml_trait",
            "iter_reml_ndegree",
            "iter_reml_grm_threshold",
            "iter_reml_phase1_probes",
            "iter_reml_phase2_probes",
            "iter_reml_max_iter",
            "iter_reml_tol",
            "iter_reml_pcg_tol",
            "iter_reml_pcg_max_iter",
            "iter_reml_pc_type",
            "iter_reml_deflation_k",
            "iter_reml_trace_method",
            "iter_reml_hutchpp_sketch_size",
            "iter_reml_compute_logdet",
            "iter_reml_slq_lanczos_steps",
            "iter_reml_slq_probes",
            "iter_reml_emit_probe_traces",
            "grm_n_pcs",
            "grm_min_kinship",
            "export_grm_threshold",
            "export_pair_list_min_kinship",
            "export_pgs_r2",
        }
        assert set(flat.keys()) == expected_keys

    def test_default_yaml_values_match(self):
        import yaml

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
        assert flat["prevalence1"] == 0.10
        assert flat["censor_age"] == 80
        assert flat["death_scale"] == 164
        assert flat["epimight_mi_K"] == 20
        assert flat["pafgrs_ndegree"] == 2
