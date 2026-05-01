"""Cross-scenario comparison plots that feed the Examples docs pages.

Each rule collects validation.yaml outputs across a hardcoded scenario group
and writes a comparison PNG directly into ``docs/images/examples/<topic>/``.
The mkdocs pages reference those image paths, so rebuilding these rules is
what keeps the docs site in sync with the latest simulation state.
"""

AM_HERITABILITY_SCENARIOS = ["am_none", "am_weak", "am_strong"]
AM_HERITABILITY_LABELS = ["no AM", "weak AM (0.2)", "strong AM (0.4)"]

OBSERVED_VS_LIABILITY_SCENARIOS = [
    "model_ltm",
    "model_cure_frailty_ln",
    "model_frailty_wb",
]
OBSERVED_VS_LIABILITY_LABELS = [
    "LTM",
    "Cure-frailty (lognormal)",
    "Frailty (Weibull)",
]

INCREASING_E_TRAJECTORIES = [
    "e_flat",
    "e_rise_mild",
    "e_rise_steep",
    "e_fall_steep",
]
INCREASING_E_LABELS = [
    "E flat at 0.5",
    "E rising 0.5→0.6",
    "E rising 0.5→0.7",
    "E falling 0.5→0.3",
]
INCREASING_E_STD_SCENARIOS = [f"{traj}_std" for traj in INCREASING_E_TRAJECTORIES]
INCREASING_E_NOSTD_SCENARIOS = [f"{traj}_nostd" for traj in INCREASING_E_TRAJECTORIES]
INCREASING_E_PERGEN_SCENARIOS = [f"{traj}_pergen" for traj in INCREASING_E_TRAJECTORIES]


def _increasing_e_per_gen_E(scen):
    """Resolve per-gen E1 schedule from the config dict for a scenario."""
    e_dict = get_param(config, scen, "E1")
    if isinstance(e_dict, dict):
        sorted_keys = sorted(e_dict)
        return [float(e_dict[k]) for k in sorted_keys]
    # Constant fallback (shouldn't occur for these scenarios but stay robust).
    return [float(e_dict)] * 10


rule compare_am_heritability:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/validation.yaml"
            for scen in AM_HERITABILITY_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/am/vA_trajectory.png",
    log:
        "logs/examples/compare_am_heritability.log",
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    params:
        scenarios=AM_HERITABILITY_SCENARIOS,
        labels=AM_HERITABILITY_LABELS,
        expected_A=lambda w: get_param(config, AM_HERITABILITY_SCENARIOS[0], "A1"),
        expected_C=lambda w: get_param(config, AM_HERITABILITY_SCENARIOS[0], "C1"),
        expected_E=lambda w: get_param(config, AM_HERITABILITY_SCENARIOS[0], "E1"),
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates") for scen in AM_HERITABILITY_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_am_heritability.py"


rule compare_am_component_distributions:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/pedigree.parquet"
            for scen in AM_HERITABILITY_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/am/component_distributions.png",
    log:
        "logs/examples/compare_am_component_distributions.log",
    threads: 1
    resources:
        mem_mb=4000,
        runtime=5,
    params:
        labels=AM_HERITABILITY_LABELS,
        min_generation=lambda w: max(
            1, get_param(config, AM_HERITABILITY_SCENARIOS[0], "G_pheno") // 2
        ),
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates") for scen in AM_HERITABILITY_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_am_component_distributions.py"


rule compare_am_correlations_by_relclass:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/pedigree.parquet"
            for scen in AM_HERITABILITY_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/am/corr_by_relclass.png",
    log:
        "logs/examples/compare_am_correlations_by_relclass.log",
    threads: 1
    resources:
        mem_mb=4000,
        runtime=5,
    params:
        labels=AM_HERITABILITY_LABELS,
        expected_A=lambda w: get_param(config, AM_HERITABILITY_SCENARIOS[0], "A1"),
        expected_C=lambda w: get_param(config, AM_HERITABILITY_SCENARIOS[0], "C1"),
        min_generation=lambda w: max(
            1, get_param(config, AM_HERITABILITY_SCENARIOS[0], "G_pheno") // 2
        ),
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates") for scen in AM_HERITABILITY_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_am_correlations.py"


rule compare_am_sib_liability:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/pedigree.parquet"
            for scen in AM_HERITABILITY_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/am/sib_liability_scatter.png",
    log:
        "logs/examples/compare_am_sib_liability.log",
    threads: 1
    resources:
        mem_mb=4000,
        runtime=5,
    params:
        labels=AM_HERITABILITY_LABELS,
        min_generation=lambda w: max(
            1, get_param(config, AM_HERITABILITY_SCENARIOS[0], "G_pheno") // 2
        ),
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates") for scen in AM_HERITABILITY_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_am_sib_liability.py"


rule compare_am_naive_estimators:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/pedigree.parquet"
            for scen in AM_HERITABILITY_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/am/naive_estimators.png",
    log:
        "logs/examples/compare_am_naive_estimators.log",
    threads: 1
    resources:
        mem_mb=4000,
        runtime=5,
    params:
        labels=AM_HERITABILITY_LABELS,
        input_h2=lambda w: (
            get_param(config, AM_HERITABILITY_SCENARIOS[0], "A1")
            / (
                get_param(config, AM_HERITABILITY_SCENARIOS[0], "A1")
                + get_param(config, AM_HERITABILITY_SCENARIOS[0], "C1")
                + get_param(config, AM_HERITABILITY_SCENARIOS[0], "E1")
            )
        ),
        min_generation=lambda w: max(
            1, get_param(config, AM_HERITABILITY_SCENARIOS[0], "G_pheno") // 2
        ),
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates") for scen in AM_HERITABILITY_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_am_naive_estimators.py"


rule compare_observed_vs_liability_h2:
    input:
        pedigree=lambda w: [
            f"results/examples/{scen}/rep{rep}/pedigree.parquet"
            for scen in OBSERVED_VS_LIABILITY_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
        phenotype_stats=lambda w: [
            f"results/examples/{scen}/rep{rep}/phenotype_stats.yaml"
            for scen in OBSERVED_VS_LIABILITY_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/models/observed_vs_liability.png",
    log:
        "logs/examples/compare_observed_vs_liability_h2.log",
    threads: 1
    resources:
        mem_mb=4000,
        runtime=5,
    params:
        labels=OBSERVED_VS_LIABILITY_LABELS,
        input_h2=lambda w: (
            get_param(config, OBSERVED_VS_LIABILITY_SCENARIOS[0], "A1")
            / (
                get_param(config, OBSERVED_VS_LIABILITY_SCENARIOS[0], "A1")
                + get_param(config, OBSERVED_VS_LIABILITY_SCENARIOS[0], "C1")
                + get_param(config, OBSERVED_VS_LIABILITY_SCENARIOS[0], "E1")
            )
        ),
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates")
            for scen in OBSERVED_VS_LIABILITY_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_observed_vs_liability_h2.py"


rule compare_increasing_e_trajectory:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/validation.yaml"
            for scen in INCREASING_E_STD_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/increasing_e/realized_components_trajectory.png",
    log:
        "logs/examples/compare_increasing_e_trajectory.log",
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    params:
        labels=INCREASING_E_LABELS,
        expected_A=lambda w: get_param(config, INCREASING_E_STD_SCENARIOS[0], "A1"),
        expected_C=lambda w: get_param(config, INCREASING_E_STD_SCENARIOS[0], "C1"),
        expected_E=lambda w: _increasing_e_per_gen_E(INCREASING_E_STD_SCENARIOS[0]),
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates")
            for scen in INCREASING_E_STD_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_increasing_e_trajectory.py"


rule compare_increasing_e_cohort_fs:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/pedigree.parquet"
            for scen in INCREASING_E_STD_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/increasing_e/fs_corr_by_gen.png",
    log:
        "logs/examples/compare_increasing_e_cohort_fs.log",
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    params:
        labels=INCREASING_E_LABELS,
        expected_A=lambda w: get_param(config, INCREASING_E_STD_SCENARIOS[0], "A1"),
        expected_C=lambda w: get_param(config, INCREASING_E_STD_SCENARIOS[0], "C1"),
        min_generation=1,
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates")
            for scen in INCREASING_E_STD_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_increasing_e_cohort_fs.py"


rule compare_increasing_e_cohort_falconer:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/pedigree.parquet"
            for scen in INCREASING_E_STD_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/increasing_e/cohort_falconer.png",
    log:
        "logs/examples/compare_increasing_e_cohort_falconer.log",
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    params:
        labels=INCREASING_E_LABELS,
        min_generation=1,
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates")
            for scen in INCREASING_E_STD_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_increasing_e_cohort_falconer.py"


rule compare_increasing_e_components_by_gen:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/pedigree.parquet"
            for scen in INCREASING_E_STD_SCENARIOS
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/increasing_e/components_by_gen.png",
    log:
        "logs/examples/compare_increasing_e_components_by_gen.log",
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    params:
        labels=INCREASING_E_LABELS,
        show_generations=[1, 5, 9],
        reps_per_scenario=lambda w: [
            get_param(config, scen, "replicates")
            for scen in INCREASING_E_STD_SCENARIOS
        ],
    script:
        "../../scripts/simace/compare_increasing_e_components_by_gen.py"


rule compare_increasing_e_prevalence:
    input:
        lambda w: [
            f"results/examples/{scen}/rep{rep}/phenotype_stats.yaml"
            for traj in INCREASING_E_TRAJECTORIES
            for scen in (f"{traj}_std", f"{traj}_nostd", f"{traj}_pergen")
            for rep in range(1, get_param(config, scen, "replicates") + 1)
        ],
    output:
        "docs/images/examples/increasing_e/prevalence_drift.png",
    log:
        "logs/examples/compare_increasing_e_prevalence.log",
    threads: 1
    resources:
        mem_mb=2000,
        runtime=5,
    params:
        labels=INCREASING_E_LABELS,
        target_prevalence=0.1,
        reps_per_trajectory=lambda w: [
            get_param(config, f"{traj}_std", "replicates")
            for traj in INCREASING_E_TRAJECTORIES
        ],
    script:
        "../../scripts/simace/compare_increasing_e_prevalence.py"


rule examples_all:
    """Aggregate target: build every comparison plot used by the docs/examples/ pages."""
    input:
        "docs/images/examples/am/vA_trajectory.png",
        "docs/images/examples/am/component_distributions.png",
        "docs/images/examples/am/corr_by_relclass.png",
        "docs/images/examples/am/sib_liability_scatter.png",
        "docs/images/examples/am/naive_estimators.png",
        "docs/images/examples/models/observed_vs_liability.png",
        "docs/images/examples/increasing_e/realized_components_trajectory.png",
        "docs/images/examples/increasing_e/components_by_gen.png",
        "docs/images/examples/increasing_e/fs_corr_by_gen.png",
        "docs/images/examples/increasing_e/cohort_falconer.png",
        "docs/images/examples/increasing_e/prevalence_drift.png",
