"""Cross-scenario comparison plots that feed the Examples docs pages.

Each rule collects validation.yaml outputs across a hardcoded scenario group
and writes a comparison PNG directly into ``docs/images/examples/<topic>/``.
The mkdocs pages reference those image paths, so rebuilding these rules is
what keeps the docs site in sync with the latest simulation state.
"""

AM_HERITABILITY_SCENARIOS = ["am_none", "am_weak", "am_strong"]
AM_HERITABILITY_LABELS = ["no AM", "weak AM (0.2)", "strong AM (0.4)"]


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


rule examples_all:
    """Aggregate target: build every comparison plot used by the docs/examples/ pages."""
    input:
        "docs/images/examples/am/vA_trajectory.png",
        "docs/images/examples/am/component_distributions.png",
        "docs/images/examples/am/corr_by_relclass.png",
        "docs/images/examples/am/sib_liability_scatter.png",
        "docs/images/examples/am/naive_estimators.png",
