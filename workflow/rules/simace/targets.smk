rule all:
    input:
        [
            f"results/{get_folder(config, s)}/{s}/scenario.done"
            for s in config["scenarios"]
        ],


rule folder:
    """Build all scenario outputs for a single folder grouping."""
    input:
        lambda w: [
            f"results/{w.folder}/{s}/scenario.done"
            for s in get_scenarios_for_folder(config, w.folder)
        ],
    output:
        touch("results/{folder}/folder.done"),


rule scenario:
    """Build all sim/validation/plot outputs for a single scenario."""
    input:
        lambda w: get_scenario_sim_outputs(config, w.scenario, plot_ext=PLOT_EXT),
        lambda w: f"results/{get_folder(config, w.scenario)}/validation_summary.tsv",
        lambda w: f"results/{get_folder(config, w.scenario)}/plots/{VALIDATION_PLOTS[0]}",
    output:
        touch("results/{folder}/{scenario}/scenario.done"),


rule simulate_scenario:
    """Run pedigree simulation only for a single scenario."""
    input:
        lambda w: [
            f"results/{w.folder}/{w.scenario}/rep{r}/pedigree.parquet"
            for r in range(1, get_param(config, w.scenario, "replicates") + 1)
        ],
    output:
        touch("results/{folder}/{scenario}/simulate.done"),


rule phenotype_scenario:
    """Run simulation + phenotyping for a single scenario."""
    input:
        lambda w: [
            f"results/{w.folder}/{w.scenario}/rep{r}/{f}"
            for r in range(1, get_param(config, w.scenario, "replicates") + 1)
            for f in [
                "phenotype.parquet",
                "phenotype.simple_ltm.parquet",
            ]
        ],
    output:
        touch("results/{folder}/{scenario}/phenotype.done"),


rule validate_scenario:
    """Run simulation + validation for a single scenario."""
    input:
        lambda w: [
            f"results/{w.folder}/{w.scenario}/rep{r}/validation.yaml"
            for r in range(1, get_param(config, w.scenario, "replicates") + 1)
        ],
        lambda w: f"results/{get_folder(config, w.scenario)}/validation_summary.tsv",
        lambda w: f"results/{get_folder(config, w.scenario)}/plots/{VALIDATION_PLOTS[0]}",
    output:
        touch("results/{folder}/{scenario}/validate.done"),


rule stats_scenario:
    """Run phenotyping + stats + plots for a single scenario."""
    input:
        lambda w: [
            f"results/{w.folder}/{w.scenario}/rep{r}/phenotype_stats.yaml"
            for r in range(1, get_param(config, w.scenario, "replicates") + 1)
        ],
        lambda w: [
            f"results/{w.folder}/{w.scenario}/plots/{plot}"
            for plot in PHENOTYPE_PLOTS
        ],
    output:
        touch("results/{folder}/{scenario}/stats.done"),
