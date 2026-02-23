rule all:
    input:
        [f"results/{get_folder(config, s)}/{s}/scenario.done" for s in config["scenarios"]]


rule simulate_all:
    """Run pedigree simulation only (no phenotyping, validation, or plots)."""
    input:
        [f"results/{get_folder(config, s)}/{s}/rep{r}/pedigree.parquet"
         for s in config["scenarios"]
         for r in range(1, get_param(config, s, "replicates") + 1)]


rule phenotype_all:
    """Run simulation + phenotyping (Weibull and threshold)."""
    input:
        [f"results/{get_folder(config, s)}/{s}/rep{r}/{f}"
         for s in config["scenarios"]
         for r in range(1, get_param(config, s, "replicates") + 1)
         for f in ["phenotype.weibull.parquet", "phenotype.liability_threshold.parquet"]]


rule validate_all:
    """Run simulation + validation + folder summaries."""
    input:
        [f"results/{folder}/validation_summary.tsv"
         for folder in get_all_folders(config)],
        [f"results/{folder}/plots/variance_components.png"
         for folder in get_all_folders(config)]


rule stats_all:
    """Run phenotyping + stats + phenotype/threshold plots."""
    input:
        [f"results/{get_folder(config, s)}/{s}/plots/{plot}"
         for s in config["scenarios"]
         for plot in PHENOTYPE_PLOTS + THRESHOLD_PLOTS]


rule folder:
    """Build all scenario outputs for a single folder grouping."""
    input:
        lambda w: [f"results/{w.folder}/{s}/scenario.done"
                    for s in get_scenarios_for_folder(config, w.folder)]
    output:
        touch("results/{folder}/folder.done")


rule scenario:
    """Build all sim/validation/plot outputs for a single scenario."""
    input:
        lambda w: get_scenario_sim_outputs(config, w.scenario),
        lambda w: f"results/{get_folder(config, w.scenario)}/validation_summary.tsv",
        lambda w: f"results/{get_folder(config, w.scenario)}/plots/variance_components.png",
    output:
        touch("results/{folder}/{scenario}/scenario.done")


