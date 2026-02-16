configfile: "config/config.yaml"


def get_param(scenario, param):
    """Get parameter value, falling back to defaults if not specified in scenario."""
    scenario_config = config["scenarios"].get(scenario, {})
    if param in scenario_config:
        return scenario_config[param]
    return config["defaults"][param]


def get_scenario_analysis_outputs(scenario):
    """Generate all Weibull result paths for a single scenario."""
    n_reps = get_param(scenario, "replicates")
    outputs = []
    for rep in range(1, n_reps + 1):
        for trait in [1, 2]:
            outputs.append(
                f"results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/results.rwe"
            )
    return outputs


rule all:
    input:
        expand("results/analysis/{scenario}/scenario.analyzed", scenario=config["scenarios"].keys())


rule scenario:
    """Build all analysis outputs for a single scenario."""
    input:
        lambda w: get_scenario_analysis_outputs(w.scenario)
    output:
        touch("results/analysis/{scenario}/scenario.analyzed")


rule prepare_weibull:
    input:
        pedigree="results/{scenario}/rep{rep}/pedigree.parquet",
        phenotype="results/{scenario}/rep{rep}/phenotype.weibull.parquet"
    output:
        data="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/data.dat",
        codelist="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/codelist.txt",
        varlist="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/varlist.txt",
        pedigree_ped="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/pedigree.ped",
        weibull_config="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/weibull.txt"
    log:
        "logs/analysis/{scenario}/rep{rep}/weibull/trait{trait}/prepare.log"
    script:
        "scripts/prepare_weibull.py"


rule run_weibull:
    input:
        data="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/data.dat",
        codelist="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/codelist.txt",
        varlist="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/varlist.txt",
        pedigree_ped="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/pedigree.ped",
        weibull_config="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/weibull.txt"
    output:
        rwe="results/analysis/{scenario}/rep{rep}/weibull/trait{trait}/results.rwe"
    log:
        "logs/analysis/{scenario}/rep{rep}/weibull/trait{trait}/run.log"
    shell:
        "cd results/analysis/{wildcards.scenario}/rep{wildcards.rep}/weibull/trait{wildcards.trait} "
        "&& /home/ryanw/Documents/ACE/external/Survival_Kit_executables_Linux/weibull.exe "
        "> {workflow.basedir}/../{log} 2>&1"
