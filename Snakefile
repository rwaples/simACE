configfile: "config/_default.yaml"


from workflow.common import (
    get_param,
    get_folder,
    get_scenarios_for_folder,
    get_all_folders,
    get_folder_validations,
    get_scenario_sim_outputs,
    plot_filenames,
    _scale_mem,
    _scale_runtime,
    phenotype_basenames,
    validation_basenames,
    load_folder_configs,
)

load_folder_configs(config)

PLOT_EXT = config["defaults"].get("plot_format", "png")
PHENOTYPE_PLOTS = plot_filenames(phenotype_basenames(), PLOT_EXT)
VALIDATION_PLOTS = plot_filenames(validation_basenames(), PLOT_EXT)


wildcard_constraints:
    folder="[a-zA-Z0-9_]+",
    scenario="[a-zA-Z0-9_]+",
    rep="\\d+",
    kind="[a-zA-Z0-9]+",


include: "workflow/rules/simace/targets.smk"
include: "workflow/rules/simace/simulate.smk"
include: "workflow/rules/simace/dropout.smk"
include: "workflow/rules/simace/phenotype.smk"
include: "workflow/rules/simace/sample.smk"
include: "workflow/rules/simace/validate.smk"
include: "workflow/rules/simace/stats.smk"
include: "workflow/rules/simace/utils.smk"
include: "workflow/rules/simace/examples.smk"
include: "workflow/rules/simace/tskit_preprocess.smk"
include: "workflow/rules/simace/genotype_drop.smk"
include: "workflow/rules/simace/tstrait_phenotype.smk"
