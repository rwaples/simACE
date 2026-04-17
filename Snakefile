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
    _PHENOTYPE_BASENAMES,
    _VALIDATION_BASENAMES,
    load_folder_configs,
)

load_folder_configs(config)

PLOT_EXT = config["defaults"].get("plot_format", "png")
PHENOTYPE_PLOTS = plot_filenames(_PHENOTYPE_BASENAMES, PLOT_EXT)
VALIDATION_PLOTS = plot_filenames(_VALIDATION_BASENAMES, PLOT_EXT)


wildcard_constraints:
    folder="[a-zA-Z0-9_]+",
    scenario="[a-zA-Z0-9_]+",
    rep="\\d+",
    kind="[a-zA-Z0-9]+",


include: "workflow/rules/sim_ace/targets.smk"
include: "workflow/rules/sim_ace/simulate.smk"
include: "workflow/rules/sim_ace/dropout.smk"
include: "workflow/rules/sim_ace/phenotype.smk"
include: "workflow/rules/sim_ace/sample.smk"
include: "workflow/rules/sim_ace/validate.smk"
include: "workflow/rules/sim_ace/stats.smk"
include: "workflow/rules/sim_ace/utils.smk"
