configfile: "config/config.yaml"

from workflow.common import (
    get_param, get_folder, get_scenarios_for_folder,
    get_all_folders, get_folder_validations,
    get_scenario_sim_outputs, plot_filenames, _scale_mem, _scale_runtime,
    _PHENOTYPE_BASENAMES, _SIMPLE_LTM_BASENAMES, _VALIDATION_BASENAMES,
)

PLOT_EXT = config["defaults"].get("plot_format", "png")
PHENOTYPE_PLOTS = plot_filenames(_PHENOTYPE_BASENAMES, PLOT_EXT)
SIMPLE_LTM_PLOTS = plot_filenames(_SIMPLE_LTM_BASENAMES, PLOT_EXT)
VALIDATION_PLOTS = plot_filenames(_VALIDATION_BASENAMES, PLOT_EXT)

wildcard_constraints:
    folder="[a-zA-Z0-9_]+",
    scenario="[a-zA-Z0-9_]+",
    rep="\\d+",
    kind="[a-zA-Z0-9]+"

include: "workflow/rules/targets.smk"
include: "workflow/rules/simulate.smk"
include: "workflow/rules/dropout.smk"
include: "workflow/rules/phenotype.smk"
include: "workflow/rules/sample.smk"
include: "workflow/rules/validate.smk"
include: "workflow/rules/stats.smk"
include: "workflow/rules/epimight.smk"
include: "workflow/rules/epimight_bias.smk"
include: "workflow/rules/epimight_single.smk"
