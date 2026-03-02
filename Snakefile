configfile: "config/config.yaml"

from workflow.common import (
    get_param, get_folder, get_scenarios_for_folder,
    get_all_folders, get_folder_validations,
    get_scenario_sim_outputs, plot_filenames,
    _PHENOTYPE_BASENAMES, _THRESHOLD_BASENAMES, _VALIDATION_BASENAMES,
)

PLOT_EXT = config["defaults"].get("plot_format", "png")
PHENOTYPE_PLOTS = plot_filenames(_PHENOTYPE_BASENAMES, PLOT_EXT)
THRESHOLD_PLOTS = plot_filenames(_THRESHOLD_BASENAMES, PLOT_EXT)
VALIDATION_PLOTS = plot_filenames(_VALIDATION_BASENAMES, PLOT_EXT)

wildcard_constraints:
    folder="[a-zA-Z0-9_]+",
    scenario="[a-zA-Z0-9_]+",
    rep="\\d+"

include: "workflow/rules/targets.smk"
include: "workflow/rules/simulate.smk"
include: "workflow/rules/phenotype.smk"
include: "workflow/rules/sample.smk"
include: "workflow/rules/validate.smk"
include: "workflow/rules/stats.smk"
