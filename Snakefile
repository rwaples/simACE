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
    _epimight_mem,
    _epimight_runtime,
    _PHENOTYPE_BASENAMES,
    _SIMPLE_LTM_BASENAMES,
    _VALIDATION_BASENAMES,
    load_folder_configs,
)

load_folder_configs(config)

PLOT_EXT = config["defaults"].get("plot_format", "png")
PHENOTYPE_PLOTS = plot_filenames(_PHENOTYPE_BASENAMES, PLOT_EXT)
SIMPLE_LTM_PLOTS = plot_filenames(_SIMPLE_LTM_BASENAMES, PLOT_EXT)
VALIDATION_PLOTS = plot_filenames(_VALIDATION_BASENAMES, PLOT_EXT)


wildcard_constraints:
    folder="[a-zA-Z0-9_]+",
    scenario="[a-zA-Z0-9_]+",
    rep="\\d+",
    kind="[a-zA-Z0-9]+",


# ── sim_ace rules ──────────────────────────────────────────────────────────
include: "workflow/rules/sim_ace/targets.smk"
include: "workflow/rules/sim_ace/simulate.smk"
include: "workflow/rules/sim_ace/dropout.smk"
include: "workflow/rules/sim_ace/phenotype.smk"
include: "workflow/rules/sim_ace/sample.smk"
include: "workflow/rules/sim_ace/validate.smk"
include: "workflow/rules/sim_ace/stats.smk"
include: "workflow/rules/sim_ace/utils.smk"

# ── fit_ace rules ──────────────────────────────────────────────────────────
include: "workflow/rules/fit_ace/simple_ltm.smk"
include: "workflow/rules/fit_ace/epimight.smk"
include: "workflow/rules/fit_ace/epimight_bias.smk"
include: "workflow/rules/fit_ace/pafgrs.smk"
include: "workflow/rules/fit_ace/grm.smk"
include: "workflow/rules/fit_ace/iter_reml.smk"
include: "workflow/rules/fit_ace/export.smk"
