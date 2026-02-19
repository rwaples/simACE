"""Backward-compatible wrapper: snakemake -s workflow/analyze.smk still works."""
configfile: "config/config.yaml"

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(workflow.basedir), ""))

from workflow.common import (
    get_param, get_folder, get_scenarios_for_folder,
    get_all_folders, get_scenario_analysis_outputs,
    PHENOTYPE_PLOTS, THRESHOLD_PLOTS,
)

wildcard_constraints:
    folder="[a-zA-Z0-9_]+",
    scenario="[a-zA-Z0-9_]+",
    rep="\\d+"

include: "rules/targets.smk"
include: "rules/simulate.smk"
include: "rules/phenotype.smk"
include: "rules/analyze.smk"
