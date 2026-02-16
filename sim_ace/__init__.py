"""sim_ace - ACE pedigree simulation with A/C/E variance components."""

__version__ = "0.1.0"

from sim_ace.simulate import (
    run_simulation,
    generate_correlated_components,
    mating,
    reproduce,
    add_to_pedigree,
)
from sim_ace.phenotype import simulate_phenotype, age_censor, death_censor
from sim_ace.threshold import apply_threshold
from sim_ace.validate import run_validation
from sim_ace.stats import tetrachoric_corr, tetrachoric_corr_se
