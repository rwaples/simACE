# Unify dropout and subsampling into a single ascertainment stage after phenotype

- **Status:** accepted
- **Implemented:** [`af138c8`](../../../../commit/af138c8) (simACE), [`1c3065a`](https://github.com/rwaples/fitACE/commit/1c3065a) (fitACE)

Previously simACE had two separate data-reduction stages: pedigree dropout ran *before* phenotype (deleting individuals from the pedigree to model registry incompleteness) and subsampling ran *after* censor (drawing a study sample with case-ascertainment weighting). The split was a recurring source of confusion because both mechanisms reduce the analysis dataset and the semantics of "removed" differed subtly between them.

We've collapsed the two into one **ascertainment** stage that runs after phenotype/censor. It carries two independent knobs — a `dropout_rate` (uniform random removal) and a `case_ascertainment_ratio` (trait-weighted selection) — plus a target `N_sample`. The trade-off: we lose the "registry never observed this individual" semantic that pre-phenotype dropout carried (an individual that didn't exist at observation time) in favour of "this individual is excluded from the analysis sample" (an individual we know about but don't use). For simulation output the two are indistinguishable; for the conceptual model of what's being simulated, the new framing is "study enrollment selectively recovers from a known population", which we judged the more useful framing and the simpler pipeline graph.
