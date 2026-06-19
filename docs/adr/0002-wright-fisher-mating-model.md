# Add a sex-structured Wright-Fisher mating model alongside the standard model

- **Status:** accepted
- **Implemented:** dev branch, 2026-05

## Context

simACE shipped with a single mating algorithm: a sex-structured, ZTP-fecundity, paired model with optional assortative mating. That's a defensible *family-study* generative model, but it's not the model that population-genetics theory (drift, inbreeding, $N_e$, the coalescent) is built against. The glossary already pinned $N_e$ to "the size of an idealized Wright-Fisher population that would produce the same rate of drift / inbreeding / mean-kinship accumulation as the simulated pedigree" (`CONTEXT.md`), yet the simulator had no way to actually produce one. Three motivations for adding WF:

1. Validate the $N_e$ estimators against the reference they're defined against — under an idealized WF, $N_e \to N$ and every estimator should recover that.
2. Provide a null / theoretical-baseline mating model so we can reason about how more realistic mating structures shift kinship distributions, relative-pair frequencies, and downstream ACE estimates.
3. Support end-to-end A+C+E simulations under a WF mating structure (full pipeline output, not just a pedigree-validation tool).

## Decision

Add `pedigree.mating_model` as a config key with two values, `standard` (default, preserves all existing behavior) and `wright_fisher`. The WF flavor is **sex-structured idealized Wright-Fisher**:

- Two sexes are retained; sex of offspring is a fresh Bernoulli(0.5), unchanged from the standard path.
- For each of the $N$ offspring next generation, one mother is drawn uniformly at random from the prior generation's females and one father uniformly from the males, both **with replacement**. Multinomial offspring counts per parent.
- No persistent mating-pair structure: each offspring's two parents are an independent draw event.
- Selfing is impossible by construction (parents must be opposite-sex).
- Households are still grouped by mother; $C$ is drawn fresh per household per generation, identical to the standard path. End-to-end A+C+E continues to work.
- No MZ twins: textbook Wright-Fisher has no notion of a single fertilization event from which a twin pair could descend, so the WF code path produces no twins. The `twin` column is `-1` for every WF offspring.
- `mating_lambda`, `p_mztwin`, `assort1`, `assort2`, `assort_matrix` are no-ops at runtime under WF. Inherited defaults are silently ignored. Explicit scenario-level overrides to non-no-op values are rejected at config-load with a clear message.

`params.yaml` records the chosen `mating_model` faithfully. Downstream consumers (`validate_twins`, `theoretical_expectations`, `gather`, `plot_validation`, `plot_effective_size`) branch on this rather than on normalized inputs — see *Trade-offs* below for why.

## Why this flavor

The single most consequential design choice was sex-structured vs textbook hermaphroditic WF.

- **Textbook hermaphroditic WF** is the canonical pop-gen reference: each offspring picks two parents (with replacement, selfing allowed) from a single sexless pool. But it has no mother and no household, so $C$ has no structural anchor — running A+C+E under hermaphroditic WF requires either dropping $C$ entirely or inventing an artificial "designate the first-drawn parent as the mother" rule that has no analogue in WF theory. Neither was acceptable given motivation (3).
- **Sex-structured WF** keeps both the mother (so households and $C$ still work) and a meaningful binary `sex` column (so phenotype models with sex-specific prevalence still work), while still being close enough to canonical WF that $N_e \to N$ holds under random mating with equal sex ratio. The cost is honest: this is "WF-like with two sexes", not the strictly textbook model. We document that explicitly in `CONTEXT.md` and here.

For the regression-based $N_e$ estimators (`ne_inbreeding`, `ne_coancestry`, `ne_caballero_toro`), the same Jensen-bias regime gate applies as in the standard path. Under WF the gate reduces to $G_{\text{ped}}^2 \geq 120$, i.e. $G_{\text{ped}} \geq 11$. Below that, the *expected* values are reported as `None` and the validator passes vacuously; the *observed* estimators are still computed and reported. This means the simplest WF validation requires either large $G_{\text{ped}}$ or accepting that the regression-based estimators won't have a theoretical reference to compare against.

## Trade-offs and alternatives rejected

- **Strict-validation vs silent-ignore for incompatible knobs.** Strict (errors at config-load when a WF scenario explicitly sets `mating_lambda`, non-zero `p_mztwin`/`assort*`, or non-null `assort_matrix`) won. Silent-ignore would let a user believe they're simulating assortative-WF when they're not. The validation rule operates on *scenario-level* overrides only (not the merged config), so inherited defaults from `_default.yaml` flow through unchanged — a WF scenario file can be as terse as a `mating_model: wright_fisher` line plus its variance components.
- **Post-hoc MZ twin assignment under WF.** Considered: after offspring are drawn, group by mother and apply `p_mztwin` within each maternal group of size ≥ 2. Rejected because MZ twins biologically come from a single fertilization event, and a per-mother post-hoc rule would mark maternal-half-siblings as MZ — a different physical model dressed up as MZ twins.
- **Configurable sex ratio / exact $N/2$ split.** The N_e correction from binomial sex-ratio fluctuation under fresh Bernoulli(0.5) is sub-1% at $N \geq 10\text{K}$. Adding a knob (or hard-coding exact $N/2$) buys a fractional-percent improvement at the cost of either config-surface bloat or divergence from how the standard path assigns sex. Rejected.
- **Two WF variants (`strict_wf` hermaphroditic + `wright_fisher` sex-structured).** Doubles surface area, requires twice the test coverage, fragments the "use one model, get one answer" mental model. Rejected; one model only.
- **Normalize standard-only knobs to no-op values inside `emit_params`.** Considered: under WF, write `p_mztwin: 0` and `mating_lambda: null` into `params.yaml` so existing downstream consumers don't need to branch. Rejected because it makes `params.yaml` no longer a faithful echo of the scenario config (`simace/simulation/emit_params.py` opens with the docstring "echo of the scenario config — no computation"). The chosen alternative — record everything faithfully and have downstream consumers branch on `mating_model` — preserves provenance at the cost of N+1 small branches in the `validate` package (`simace/analysis/validate/`), `simace/analysis/stats/effective_size.py`, `gather.py`, and two plotting modules.
- **$C$ forbidden under WF (treat WF as A+E only).** Theoretically clean but contradicts motivation (3). Rejected because sex-structured WF naturally preserves households-per-mother, which gives $C$ a real structural anchor.
