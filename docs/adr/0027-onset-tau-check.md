# ADR 0027: Score onset τ against its realized expectation

## Status

Accepted and implemented 2026-10-17 for issue #29, decided in a design interview, revised
twice the same day after review, and moved from the cohort stage to an
on-demand `simace validate` during implementation; plan in
`plans/onset-tau-check-v3.md` (session-local draft) with the later decisions
recorded here. Accepted 2026-10-17: the calibration in `config/onset_tau.yaml`
(5 scenarios × 100 reps; four baselines, all three standardization modes,
assortative mating, a sex effect) passed every one of the 161 prespecified
tests of `scripts/onset_tau_calibration.py` at a familywise 5%.

**Status note (2026-10-17).** [ADR 0028](0028-analyze-validate-boundary.md)
amends §1 and §8. A check goes to `simace validate` because it is a
calibrated test, not because of its cost. Onset τ keeps its cells, target, SE
and per-rep Bonferroni bound, but the bound now feeds a verdict across reps
(the acceptance tests below, run by `simace validate` itself) and no longer
sets a per-rep pass.

- Per scenario × cell: mean z from −0.287 to +0.187, SD from 0.876 to 1.167,
  every cell scored in every rep.
- Pooled per cell type over 500 reps: mean z from −0.056 to +0.064, SD from
  0.970 to 1.027.
- Reps with a failing cell: 20 of 500 (4.0%).

A first run spaced the scenario seeds 10 apart. Since a rep's seed is
`seed + rep − 1`, scenarios shared pedigrees and draws, and the pooled
tests, which count those draws more than once, missed. The seeds are now
1000 apart.

## Context

The `frailty` phenotype model draws `E ~ Exp(1)` per person and trait and sets
the raw onset `T = H0⁻¹(E · e^{-η})`, where the linear predictor is
`η = b·(L − m) + beta_sex·sex` (`FrailtyModel.simulate`,
`standardize_beta`). No check covered this time-to-event layer: the
tetrachoric and liability checks validate affected status and liability.

Kendall's τ is unchanged by an increasing transform of either margin, so onset
τ does not depend on the baseline hazard. Given η, onsets order as the
independent exponentials `V_i = E_i / λ_i` with rates `λ_i = e^{η_i}`. Every
comparison of two entries therefore has a closed-form expectation given η, by
the Plackett–Luce probabilities of exponential orderings. For two relationship
pairs `(x, y)` and `(x', y')` with four distinct members it reduces to

    E[sign(T_x − T_x') · sign(T_y − T_y') | η] = tanh((η_x − η_x')/2) · tanh((η_y − η_y')/2).

A comparison in which the two entries share a person does not reduce to this
form. With equal η, `E[sign(T_a − T_c) · sign(T_b − T_a)] = −1/3`, not 0. A
complete sibship of 15 full sibs has observed τ = −1/24 for every ordering of
their onsets.

Averaging the four-person form over a bivariate normal η with standard
deviations `s1, s2` and correlation `r` gives the parametric oracle the issue
asked for, `tau_oracle(s1, s2, r) = E[tanh(s1·X/2) · tanh(s2·Y/2)]` with
`(X, Y) ~ N(0, 2·[[1, r], [r, 1]])`. Its limits are 0 at β = 0 and
`(2/π)·arcsin(r)` as β → ∞. Monte Carlo agrees with it to 3.5e-5. It is exact
only when η is bivariate normal with one `r` and one scale across a cell's
pairs. In a probe on 200k-founder pedigrees it differed from the realized
expectation by up to 0.6 of the conditional SD; at 1M founders by up to
0.24. `beta_sex ≠ 0` makes η a mixture.

simACE validation tolerances have so far been gross-error floors, such as
`max(0.05, 4·SE)`. ADR 0026 left a multiplicity rule for later.

## Decision

1. **Where.** Onset τ is an on-demand check, not part of `simace run`.
   `simace validate <scenario|folder>` runs it on complete reps and writes
   each rep's `validation.yaml`; `simace gather` and the atlas read that file
   when it is present. A rep stores no raw onsets for the phenotyped
   population, so the check regenerates them. The phenotype stage is
   deterministic in the rep seed (ADR 0021), so `run_phenotype` on
   `pedigree.parquet` with the rep's seed and phenotype settings reproduces
   the cohort stage's onsets, in float64. A test checks this against
   `cohort.parquet`. The check covers the whole phenotyped population before
   ascertainment, so a case-weighted draw cannot bias it. A recompute of a rep
   deletes its `validation.yaml`, so the file never describes outputs that
   have been replaced.

   `simace validate` holds a small registry of on-demand check families
   (`simace.analysis.validate.on_demand`), of which onset τ is the first. It
   is the home for any check that costs too much for every rep or needs
   outcomes a rep does not store.
2. **Cells.** For each trait whose model is `frailty`, one cell per
   relationship type `MZ`, `FS`, `MHS`, `PHS`, `MO`, `FO`, over the relationship
   pairs with both members phenotyped. When both traits are frailty, one more
   within-person cell compares `t1` with `t2`. A symmetric type counts both
   orientations of each pair as entries.
3. **Statistic.** Observed onset τ is Kendall's τ-a over all pairs of entries,
   with ties counting 0. A symmetric entry's comparison with its own mirror
   is excluded: it contributes −1 when the pair's onsets differ and 0 when
   they are tied, so exactly the untied mirror comparisons are removed.
4. **Target.** The realized onset-τ expectation: the exact expectation, given
   η, of every comparison the statistic counts. Comparisons between entries
   with four distinct members use the tanh product. For cells of at most
   2,000 entries they are summed directly. Larger cells use a second-order
   Taylor expansion over a B × B grid of the entries' `(η_x, η_y)`, built from
   per-bin offset moments. Every term's remainder is then at most
   `0.1796·h³` (from the bounds 1/2, 1/(3√3) and 1/4 on the first three
   derivatives of `tanh(t/2)`). That bounds every row expectation and the
   target, and the report carries the bound beside the value. B is the
   smallest power of two that holds the bound to 0.01·SE. Comparisons that
   share a person are enumerated through per-person incidence lists (work
   `Σ_p deg_p²`, Θ(k³) for a complete sibship of k) and given their
   Plackett–Luce expectation. There is no Monte Carlo, so the target is
   deterministic. η comes from the phenotype code's own `standardize_beta`
   under the trait's resolved standardization mode. Nothing here assumes a
   distribution for η, so `beta_sex`, all three standardization modes,
   assortative mating and C transmission need no restriction.
   `tau_oracle(s1, s2, r)` is implemented and unit-tested, and the atlas
   plots it as the τ–r curve, but it is information only.
5. **SE.** Given η, only the raw-onset draws are random, and they are
   independent across people. Each entry's influence is
   `H_e = (2/N)·(c̄_e − E[c̄_e | η])`, where `c̄_e` is its observed mean
   concordance against all other entries (one O(N log N) pass). Influences are
   clustered by connected component of the cell's own pair graph:
   `Var = Σ_c (Σ_{e∈c} H_e)²`. Distinct components share no person, so this
   estimator is non-negative and, to first order, unbiased. A single
   component gives `Σ_e H_e = 2(τ̂ − target)`, which is 0 under the exact
   target. A person-sharing (dyadic) sandwich was rejected because it goes
   negative on dense cells.
6. **Scoring gates.** A cell is scored when all four of these hold:
   - its trait is frailty;
   - the effective number of components, `n_eff = (Σ_c w_c)² / Σ_c w_c²` with
     `w_c` the entries in component c, is at least 100 (100 persons for
     within-person);
   - its shared-person workload is at most 5e8;
   - the share of comparisons tied between distinct people (both clips and
     any other equal values) plus the numerical bound is at most 0.05·SE.

   In simulations with up to 1,500 components, cells with n_eff of 72–1,097
   covered at about the nominal rate. Cells dominated by one component (n_eff
   1.4–26) became conservative, never anti-conservative. Any cell that fails
   a gate is reported as information, with the reason.
7. **Entry cap.** A cell over 100,000 entries is scored on a seeded random
   subset of whole components, about 100,000 entries in all. Given η,
   components are independent, so the target and SE stay exact for the
   kept set. When a category is expected to exceed the cap, its pairs are
   extracted only within a seeded sample of family units: the children of
   one mother for `FS`, `MZ` and `MHS`, of one father for `PHS`, and one
   maternal or paternal lineage for `MO` and `FO`. Each unit holds whole
   components, and pedigree-graph still classifies the pairs, on the
   subgraph of the sampled units' rows. The report records each cell's
   sampled fraction. The cap was chosen on cost (decided 2026-10-17), while
   the check still ran on every rep: the exact all-pairs check took about
   7.5 s at `bench500K` shape, with a floor of about 4 s single-threaded. At
   100,000 entries a single rep detects a bias of about 3 SE ≈ 0.006 in
   full-sib onset τ (τ ≈ 0.067).
8. **Tolerance.** A scored cell passes when `|observed − target| ≤ z·SE`, with
   `z = Φ⁻¹(1 − 0.025/K)` and `K` the number of cells scored in that rep:
   2.89 at K = 13. For an ideal normal statistic, a shift of 0.05·SE raises
   the per-cell rate at z = 2.89 from 0.385% to 0.389%. A correct simulation
   therefore fails about 5% of reps, up to that shift and to error in the SE
   estimate.

## Rejected alternatives

- **The parametric oracle as the scored target**, with pg-phenotype's
  MRCA-or-shared-member sandwich (its ADR 0013) applied by the delta method.
  This is the issue's design. It needs normal η, and the probe gap is up to
  0.6 SD at 200k founders.
- **Excluding comparisons that share a person.** The statistic would no
  longer be standard Kendall τ, and the exact expectation is cheap.
- **A person-sharing sandwich**, `Σ_p (Σ_{u∋p} H_u)² − Σ_u H_u²`. It was
  negative in all 200 onset orderings of a 15-sib sibship.
- **MRCA clusters.** Given η, pairs that share an ancestor but no member are
  independent, so the cell's own components are enough.
- **Monte Carlo for the target and SE.** A seed-dependent result with added
  variance, and a split-sample variance that can go negative.
- **Analyze on the analysis sample.** The check would have to be skipped
  under a case-weighted draw.
- **The cohort stage, on every rep.** Built and measured first (see
  Consequences); it cost every simulation for a check most runs do not need.
- **A config switch.** A new config key changes every existing rep's
  recorded config, so `simace run` would refuse them until `--force`.
- **A floor tolerance**, such as `max(0.05, 4·SE)`. Full-sib onset τ is about
  0.067 under the default config.
- **Modelling the clip in the target.** It needs `H0(1e6)` for each baseline;
  heavily clipped cells become information instead.

## Consequences

- `simace run` costs nothing more. The check was first built into the
  cohort stage; measured there (medians of three interleaved runs), it raised
  the rep wall from 6.96 s to 9.33 s at `bench500K` and from 36.87 s to
  41.48 s at `bench5M`, with peak memory unchanged. That was too much to pay
  on every simulation, so it moved to `simace validate` (decided 2026-10-17).
  The 100,000-entry cap and the family-unit extraction stay. They bound the
  check's own time and memory: about 2 s for `bench500K` and 3.5 s for
  `bench5M` per rep, single-threaded.
- Building the pair lists materializes pair blocks, which pedigree-graph
  documents at a construction peak of about 2.3 times the result. The
  family-unit extraction keeps that to the sampled units at scale.
- The lognormal inverter puts onsets at the `1e6` ceiling when `exp(−target)`
  underflows. At β = 2 that was 1.2% of people, which turns its cells
  informational through the tie gate.
- Every cell, scored or not, is kept in `validation.yaml` with its
  statistics, status and reason. `report.yaml` and Analyze are unchanged, and
  a rep that has not been validated simply has no `validation.yaml`; the
  atlas shows a placeholder for it and `gather` leaves its columns empty.
- Regeneration needs the phenotype code that built the rep. A rep validated
  by a later simace checks that version's phenotype stage on the rep's
  pedigree, which is still a valid check of the frailty model, but its onsets
  can differ from the rep's `cohort.parquet` if the phenotype stage changed.
- The other phenotype models are out of scope. `cure_frailty` gives onsets
  only to cases, and whether ADuLT `ltm` onset is a deterministic function of
  liability is unverified.
- Observed τ from censored onsets is a separate question.
