# ADR 0033: Three inbreeding modes scale the Mendelian residual by parental inbreeding

## Status

Proposed 2026-10-24 for issue #76. Decided in a `/grill-with-docs` session on the evidence in
`notes/issue-76-recommendation.md` (random mating, engines, costs) and `notes/issue-76-am.md` (assortative
mating). Not implemented.

## Context

- simACE draws every offspring's **Mendelian residual** with variance A/2, whatever its parents' inbreeding. Under
  the infinitesimal model the variance is (A/2)(1 − F̄), with F̄ the mean of the two parents' **inbreeding
  coefficients**. The constant draw overstates Var(A) once the pedigree is inbred: by 2.9e-2 at N 1k/G 100 under
  random mating, and by 0.17 under assortment 0.9 (the recommendation §3; the AM note §3.1).
- Exact F over the whole simulated ancestry (burn-in included) costs Θ(N²) memory (dense) or work that grows with
  depth (path walks). At 1M/G 20 the fastest exact engine took 68 min. A depth-truncated approximation (xexact)
  costs 11 s at 5M/G 20 on 16 threads, with per-individual RMS ≈ 0.8·2^−L/(2√N) for N ≥ 2k at random mating.
- Under assortative mating a pedigree depends on A, so any change in F changes later pairings. Modes, and engines
  that differ in F's last bits, can then produce different pedigrees from one seed. Every engine is bitwise
  thread-invariant.
- `analysis.estimate_inbreeding` already exists. It estimates F from the recorded pedigree at analysis time and
  does not affect the simulation.

## Decision

1. **Config.** A nested block under `pedigree`:

   ```yaml
   pedigree:
     inbreeding:
       mode: "off"   # "off" | approx | exact (quoted: YAML reads a bare off as false)
       depth: 6      # approx only: generations traced exactly (integer >= 1)
   ```

   `off` is the default and is unchanged: same output bytes, same random stream.
2. **Approx.** xexact: each parent's ancestry traced exactly for `depth` generations, the rest closed with the
   generation's mean kinship. The knob is the integer depth, default 6, not an error tolerance: the RMS law fails at
   small N and under strong AM, and the max error has no law. `depth ≥ G_sim` is allowed and runs the exact mode (its engine and refusal rule), so its F is exact
   by construction. The
   documentation states the RMS law with its range (N ≥ 2k, assortment ≤ 0.6) and that under assortment ≥ 0.75
   depth 6's Var(A) margin over off shrinks (107–170× at 0.9, G ≤ 50, N 1k); depth 8 or exact restores it. No
   config-load warning.
3. **Small-N handoff.** `mode: approx` runs exact when 8·N² ≤ 1 GiB and N²·G_sim ≤ 1e9. The rule reads the config
   only, so every host makes the same choice. `run.yaml` records the mode that ran.
4. **Exact engine and refusal.** A pure function of (N, G_sim), using the fitted cost models (the recommendation
   §8.5) at a fixed reference of 8 threads: densep when 8·N² ≤ 16 GiB and its predicted time does not exceed
   walkd2's, else walkd2. Refuse, naming the prediction and pointing to `approx`, when twice the chosen engine's
   prediction exceeds 1 h. The thread count and `--max-memory` never change the choice. The refusal happens in the simulate stage, not at
   config load, so it fails only that scenario's reps. `run.yaml` records the
   engine.
5. **Tail layout.** xexact packs generation offset, local id and path multiplicity into one uint32 when
   bitlen(N − 1) + (depth − 1) + bitlen(depth − 1) ≤ 32, else it uses an int32 id with a float32 coefficient. F is
   bitwise the same either way. The packed word exists only in memory, never on disk. This is an explicit exception
   to the rule against storing ids and counts below int32.
6. **No slot recycling and no xclose** in v1. Both trade code for memory (0.77 GB and 30% at 5M); neither is needed
   at production N (≤ 1M).
7. **Threads.** Under `--jobs N > 1` the simulate stage gets `NUMBA_NUM_THREADS = max(1, cores // N)`, as
   pedigree-graph gets cores/N; other stages stay at 1.
8. **F output.** When the mode is not `off`, the pedigree carries a float32 column `F` after `household_id`: the
   inbreeding coefficient the simulation used (approximate under `approx`). With `off` the column is absent.
9. **Home.** A numba module in simACE (`simace/simulation/inbreeding.py`) with a per-generation step called from
   `run_simulation`'s loop. pedigree-graph stays an independent oracle.
10. **Validation.**
    - pytest in simACE:
      - exact F against pedigree-graph `inbreeding()` and dense float64 on small pedigrees (MZ twins, burn-in);
      - approx with depth ≥ G_sim resolves to exact, and xexact (forced past the handoff) meets an RMS band at
       random mating;
      - `off` byte-identical with an unchanged random stream;
      - F identical at 1 and 4 threads.
    - simACE-validate: the calibrated test of exact mode's per-generation Var(A1) against the pedigree-conditional
      infinitesimal expectation under random mating. Under AM, the Bulmer recursion of the AM note is a reported
      check, not a scored one.

## Consequences

- One seed reproduces the same pedigree on any host and thread count for a given config and simACE version.
  Changing the cost-model constants in code can change an engine choice, so `run.yaml` records the engine.
- Under assortative mating, runs that differ only in mode (or depth) are not paired: their pedigrees diverge within
  2–3 generations of the first F difference. Comparisons across modes must be distributional.
- Approx at depth 6 stays the default even under strong AM. The caveat lives in the documentation only.
- 1M/G 20 and similar deep, large exact runs are refused; they need `approx`.
- `simace validate` checks that assume a Mendelian variance of A/2 change with the mode: the inheritance
  variance cell scores residuals normalised by √(1 − F̄), and the generation-0 total-variance check is not
  applicable when the mode is not `off`.
- The pedigree schema gains a conditional column; consumers that estimate F from the recorded pedigree
  (fitACE `export_inbreeding`, simACE-validate) should read `F` when present.

## Alternatives rejected

- **A tolerance knob** (RMS mapped to depth): promises accuracy the law does not deliver at small N or strong AM.
- **Host-aware engine choice** (fastest at the run's threads and memory budget): engine, and so F's last bits and
  possibly an AM pedigree, could differ between a laptop and a cluster run of one seed.
- **Auto-raising depth under strong AM, or a default of 8**: a hidden dependence on another key, or a cost paid by
  every run; the measured gain from 6 to 8 under AM is about 2×, not 4×.
- **Engines in pedigree-graph**: needs an incremental API against an immutable graph (pg ADR 0007), a pg release
  per change, and costs pedigree-graph its role as an independent oracle.
