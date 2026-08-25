# Prevalence-target semantics and phenotype-model assignment for the calibrated scenario folders

- **Status:** accepted

The `neurodev` and `paper_simulation` folders were authored with documented
target prevalences (e.g. ADHD 6%, ID 0.4%, the `prev_*` sweep 2%/10%/35%) but
realized 10–250× off, because every scenario used `model: frailty`, whose case
fraction is emergent rather than configured. Fixing this forced two coupled
decisions about what "hit the target prevalence" means and which phenotype
model each scenario should use.

**The root fact: plain `frailty` has no lifetime prevalence $K$.** A Weibull
baseline cumulative hazard $H_0(t)=(t/\text{scale})^\rho \to \infty$, so every
individual onsets eventually — the asymptotic prevalence is 100% regardless of
`scale`. This is *why* the model rejects a `prevalence` param. Plain frailty can
therefore only be calibrated to **prevalence-by-a-reference-age** (we use the
gen-0, `[0,80]`, fully-observed cohort's `affected` fraction). The
threshold-bearing models (`adult`/`ltm` and `cure_frailty`) *do* carry a true
lifetime $K$ via a liability threshold, so their `prevalence:` config stays
equal to the round-number target — clean ground truth for fitACE to recover.

We accept that the two arms target **different definitions** of prevalence
rather than forcing them to match. We rejected back-solving the threshold
models' $K$ so their *observed* prevalence equals the plain-frailty arm
(option b): it would make ground-truth $K$ a non-round number and defeat the
point of having a clean configured target. For early-onset scenarios the two
definitions nearly coincide; for late-onset they diverge by the
$K$-vs-observed (censoring + competing-death) gap, which we treat as a
*reported output* of the frailty-vs-LTM comparison.

**A second consequence: early-onset + modest-prevalence is intrinsically a cure
model.** In plain frailty the same hazard governs both onset age and
prevalence-by-80 — early onset implies high by-80 incidence — so you cannot have
"a minority gets it, and they get it young" without a cure fraction. Forcing it
via `scale` would push the median onset to hundreds of years and scatter the
few by-80 cases across all ages, contradicting the scenarios' own
documentation. We therefore assign each frailty scenario to one of three tiers:

| Tier | Model | Scenarios |
|------|-------|-----------|
| A. early-onset | `cure_frailty` ($K$=target, hazard preserved) | ADHD, ASD, ID, `onset_early`, `stress_low_herit_early_rare` |
| B. late-onset | plain `frailty` (calibrate `scale` to observed-by-80) | `onset_late`, `stress_high_herit_late_rare` |
| C. onset-agnostic | plain `frailty` (calibrate `scale` to observed-by-80) | `herit_low/moderate/high`, `prev_rare/moderate/common`, `censoring_no_mortality`, `censoring_with_mortality`, `stress_shared_env_common` |
| D. already threshold-bearing | `adult` (`method: ltm`) — untouched by this ADR | `onset_adult` |

Tier D is a bookkeeping row, not a reassignment. `paper_simulation`'s
`onset_adult` was authored on `adult`/`ltm` from the start, so it never had the
emergent-prevalence problem this ADR exists to fix: its `prevalence:` config
(0.10 / 0.20) is already a true lifetime $K$ carried by a liability threshold,
and its onset timing comes from the cumulative-incidence curve (`cip_x0`,
`cip_k`) rather than from a Weibull hazard. It is listed here only so the table
accounts for every scenario in the two folders — the three tiers above partition
the *frailty* scenarios, which is what needed deciding.

We rejected staying strictly in plain `frailty` everywhere (option i): the tier-A
scenarios would lose the early-onset clustering that defines them. The parallel
`neurodev_ltm` / `paper_simulation_ltm` folders run the same scenarios under
`adult`/`ltm` as a different-mechanism comparison at matched $K$, using the same
per-scenario seed so the two arms are paired on identical pedigrees and
liabilities.
