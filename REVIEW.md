# Peer Review: ACE Population Genetics Simulation Framework

**Manuscript type:** Software/Methods paper
**Reviewer recommendation:** Accept with minor revisions

---

## General Comments

This manuscript presents a well-engineered simulation framework for generating multi-generational pedigrees with classical ACE (Additive genetic, Common environment, unique Environment) variance-component decomposition. The software maps continuous liabilities to observable phenotypes via two complementary models—a Weibull proportional-hazards frailty model and a liability-threshold model—and includes a comprehensive automated validation and statistical analysis pipeline. The mathematical formulations are correct, the implementation is computationally efficient, and the documentation (particularly `methods.md`) approaches publication-ready quality. Below I detail both strengths and areas requiring revision.

---

## Scientific Methods — Strengths

### Variance decomposition and inheritance model
The core simulation correctly implements the bivariate normal ACE model with cross-trait genetic (rA) and shared-environment (rC) correlations. The infinitesimal model for additive genetic inheritance (midparent + Mendelian sampling with segregation variance σ²_A/2) follows Bulmer (1971) and is standard for quantitative genetics simulations. MZ twins correctly share identical breeding values and sex. The common-environment component is properly drawn per household (not transmitted across generations), consistent with the classical ACE assumption.

### Weibull frailty model
The proportional-hazards frailty model is correctly parameterised and event times are properly generated via inverse-CDF sampling. The standardisation of liability within recorded generations before applying the frailty transformation is methodologically sound and prevents founder-generation bias.

### Censoring mechanisms
The dual-censoring model (age-window + competing-risk death) is well-designed. The implementation correctly handles left-censoring, right-censoring, and death-censoring with proper indicator assignment. Per-generation censoring windows allow realistic cohort-based observation patterns.

### Pairwise Weibull survival correlation
The composite likelihood approach with Gauss-Hermite quadrature (Section 3.3 of methods.md) is technically sophisticated and appropriate for estimating liability correlations from censored survival data. The Cholesky decomposition for correlated bivariate integration and log-sum-exp stabilisation demonstrate strong numerical methods. The Numba JIT parallelisation is a practical contribution.

### Validation suite
The five-category validation framework (structural, twin, half-sibling, variance-component, population-level) is thorough. The tolerance formula max(4·SE, ε_min) is statistically appropriate, and the twin-rate expected-value formula correctly accounts for eligible positions and the doubling effect of MZ pairs.

---

## Scientific Methods — Concerns

### 1. Model limitations insufficiently discussed
The methods document should explicitly state the following assumptions as limitations:

- **No assortative mating.** Spouses are paired randomly. Real populations exhibit phenotypic and genetic assortative mating, which inflates observed sibling correlations and can bias heritability estimates upward. The absence of assortative mating should be noted as a simplification.

- **No gene-environment interaction (G×E).** Liability is strictly additive (L = A + C + E). If the true data-generating process has multiplicative or interactive effects, estimates from the ACE model will be biased. This is inherent to the ACE framework but warrants explicit statement.

- **No cross-trait unique-environment correlation (rE = 0).** While this is the standard assumption, it limits the ability to model trait pairs where environmental factors cluster across phenotypes (e.g., shared lifestyle exposures).

- **No environmental transmission across generations.** C is freshly drawn each generation with no autoregressive component. This is standard but unrealistic for many applications (socioeconomic persistence, neighbourhood effects).

**Recommendation:** Add a "Limitations" section to methods.md enumerating these assumptions and their implications.

### 2. Numerical Hessian for standard errors
The SE for the pairwise Weibull correlation estimate uses a numerical second-difference approximation with fixed step h = 10⁻⁴. While adequate, this is sensitive to the choice of h and provides no error bounds on the SE itself. Consider:
- Documenting sensitivity to h (e.g., does SE change meaningfully for h ∈ {10⁻³, 10⁻⁴, 10⁻⁵}?)
- Alternatively, implementing the analytic observed Fisher information, which is feasible given the known Weibull likelihood form.

### 3. Tetrachoric correlation under censoring
The tetrachoric correlation from binary affection status will be attenuated when censoring reduces observed prevalence relative to true prevalence. This is a known limitation that should be explicitly documented. The pairwise Weibull method is the appropriate alternative, but users should be warned that tetrachoric estimates from censored data are biased.

---

## Reproducibility — Major Issues

### 4. CRITICAL: No dependency version pinning (environment.yml)
`environment.yml` lists all dependencies without version constraints:
```yaml
dependencies:
  - numpy       # no version
  - scipy       # no version
  - matplotlib  # no version
  - snakemake   # no version
  - numba       # no version
  ...
```
This is the single most significant reproducibility gap in the package. Over time, upstream changes in numpy, scipy, or snakemake will silently alter numerical results, break API compatibility, or change workflow semantics. For an academic publication, exact reproducibility is essential.

**Recommendation:** Generate and commit a lock file (e.g., `conda-lock.yml` or `conda env export > environment-lock.yml`). Document in README that users should install from the lock file for exact reproducibility, while keeping the unpinned `environment.yml` for development flexibility.

### 5. README.md is outdated
The README references obsolete parameter names (`A`, `C`, `ngen` instead of `A1`, `A2`, `C1`, `C2`, `G_ped`), an obsolete Snakefile path (`-s workflow/Snakefile` instead of the root `Snakefile`), and an outdated project structure that omits the entire `sim_ace/` package. A reader following the README instructions would be unable to correctly configure or run the software.

**Recommendation:** Rewrite the README to match the current architecture as documented in CLAUDE.md. Include:
- Updated parameter names and config format
- Correct snakemake invocation (no `-s` flag)
- Current project structure showing `sim_ace/` package
- Output structure with `{folder}/{scenario}/rep{N}/` hierarchy

---

## Reproducibility — Minor Issues

### 6. No CITATION.cff file
No citation metadata exists. For academic publication, a `CITATION.cff` file (Citation File Format) enables automatic citation generation by GitHub and reference managers.

**Recommendation:** Add `CITATION.cff` with title, authors, version, license, DOI placeholder.

### 7. Seed offset collision risk
Different pipeline stages use fixed offsets from the base seed: trait-2 phenotyping uses `seed + 100`, death censoring uses `seed + 1000`. These offsets are undocumented outside of `methods.md` line 233 and could collide if a user inadvertently sets base seeds for different scenarios within 100 of each other.

**Recommendation:** Document the seed offset scheme prominently (e.g., in config.yaml comments) and add a validation check that warns if scenario seeds are within 1000 of each other.

---

## Test Coverage

### 8. Test suite assessment
The test suite is strong: ~87 test functions across 9 modules cover unit tests, statistical property validation, and edge cases. Statistical tests use appropriately seeded large pedigrees (N=5000) with realistic tolerance bands. Key strengths include:
- Founder variance recovery tests
- MZ twin identity verification
- Heritability estimation via Falconer's formula
- Tetrachoric correlation accuracy
- Edge cases (zero variance, no twins, pure-E model)

**Gaps to address:**
- **No plot tests.** The 6 visualization modules (`plot_*.py`) have zero test coverage. Even smoke tests (generate plot without error) would catch regressions.
- **No regression baselines.** No committed reference outputs for bit-for-bit comparison across environments.
- **Scale testing limited.** All tests run at N=5000, but production scenarios use N=100K-2M. Consider at least one integration test at larger scale.

---

## Documentation Quality

### 9. methods.md — Excellent
The 234-line methods document is the strongest component of the documentation. It provides complete mathematical formulations with proper LaTeX notation for all models (variance decomposition, Weibull frailty, censoring, liability threshold, tetrachoric correlation, pairwise Weibull likelihood, heritability estimation). This document is suitable as a manuscript methods section or supplementary material with minimal editing.

**Minor improvements needed:**
- Add a formal references/bibliography section (currently only Bulmer 1971 is cited inline; Owen's T function and Falconer's formula deserve proper citations)
- Add a notation table defining all symbols in one place
- Number the equations for cross-referencing

### 10. Code documentation — Very Good
All modules have docstrings, type hints (Python 3.10+ syntax), and logging. The `survival_corr.py` module docstring (19 lines with full model specification) is exemplary. Error handling with informative messages is present throughout `simulate.py`.

**Gaps:** Plot modules have sparser docstrings than simulation modules. The `gather.py` function (165 lines extracting 55+ metrics) would benefit from modularisation.

---

## Visualisation Quality

### 11. Publication readiness — Very Good
The 20+ plot types across 6 modules generally meet journal figure standards:
- Proper axis labels, titles, and legends throughout
- Statistical annotations (R², prevalence, sample sizes, correlation values)
- High-resolution output (dpi=150)
- Appropriate use of transparency, colour, and multi-panel layouts
- Downsampling to 100K points prevents rendering issues

**Improvements needed for submission:**
- **Output format.** Plots are saved as PNG only. Journal submissions typically require vector formats (PDF/SVG) for figures. Add PDF output option.
- **Colour-blind accessibility.** No explicit colour-blind-safe palette is enforced. Consider using `seaborn.set_palette("colorblind")` or similar.
- **Font sizes.** Some panels have axis labels that may be too small for 2-column journal layouts. Verify readability at typical figure widths (3.5 in single-column, 7 in double-column).
- **Figure captions.** No separate caption text exists for any plot. Create a supplementary figure legends document.

---

## Software Engineering

### 12. Architecture — Excellent
The separation into an installable Python package (`sim_ace/`) with thin Snakemake wrapper scripts is clean and enables both pipeline and standalone use. The modular decomposition (simulate, phenotype, censor, threshold, validate, stats, plot_*) maps naturally to the scientific workflow. The `cli_base.py` and `utils.py` shared modules reduce duplication.

### 13. Performance — Good
Numba JIT for the pairwise Weibull inner loop (5-7x speedup) and vectorised numpy operations throughout the simulation core demonstrate attention to computational efficiency. The Snakemake pipeline supports SLURM-based HPC execution for large-scale runs.

### 14. License — Present
MIT License is included, which is appropriate for academic software.

---

## Summary of Required Revisions

### Must address before publication:
| # | Issue | Severity |
|---|-------|----------|
| 4 | Pin dependency versions in environment.yml (or provide lock file) | Critical |
| 5 | Update README.md to match current codebase | Major |
| 1 | Add Limitations section to methods.md | Major |

### Should address:
| # | Issue | Severity |
|---|-------|----------|
| 6 | Add CITATION.cff | Minor |
| 9 | Add references/bibliography to methods.md | Minor |
| 11 | Add PDF/SVG output for figures | Minor |
| 8 | Add smoke tests for plot modules | Minor |
| 3 | Document tetrachoric bias under censoring | Minor |

### Optional improvements:
| # | Issue |
|---|-------|
| 2 | Sensitivity analysis for numerical Hessian step size |
| 7 | Document seed offset scheme more prominently |
| 11 | Colour-blind palette, font size review |
| 8 | Regression baseline outputs, larger-scale integration test |
| 9 | Equation numbering and notation table in methods.md |

---

## Verdict

**Accept with minor revisions.** The scientific methods are sound, the implementation is correct and efficient, and the documentation (particularly methods.md) is of high quality. The critical revision is dependency pinning for reproducibility. The README must be updated to match the current codebase. A brief Limitations section should be added to the methods. Remaining items are minor enhancements that would strengthen the package for publication but do not affect scientific validity.
