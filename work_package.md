# ACE Package Review: Required Changes Before Publication

## 1. No Package Structure Exists

There is no `pyproject.toml`, `setup.py`, or `__init__.py`. The code lives as flat Snakemake scripts under `workflow/scripts/` with no importable module.

**Required:**
- Create a proper package (e.g., `ace_sim/`) with `__init__.py` modules separating simulation, phenotyping, validation, and plotting.
- Add `pyproject.toml` with metadata (name, version, license, authors, Python version constraint, entry points).
- The Snakemake workflow should import from the package rather than using `script:` directives that rely on implicit `snakemake` global variables.

## 2. No Tests

There are zero test files. For a methods paper accompanying a software package, this is a blocking deficiency.

**Required:**
- **Unit tests** for core functions: `mating()`, `reproduce()`, `generate_correlated_components()`, `generate_mendelian_noise()`, `apply_threshold()`, `simulate_phenotype()`, `age_censor()`, `death_censor()`, `tetrachoric_corr_se()`.
- **Statistical tests** (seeded): verify that `Var(A1)` in founders matches the input parameter, MZ twins share identical `A` values, sibling `A` correlation converges to 0.5, cross-trait correlations match `rA`/`rC`, E components are independent across siblings.
- **Regression tests**: a small golden-seed scenario (e.g., N=500, G_ped=2) with saved expected outputs, checked to bit-level reproducibility.
- **Edge case tests**: A=0, C=0, p_mztwin=0, p_nonsocial_father=1.0, single-generation pedigrees.

## 3. No Input Validation in Core Functions

`run_simulation()` only checks `G_sim >= G_ped`. The following are silently accepted with undefined behavior:

**Required:**
- `simulate.py:run_simulation()` - Validate `A1 + C1 <= 1.0` and `A2 + C2 <= 1.0` (otherwise `E` variance is negative, `np.sqrt` produces `nan`).
- Validate `0 <= A1, C1, A2, C2 <= 1`, `N > 0`, `G_ped >= 1`, `fam_size > 0`.
- Validate `-1 <= rA, rC <= 1` and that the resulting covariance matrices are positive semi-definite.
- `phenotype.py:simulate_phenotype()` - Validate `rate > 0`, `k > 0`, `beta` finite.
- `phenotype_threshold.py:apply_threshold()` - Validate `0 < prevalence < 1`.

## 4. E Variance Is Not a Free Parameter (Implicit Constraint Undocumented)

`simulate.py:248` computes `E1 = 1.0 - A1 - C1`. This means the total variance is hard-coded to 1.0 and E is residual. This is a modeling choice that is:
- Not documented in the function docstring for `run_simulation()`
- Not stated in the config comments
- Not validated (negative E is silently allowed)

**Required:** Document this constraint explicitly in the docstring and the config file. Add a validation guard.

## 5. Snakemake-Coupled Entry Points

All scripts use the pattern:
```python
if __name__ == "__main__":
    params = snakemake.params  # magic global
```

This makes the code un-callable outside Snakemake. `compute_phenotype_stats.py` and `plot_validation.py` have partial CLI fallbacks, but `simulate.py`, `phenotype.py`, `validate.py`, and `phenotype_threshold.py` do not.

**Required:**
- Every script should have a standalone CLI entry point (e.g., `argparse` or `click`) independent of Snakemake.
- The package should expose a programmatic Python API (e.g., `ace_sim.simulate(...)`, `ace_sim.phenotype(...)`) that the CLI and Snakemake both call.

## 6. Reproducibility: Seed Management Is Fragile

- `phenotype.py:86` uses `seed + 100` for trait 2 and `seed + 1000` for death ages. These magic offsets are undocumented and could collide for large seed values.
- `compute_phenotype_stats.py` and `compute_threshold_stats.py` hard-code `seed=42` as a default in their `main()` functions.
- The simulate rule computes `seed + int(w.rep) - 1` (`simulate.smk:90`), meaning rep seeds differ by 1 -- marginal independence between replicates.

**Required:**
- Use a proper `SeedSequence`-based approach (`np.random.SeedSequence(seed).spawn(n)`) to generate independent sub-streams for each component (trait 1, trait 2, death, each replicate).
- Document the seed derivation scheme.

## 7. `sys.path` Manipulation for Cross-Script Imports

`compute_threshold_stats.py:14` and the plotting scripts do:
```python
sys.path.insert(0, str(Path(__file__).parent))
from compute_phenotype_stats import tetrachoric_corr_se, ...
```

This is fragile and breaks under editable installs, zip imports, and namespace conflicts.

**Required:** Once a proper package structure exists, use standard relative or absolute imports.

## 8. No Docstring / API Documentation

While internal function docstrings exist, there is no user-facing documentation:
- No API reference
- No tutorial or quickstart
- No mathematical specification of the model (the proportional hazards frailty model in `phenotype.py` header comment is the closest, but incomplete)

**Required:**
- A methods document (or supplementary material) specifying the full generative model mathematically: the bivariate ACE decomposition, Mendelian segregation model, the Weibull frailty phenotype model, the censoring model, and the liability threshold model.
- API documentation (Sphinx/MkDocs).

## 9. Pinned Dependency Versions Are Missing

`environment.yml` lists dependencies without version pins (e.g., `numpy`, `scipy`, `pandas`). This risks silent behavioral changes across environments.

**Required:**
- Pin major versions in `pyproject.toml` (e.g., `numpy>=1.24,<2.0`).
- `environment.yml` should pin or at minimum specify compatible version ranges.
- Drop dependencies not required by the core package (e.g., `jupyter`, `notebook`, `graph-tool`, `msprime`, `tskit` appear unused in the simulation code).

## 10. Scientific Correctness Concerns

**10a. Common environment model.** `reproduce()` draws fresh `C` per household each generation (`simulate.py:163`). This means `C` is NOT transmitted from parent to child -- it is re-drawn. This is the standard ACE model assumption but should be explicitly stated, as some readers may expect C to have an autoregressive component across generations.

**10b. Mendelian noise variance.** `generate_mendelian_noise()` uses `sd_A / sqrt(2)` (`simulate.py:50`), which is correct under infinitesimal model assumptions (mid-parent value + segregation variance = parental variance). This derivation should be cited or proven in documentation.

**10c. Twin rate parameterization.** The conversion `p_twin_birth = p_mztwin / (2 - p_mztwin)` (`simulate.py:82`) assumes every twin birth produces exactly 2 individuals. The docstring says `p_mztwin` is the "fraction of individuals who are twins", but the config comment says "probability of a birth producing MZ twins." These are different quantities -- clarify which is the input parameter.

**10d. Tetrachoric correlation SE.** The Fisher information is estimated numerically with `dx=1e-4` (`compute_phenotype_stats.py:96`). At boundary values of `r` near +/-1, this can produce unstable or infinite SE estimates. Consider using the analytic Fisher information or a profile-likelihood CI.

**10e. `safe_corrcoef` uses exact `== 0`** for floating-point standard deviation (`validate.py:14`). This will virtually never trigger. Use a tolerance (e.g., `< 1e-10`).

## 11. Separation of Simulation vs. Analysis

The package conflates simulation (the scientific contribution) with analysis/plotting (convenience utilities). For a clean release:

**Required:**
- Core package: `simulate`, `phenotype`, `phenotype_threshold` (the generative model).
- Optional extras: `validation`, `plotting`, `stats` (installable via `pip install ace-sim[plots]`).

## 12. Logging

Scripts use bare `print()` statements (e.g., `compute_phenotype_stats.py:389`). A published package should use Python's `logging` module to allow users to control verbosity.

## 13. Type Annotations

No type annotations anywhere. At minimum, public API functions should have typed signatures for IDE support and documentation generation.

## 14. License

No `LICENSE` file exists. This is a hard requirement for any published package.

## Summary Table

| Priority | Issue | Effort |
|----------|-------|--------|
| **Blocking** | No tests | High |
| **Blocking** | No package structure (`pyproject.toml`, `__init__.py`) | Medium |
| **Blocking** | No license | Trivial |
| **Blocking** | No input validation (negative variance possible) | Medium |
| **High** | Snakemake-coupled entry points; no standalone API | Medium |
| **High** | `sys.path` hacks for imports | Low (follows from package structure) |
| **High** | No mathematical model specification | Medium |
| **High** | Unpinned dependencies; unused deps | Low |
| **High** | Seed management fragility | Medium |
| **Medium** | E=1-A-C constraint undocumented | Low |
| **Medium** | Twin rate parameter ambiguity | Low |
| **Medium** | `print()` instead of `logging` | Low |
| **Medium** | No type annotations | Medium |
| **Low** | Tetrachoric SE numerical instability | Low |
| **Low** | `safe_corrcoef` float equality check | Trivial |
| **Low** | Separate core vs. optional extras | Medium |
