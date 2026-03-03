# Remaining cleanup: stats.py / threshold_stats.py

## 1. Duplicate `compute_tetrachoric` in threshold_stats.py

`threshold_stats.py:78-104` is a byte-for-byte copy of `stats.py:467-493`. threshold_stats already imports `tetrachoric_corr_se`, `compute_liability_correlations`, and `create_sample` from stats — it should import `compute_tetrachoric` too and delete the local copy.

## 2. `weibull_corr` (censored) computed but never consumed

`stats.py` calls `compute_weibull_pair_corr()` twice per trait — once censored, once uncensored (`compute_weibull_correlations`, ~line 766). Only the uncensored result is read downstream (by `plot_correlations.py` for reference lines). The censored result is written to YAML as `weibull_corr` but no plot or analysis reads it. Dropping the censored call halves the Weibull correlation computation time.

## 3. `liability_by_status` computed but never consumed

`threshold_stats.py:122-124` calls `compute_liability_by_status()` and writes the result to `threshold_stats.yaml`, but no downstream module reads it. Can be removed.

## 4. `tetrachoric_se()` wrapper is dead code

`stats.py:152-155` defines `tetrachoric_se(r, a, b)` — a backward-compatible wrapper that re-calls `tetrachoric_corr_se` and discards the correlation. It has zero callers anywhere in the codebase (the sibling wrapper `tetrachoric_corr()` is actively used by plot modules, but `tetrachoric_se` is not). Can be removed.
