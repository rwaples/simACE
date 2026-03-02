# Pipeline Benchmarks — February 27, 2026

Snakemake benchmark data aggregated across all scenarios and replicates.
Values are **mean wall-clock seconds / mean peak RSS (MB)** per job.

## Per-step runtime and memory by population size

| Step | N=1K | N=10K | N=100K | N=1M | N=2M |
|------|------|-------|--------|------|------|
| simulate | 2.9s / 216 MB | 4.0s / 235 MB | 8.0s / 475 MB | 27.8s / 1,974 MB | 83.9s / 3,870 MB |
| phenotype_weibull | 6.8s / 223 MB | 7.2s / 245 MB | 11.4s / 509 MB | 31.6s / 1,814 MB | 70.1s / 2,418 MB |
| censor_weibull | 12.2s / 219 MB | 14.7s / 234 MB | 14.5s / 377 MB | 55.0s / 1,180 MB | 79.1s / 2,048 MB |
| phenotype_threshold | 3.3s / 212 MB | 8.9s / 239 MB | 10.8s / 440 MB | 25.8s / 2,017 MB | 67.2s / 3,378 MB |
| validate | 6.4s / 225 MB | 4.1s / 257 MB | 23.4s / 633 MB | 133.6s / 3,976 MB | 365.9s / 7,877 MB |
| phenotype_stats | 71.3s / 263 MB | 240.7s / 318 MB | 771.8s / 772 MB | 1,044.0s / 4,235 MB | 3,791.7s / 8,148 MB |
| threshold_stats | 4.9s / 227 MB | 9.4s / 268 MB | 20.1s / 605 MB | 58.7s / 2,616 MB | 112.0s / 4,997 MB |
| plot_phenotype | 144.9s / 537 MB | 108.4s / 543 MB | 115.1s / 648 MB | 16.6s / 790 MB | 12.4s / 709 MB |
| plot_threshold | 32.3s / 358 MB | 102.3s / 432 MB | 80.7s / 465 MB | 47.4s / 432 MB | 57.1s / 472 MB |

## Peak memory (max RSS across all replicates)

| Step | N=1K | N=10K | N=100K | N=1M | N=2M |
|------|------|-------|--------|------|------|
| simulate | 216 MB | 245 MB | 529 MB | 2,311 MB | 4,489 MB |
| phenotype_weibull | 224 MB | 250 MB | 538 MB | 2,102 MB | 2,443 MB |
| censor_weibull | 222 MB | 237 MB | 441 MB | 1,347 MB | 2,048 MB |
| phenotype_threshold | 223 MB | 243 MB | 480 MB | 2,029 MB | 4,056 MB |
| validate | 227 MB | 273 MB | 780 MB | 4,059 MB | 8,207 MB |
| phenotype_stats | 264 MB | 319 MB | 901 MB | 4,253 MB | 8,168 MB |
| threshold_stats | 227 MB | 274 MB | 775 MB | 3,213 MB | 5,854 MB |
| plot_phenotype | 537 MB | 543 MB | 675 MB | 790 MB | 709 MB |
| plot_threshold | 358 MB | 432 MB | 493 MB | 432 MB | 472 MB |

## Worst-case runtime (max seconds across all replicates)

| Step | N=1K | N=10K | N=100K | N=1M | N=2M |
|------|------|-------|--------|------|------|
| simulate | 3.0s | 5.7s | 30.0s | 31.9s | 199.4s |
| phenotype_weibull | 14.5s | 12.7s | 27.5s | 44.3s | 108.8s |
| censor_weibull | 19.4s | 19.0s | 27.6s | 62.4s | 87.2s |
| phenotype_threshold | 4.4s | 15.1s | 23.8s | 40.3s | 117.6s |
| validate | 10.9s | 5.8s | 49.9s | 153.4s | 473.1s |
| phenotype_stats | 82.8s | 263.0s | 1,157.1s | 1,059.0s | 5,006.1s |
| threshold_stats | 7.6s | 15.6s | 35.8s | 81.7s | 131.7s |
| plot_phenotype | 144.9s | 108.4s | 208.4s | 16.6s | 12.4s |
| plot_threshold | 32.3s | 102.3s | 112.8s | 47.4s | 57.1s |

## Replicate counts

| N | Scenarios | Reps per scenario | Total reps |
|---|-----------|-------------------|------------|
| 1,000 | 1 (small_test) | 3 | 3 |
| 10,000 | 1 (baseline10K) | 3 | 3 |
| 100,000 | 12 | 3 | 36 |
| 1,000,000 | 1 (baseline1M) | 3 | 3 |
| 2,000,000 | 1 (baseline2M) | 3 | 3 |

## Notes

- **phenotype_stats is the pipeline bottleneck**: tetrachoric correlation estimation dominates runtime, taking ~13 min at N=100K and ~63 min at N=2M.
- **validate and phenotype_stats are the memory bottleneck** at large N: both approach 8 GB at N=2M.
- **Plotting time is roughly constant** regardless of N because plots use downsampled data.
- All benchmarks recorded by Snakemake's `benchmark:` directive into `benchmarks/{folder}/{scenario}/rep{N}/{step}.tsv`.
- Measured on a single machine; wall-clock times include Python startup overhead (~2–3s baseline visible in small-N runs).
