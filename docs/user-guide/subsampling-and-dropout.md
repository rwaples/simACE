# Subsampling and Dropout

The pipeline can shrink and bias the simulated population to mimic
real-world data limitations: subsampling reduces dataset size, case
ascertainment biases who ends up in the sample, and pedigree dropout
mimics incomplete registry coverage.

## Subsampling (`N_sample`)

When `N_sample > 0`, the pipeline randomly draws `N_sample` individuals from
the phenotype before computing statistics. This reduces runtime and disk usage
for large populations while preserving population-level signals. The sampling
step (`sample.smk`) writes a temporary `.sampled.parquet` that is auto-deleted
after stats complete.

Because sampling breaks pedigree completeness — parents and other relatives
may not be in the sample — the relationship extraction code in `PedigreeGraph`
uses two strategies to recover as many valid pairs as possible:

| Relationship type | How it works with subsampled data |
|---|---|
| **Siblings** (full, maternal HS, paternal HS) | Classified using **original pedigree parent IDs** stored in the DataFrame columns, not row indices. Two sampled individuals are detected as siblings if their `mother`/`father` columns match, regardless of whether those parents are in the sample. Full sibs share both parent IDs; half-sibs share one and differ on the other. |
| **Parent-offspring** | Detected when a parent is present in the sample (its ID maps to a valid row index). Each parent link is independent — a child with only its mother in the sample still yields a mother-offspring pair. |
| **Grandparent-grandchild, avuncular, cousins, 2nd cousins** | Detected via sparse matrix products on parent→child edges. Each edge is built independently (mother edges and father edges are separate matrices), so a child with only one parent in the sample still contributes edges through that parent. However, these relationships require intermediate ancestors to be in the sample to form multi-hop paths. |
| **MZ twin** | Detected when both twins are in the sample (twin partner ID maps to a valid row index). |

## Case ascertainment bias (`case_ascertainment_ratio`)

When `case_ascertainment_ratio != 1` and `N_sample > 0`, sampling uses
weighted probabilities instead of uniform random selection. Cases
(`affected1 == True`) receive weight = ratio while controls receive weight = 1;
weights are normalized to probabilities and passed to
`rng.choice(p=..., replace=False)`.

For example, with 10% prevalence and `case_ascertainment_ratio: 5`, a case is
5x more likely to be drawn than a control, yielding ~36% cases in the sample
versus the population 10%.

Edge cases:

- **ratio = 0**: only controls are sampled; `N_sample` is clamped to the number of available controls
- **ratio = 1** (default): uniform sampling (fast path, backward compatible)
- **0 cases or all cases**: falls back to uniform sampling with a warning
- **N_sample = 0**: ratio has no effect (all individuals pass through); logs a warning if ratio != 1
- **Extreme ratios**: warns if >90% of total cases would be expected in the sample

The ratio is recorded in per-rep stats YAML when != 1 but no correction is
applied to downstream estimates — this is intentional, as the purpose is to
study the bias.

## Pedigree dropout (`pedigree_dropout_rate`)

When `pedigree_dropout_rate > 0`, the pipeline randomly removes that fraction
of individuals from the simulated pedigree before phenotyping, modelling
incomplete real-world observation. The dropout step (`dropout.smk`) runs
between simulation and phenotyping:

```
simulate → pedigree.full.parquet (temp) → dropout → pedigree.parquet
                                      ↘ validate (reads full pedigree)
```

Dropped individuals are deleted entirely. All parent/twin links pointing to a
dropped individual are set to -1 (unknown). This means multi-hop relationships
through missing individuals (e.g. grandparent-grandchild via a removed parent)
become undetectable, and former full-sib pairs whose shared parent was dropped
are reclassified as half-sibs. Individuals with only one known parent can
still participate in half-sib detection through the surviving parent.

Three pre-configured dropout scenarios are included:
`baseline100K_dropout10` (10%), `baseline100K_dropout30` (30%), and
`baseline100K_dropout50` (50%).
