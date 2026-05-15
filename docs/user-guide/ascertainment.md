# Ascertainment

The pipeline can shrink and bias the simulated population to mimic
real-world data limitations through a single **ascertainment** stage that runs
after censoring (per [ADR 0001](../adr/0001-unified-ascertainment-stage.md)).
The stage has two knobs — a uniform `dropout_rate` and a trait-weighted
`case_ascertainment_ratio` — plus a target sample size `N_sample`. Outputs
are the canonical `pedigree.parquet` / `trait.parquet` / `trait.simple_ltm.parquet`
that downstream stats and fitACE consume.

## Algorithm

Two explicit steps applied to IDs (not weights, which would silently cancel
under a fixed-size weighted draw):

1. **Uniform pedigree dropout.** `round(N_total * dropout_rate)` individuals
   are removed uniformly at random from the full pedigree. Any
   `mother` / `father` / `twin` references pointing to dropped IDs are set to −1.
2. **Case-weighted N_sample draw** from the post-dropout *trait* pool. Weights
   are `case_ascertainment_ratio` for cases and `1` for controls; when
   `N_sample <= 0` or `>= len(post-dropout trait)`, everything passes through.

The same sampled IDs are applied to both `trait.parquet` and
`trait.simple_ltm.parquet`. The pedigree output is the **ancestor closure** of
the sampled IDs within the post-dropout pedigree, with dangling parent / twin
references rewritten to −1. Validation (`validate_*`) is unaffected — it
continues to consume `pedigree.full.parquet` (the pre-ascertainment full
pedigree).

## Config

```yaml
ascertainment:
  N_sample: 0                      # 0 = pass everything through
  case_ascertainment_ratio: 1      # 1 = uniform; >1 enriches cases
  dropout_rate: 0                  # 0 = no uniform dropout
```

## Dropout (`dropout_rate`)

Removes individuals uniformly at random. Independent of trait status, sex,
generation, or pedigree position. Models registry incompleteness or random
ascertainment failure.

Pre-configured dropout scenarios are in `config/ascertainment.yaml`:
`baseline100K_dropout10` (10 %), `baseline100K_dropout30` (30 %),
`baseline100K_dropout50` (50 %).

Because dropout severs parent/twin pointers, multi-hop relationships through
removed individuals (e.g., grandparent-grandchild via a dropped parent) become
undetectable, and former full-sib pairs whose shared parent was dropped are
reclassified as half-sibs.

## Case ascertainment (`case_ascertainment_ratio`)

When `case_ascertainment_ratio != 1` and `N_sample > 0`, the sample draw
uses weights `case_ascertainment_ratio` for cases (`affected1 == True`) vs `1`
for controls. With 10 % population prevalence and
`case_ascertainment_ratio: 5`, a case is 5× more likely to be drawn than a
control, yielding ~36 % cases in the sample.

Edge cases:

- **ratio = 0**: only controls are sampled; `N_sample` is clamped to the number
  of available controls
- **ratio = 1** (default): uniform sampling (fast path)
- **0 cases or all cases**: falls back to uniform with a warning
- **N_sample = 0 with ratio != 1**: warning logged; ratio has no effect because
  no draw happens
- **Extreme ratios**: warns if >90 % of total cases would be expected in the sample

The ratio is recorded in per-rep stats YAML when != 1; no correction is applied
to downstream estimates — the purpose is to study the bias.

## Relationship recovery after ascertainment

The output `pedigree.parquet` is the ancestor closure of sampled IDs, so most
within-sample relationships are recoverable through intact parent edges. The
relationship extraction code in `PedigreeGraph` uses two strategies:

| Relationship type | How it works with ascertained data |
|---|---|
| **Siblings** (full, maternal HS, paternal HS) | Classified using the *original* `mother` / `father` parent IDs stored in the row, not via row-index walks. Two sampled individuals are detected as siblings if their parent IDs match — even if the parent itself isn't in the closure. |
| **Parent-offspring** | Detected when a parent is in the closure (its ID maps to a valid row). Each parent link is independent — a child with only its mother in the closure still yields a mother-offspring pair. |
| **Grandparent-grandchild, avuncular, cousins, 2nd cousins** | Detected via sparse matrix products on parent→child edges. Ancestor closure ensures that grandparents (and great-grandparents for 2nd cousins) of sampled individuals are in the pedigree when reachable through intact edges. |
| **MZ twin** | Detected when both twins are in the sample; the closure-only fixup pass severs twin pointers whose partner is not in the closure. |
