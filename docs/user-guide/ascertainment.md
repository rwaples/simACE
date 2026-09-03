# Ascertainment

The ascertainment stage models an incomplete registry and a case-enriched
study sample. It runs after the censor stage and writes the `pedigree.parquet`
and `trait.parquet` that the analyze stage and fitACE read.
[Methods, Ascertainment](../concepts/methods.md#ascertainment) explains the
design, and [ADR 0001](../adr/0001-unified-ascertainment-stage.md) records why
the two effects share one stage.

The stage takes three parameters, all under the scenario's `ascertainment:`
section. [Configuration](configuration.md#ascertainment-and-analysis) lists them
with their defaults. The worked example
[When the study sample is not the population](../examples/ascertainment-bias.md)
shows what they do to the estimates.

## Two steps on IDs

The stage removes individuals in two steps. Both steps act on individual IDs
rather than on sampling weights.

1. **Dropout.** The stage removes `round(N_total * dropout_rate)` individuals
   uniformly at random from the recorded pedigree. Any `mother`, `father`, or
   `twin` link to a removed individual becomes -1.
2. **Case-weighted draw.** From the post-dropout phenotyped rows, the stage
   draws `N_sample` individuals without replacement. A case, meaning
   `affected1` is true, has weight `case_ascertainment_ratio`. A control has
   weight 1. If `N_sample` is 0 or at least the pool size, the stage keeps every
   individual.

The drawn IDs become the rows of `trait.parquet`. The pedigree output is the
ancestor closure of those IDs within the post-dropout pedigree: the drawn
individuals plus every ancestor reachable through intact parent links. Links
that leave the closure are again set to -1.

Ascertainment does not change validation. Validation reads
`pedigree.full.parquet`, the recorded pedigree.

## Dropout

Dropout ignores trait status, sex, generation, and pedigree position. It
models an incomplete registry.

Because dropout sets parent and twin links to -1, any relationship that passes
through a removed individual disappears. A grandparent and grandchild whose
connecting parent was dropped are unrelated in the output. Full siblings whose
mother was dropped become paternal half-siblings, because only the father link
survives.

`config/ascertainment.yaml` defines three ready-made scenarios,
`baseline100K_dropout10`, `baseline100K_dropout30`, and
`baseline100K_dropout50`, at 10, 30, and 50 percent dropout.

## Case weighting

With 10 percent prevalence and a ratio of 5, a case is five times as likely
to be drawn as a control. About 36 percent of the sample are then cases.

The stage handles the edge cases as follows.

- A ratio of 1, the default, is a uniform draw.
- A ratio of 0 draws controls only. If fewer controls exist than `N_sample`,
  the stage lowers `N_sample` to the number of controls. It logs a warning
  about the change. If the pool holds no controls, the stage raises an error.
- If the pool holds no cases, or only cases, the ratio has no effect. The stage
  logs a warning and draws uniformly.
- If `N_sample` is 0, no draw happens, so the ratio has no effect. The stage
  logs a warning when the ratio is not 1.

The analyze stage copies the ratio into `report.yaml` under
`inputs.ascertainment`. It applies no correction to any estimate.

## Relationships in the output pedigree

The output pedigree is the ancestor closure of the sample, so most
relationships between two sampled individuals remain visible through intact
parent links. `PedigreeGraph`, in the external pedigree-graph package, finds
them as follows.

- **Siblings.** Grouped by the `mother` and `father` IDs stored on each row,
  not by walking to a parent row. Two sampled individuals with matching parent
  IDs are siblings even when the parent is outside the closure. Full-sibling
  detection needs both parent IDs. Half-sibling detection needs one.
- **Parent and offspring.** Found when the parent is in the closure. Each
  parent link counts on its own, so a child whose mother alone is in the
  closure still yields a mother-offspring pair.
- **Grandparents, avuncular pairs, cousins, and second cousins.** Found by
  sparse matrix products over parent-to-child edges. The ancestor closure
  keeps grandparents and great-grandparents of sampled individuals whenever
  an intact edge reaches them.
- **MZ twins.** Found when both twins are in the sample. The closure step
  sets a twin link to -1 when the partner is outside the closure.
