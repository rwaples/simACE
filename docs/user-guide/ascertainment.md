# Ascertainment

The ascertainment stage models an incomplete registry and a case-enriched
study sample. It runs after the censor stage. Under `simace run` it is the
last phase of the `cohort` stage, and its result is the replicate's
`cohort.parquet`, which the analyze stage and fitACE read beside
`pedigree.parquet`.
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

The drawn individuals are the analysis sample. The analysis pedigree is the
ancestor closure of those IDs within the post-dropout pedigree: the drawn
individuals plus every ancestor reachable through intact parent links. Links
that leave the closure are again set to -1.

Ascertainment does not change validation. Validation reads
`pedigree.parquet`, the recorded pedigree, which ascertainment never
rewrites.

## Who is in `cohort.parquet`

`cohort.parquet` has one row per member of the analysis pedigree, in the
row order of `pedigree.parquet`. Its columns are `id` and the censored
outcome columns ([Output structure](output-structure.md#cohortparquet)). A
person enters it by one of three routes.

| Route | Who | Outcome columns | How to tell |
|---|---|---|---|
| A. Drawn | Survived dropout, is in the trailing `G_pheno` generations, and was kept by the draw | Set. `t1` and `t2` may still be null ([ADR 0019](../adr/0019-null-raw-onset-censoring-semantics.md)) | `affected1` is not null |
| B. Ancestor, never phenotyped | Not drawn. An ancestor of a drawn person through intact parent links, in a generation older than the phenotyped window | All null | `affected1` is null and `generation` is outside the phenotyped window |
| C. Ancestor, phenotyped but not drawn | Not drawn. An ancestor of a drawn person, inside the phenotyped window, so the stage computed outcomes for them | All null, although outcomes were computed | `affected1` is null and `generation` is inside the phenotyped window |

A person who is both drawn and an ancestor of someone drawn is route A.
`generation` is in `pedigree.parquet`, and the phenotyped window is the last
`G_pheno` generations of the recorded pedigree. Route C occurs only when the
draw is not pass-through and `G_pheno` is at least 2. The file alone cannot
separate route C from route B; this is the information that ascertainment
discards ([ADR 0021](../adr/0021-two-canonical-replicate-parquets.md)).

These people are not in `cohort.parquet`:

- Dropped individuals, in any generation, even when they are ancestors of
  drawn people. The ancestor closure cannot cross them.
- Phenotyped individuals who were not drawn and are not ancestors of anyone
  drawn.
- Unphenotyped individuals who are not ancestors of anyone drawn, such as
  childless members of early generations.

`simace.core.cohort.selected_views(pedigree, cohort)` rebuilds the two
analysis frames. The analysis sample is the rows with `affected1` not null.
The analysis pedigree is the rows of `pedigree.parquet` whose `id` is in
`cohort.parquet`, with each `mother`, `father`, or `twin` link that points
outside the cohort set to -1. A link of -1 in the analysis pedigree means one
of the following:

- The referent is unknown in the recorded pedigree, for example a parent of
  generation 0.
- The referent was dropped. This is the only way a parent link is cut, and a
  row can keep one parent and lose the other.
- The referent is outside the closure. This applies to twins only: a twin
  who was neither drawn nor an ancestor of anyone drawn. Parent links are
  never cut this way, because the closure follows every intact parent link.

`pedigree.parquet` keeps the recorded links, so comparing a row in the two
pedigrees tells which case applies.

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
