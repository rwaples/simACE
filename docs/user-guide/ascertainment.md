# Ascertainment

Real registries do not hold the whole population, and case-control studies
oversample cases. The ascertainment stage reproduces both effects. It runs
after censoring and writes the `pedigree.parquet` and `trait.parquet` that the
analyze stage and fitACE read. [ADR 0001](../adr/0001-unified-ascertainment-stage.md)
records why the two effects share one stage.

The stage takes three parameters, all under `ascertainment:` in the scenario
config. [Configuration](configuration.md#ascertainment-and-analysis) lists them
with their defaults. The worked example
[When the study sample is not the population](../examples/ascertainment-bias.md)
shows what they do to the estimates.

## Why two steps on IDs

The stage removes individuals in two steps, and both steps act on individual
IDs rather than on sampling weights. A single weighted draw of fixed size
cannot express uniform dropout: the dropout weight would cancel out of the
normalised probabilities.

1. **Dropout.** The stage removes `round(N_total * dropout_rate)` individuals
   uniformly at random from the full pedigree. Any `mother`, `father`, or
   `twin` reference to a removed individual becomes -1.
2. **Case-weighted draw.** From the post-dropout phenotyped rows, the stage
   draws `N_sample` individuals without replacement. A case, meaning
   `affected1` is true, has weight `case_ascertainment_ratio`. A control has
   weight 1. If `N_sample` is 0 or at least the pool size, every individual
   passes through.

The drawn IDs become the rows of `trait.parquet`. The pedigree output is the
ancestor closure of those IDs within the post-dropout pedigree, again with
dangling references set to -1.

Validation is not affected. It reads `pedigree.full.parquet`, the pedigree
before ascertainment.

## What dropout breaks

Dropout ignores trait status, sex, generation, and pedigree position. It
models an incomplete registry.

Because dropout severs parent and twin pointers, any relationship that passes
through a removed individual disappears. A grandparent and grandchild whose
connecting parent was dropped are unrelated in the output. Full siblings whose
mother was dropped become paternal half-siblings, because only the father link
survives.

`config/ascertainment.yaml` defines three ready-made scenarios,
`baseline100K_dropout10`, `baseline100K_dropout30`, and
`baseline100K_dropout50`, at 10, 30, and 50 percent dropout.

## What case weighting does

With 10 percent prevalence and a ratio of 5, a case is five times as likely
to be drawn as a control, and about 36 percent of the sample are cases.

The stage handles the edge cases as follows.

- A ratio of 1, the default, is a uniform draw.
- A ratio of 0 draws controls only. If fewer controls exist than `N_sample`,
  the stage lowers `N_sample` to the number of controls and logs a warning. If
  the pool holds no controls at all, the stage raises an error instead of
  drawing nobody.
- If the pool holds no cases, or only cases, the ratio has no effect. The stage
  logs a warning and draws uniformly.
- If `N_sample` is 0, no draw happens, so the ratio has no effect. The stage
  logs a warning when the ratio is not 1.

The analyze stage copies the ratio into `report.yaml` under
`inputs.ascertainment`. It applies no correction to any estimate. The point of
the stage is to measure the bias, not to remove it.

## Which relationships survive

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
  severs a twin pointer whose partner is outside the closure.
