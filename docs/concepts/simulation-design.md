# Simulation design

## Multi-generational pedigree

The simulation runs `G_sim` generations:

- The first `G_sim - G_ped` generations are burn-in. They are simulated but not recorded.
- The last `G_ped` generations are recorded in the pedigree.
- The last `G_pheno` of those are phenotyped.

Each generation has `N` individuals. At the defaults, $N = 100{,}000$ and $G_{ped} = 6$, the recorded pedigree holds $600{,}000$ individuals.

## Mating and reproduction

In each generation the simulation forms couples from the parent pool under these rules:

- An individual may belong to more than one couple.
- Males and females pair at random by default, or assortatively on liability through `assort1` and `assort2`.
- A multinomial draw distributes the `N` offspring across the couples.
- Population size stays constant, so some couples have no offspring.
- MZ twins are assigned only to couples with two or more offspring.

At the default `mating_lambda = 0.5`, about 77% of individuals have one partner and 23% have two or more. That produces full sibs, maternal half-sibs, and paternal half-sibs in one pedigree.

## Pedigree relationship types

`PedigreeGraph`, from the [`pedigree-graph`](https://github.com/rwaples/pedigree-graph) package, uses sparse matrix algebra to extract 23 relationship categories from a pedigree. Each category is defined by three numbers: `up`, the meioses from individual A up to the common ancestor, `down`, the meioses from that ancestor to individual B, and `n_ancestors`, which is 1 for a half or lineal link and 2 for a link through a mated pair. Kinship is $n_{\text{ancestors}} \times (1/2)^{(\text{up} + \text{down} + 1)}$.

| Code | Label | Up | Down | Ancestors | Kinship | Degree |
|------|-------|---:|-----:|----------:|--------:|-------:|
| MZ | MZ twin | — | — | — | 1/2 | 0 |
| MO | Mother-offspring | 1 | 0 | 1 | 1/4 | 1 |
| FO | Father-offspring | 1 | 0 | 1 | 1/4 | 1 |
| FS | Full sib | 1 | 1 | 2 | 1/4 | 1 |
| MHS | Maternal half sib | 1 | 1 | 1 | 1/8 | 2 |
| PHS | Paternal half sib | 1 | 1 | 1 | 1/8 | 2 |
| GP | Grandparent | 2 | 0 | 1 | 1/8 | 2 |
| Av | Avuncular | 1 | 2 | 2 | 1/8 | 2 |
| GGP | Great-grandparent | 3 | 0 | 1 | 1/16 | 3 |
| HAv | Half-avuncular | 1 | 2 | 1 | 1/16 | 3 |
| GAv | Great-avuncular | 1 | 3 | 2 | 1/16 | 3 |
| 1C | 1st cousin | 2 | 2 | 2 | 1/16 | 3 |
| GGGP | Great²-grandparent | 4 | 0 | 1 | 1/32 | 4 |
| HGAv | Half-great-avuncular | 1 | 3 | 1 | 1/32 | 4 |
| GGAv | Great²-avuncular | 1 | 4 | 2 | 1/32 | 4 |
| H1C | Half-1st-cousin | 2 | 2 | 1 | 1/32 | 4 |
| 1C1R | 1st cousin 1R | 2 | 3 | 2 | 1/32 | 4 |
| G3GP | Great³-grandparent | 5 | 0 | 1 | 1/64 | 5 |
| HGGAv | Half-great²-avuncular | 1 | 4 | 1 | 1/64 | 5 |
| G3Av | Great³-avuncular | 1 | 5 | 2 | 1/64 | 5 |
| H1C1R | Half-1st-cousin 1R | 2 | 3 | 1 | 1/64 | 5 |
| 1C2R | 1st cousin 2R | 2 | 4 | 2 | 1/64 | 5 |
| 2C | 2nd cousin | 3 | 3 | 2 | 1/64 | 5 |

The `max_degree` parameter sets how deep extraction goes. The default is 3, which reaches 1st cousins. The cutoff follows the Degree column exactly. Degree 2 stops at half-sibs, grandparents, and avuncular pairs. Degree 3 adds 1st cousins and the other degree-3 categories. Degree 5 reaches 2nd cousins. The registry is importable as `REL_REGISTRY` and `PAIR_KINSHIP` from `pedigree_graph`.

### Inbreeding and exact kinship

By default each relationship pair carries the nominal kinship from the table above. That value assumes one relationship path and no inbreeding. When a scenario sets `estimate_inbreeding: true`, `PedigreeGraph` reports exact values through two methods.

`compute_inbreeding()` returns each individual's inbreeding coefficient `F` by the Meuwissen and Luo ancestor walk in `_compute_F_meuwissen_luo`. It does not build the kinship matrix. It treats MZ twins as full sibs. In a pedigree with no consanguineous matings every `F` is 0.

`compute_pair_kinship(pairs)` returns the exact kinship of each requested pair by a memoised recurrence in `pedigree_graph._kinship_pairwise`. It computes only the requested pairs and never builds the $n \times n$ matrix. If a `kinship_matrix(0.0)` result is already cached it reads from that instead. Exact kinship can exceed the nominal value for three reasons: inbreeding, MZ co-coalescence, and multiple relationship paths. Double first cousins, for example, have kinship 0.125, twice the nominal first-cousin 0.0625. The recurrence derives `F` as the kinship of the parents, which is how the matrix build does it, so this `F` is MZ-aware. There is no nominal fast path. See pedigree-graph ADR 0005.

The recurrence costs about $O(\text{requested pairs} + \text{distinct ancestor pairs reached})$. The full matrix build it replaced materialised a near-dense $K$ and ran out of memory on large pedigrees. The worst case is $O(P \cdot A^2)$ in the number of distinct ancestors $A$. Deeply inbred pedigrees, and pedigrees with heavy ancestor overlap, fall outside that scaling guarantee.

### Per-generation mean kinship without K

The coancestry-rate Ne estimator, `ne_coancestry` in `pedigree_graph`, needs only the mean kinship of each generation, $\bar\theta_g$. It does not need the sparse kinship matrix $K$. `PedigreeGraph.per_gen_mean_kinship(min_kinship=0.0)` streams $\bar\theta_g$ from the kinship recurrence without building $K$'s compressed sparse column arrays. `compute_all_ne` takes this path whenever the estimator runs, which is when a scenario sets `analysis.skip_ne_coancestry: false`.

The streaming pass walks each row of the recurrence's storage, whose columns are sorted ascending. It counts each unordered same-generation non-twin pair once, where the row index is below the column index. It accumulates one float64 sum per generation.

Building $K$ stops working at a few million individuals with `G_ped=6`, because the nonzero count in `_assemble_csc` overflows int32. The streaming pass never holds the full matrix, so it scales further. `analysis.skip_ne_coancestry` defaults to `true`, which skips the coancestry estimator and its recurrence so only the seven other Ne estimators run. Set it to `false` on a scenario whose pedigree is small enough to afford the recurrence.

## Pipeline stages

The pipeline has four simulation stages:

1. **Simulate**: build the multi-generational pedigree with ACE liability components.
2. **Phenotype**: map liability to affected status and age at onset.
3. **Censor**: apply age-window censoring and competing-risk death censoring.
4. **Ascertainment**: drop individuals at random, then draw a case-weighted sample of size `N_sample` ([ADR 0001](../adr/0001-unified-ascertainment-stage.md)).

Three analysis stages follow: validation, which reads the full pre-ascertainment pedigree, summary statistics, and plotting. Model fitting runs in fitACE.

## Pipeline rule graph

The Snakemake rule graph shows which rules feed which:

![simACE rule graph](../images/rulegraph.png)

After you change the Snakefile or a rule file, regenerate the image:

```bash
scripts/regen_rulegraph.sh
```

The script writes `docs/images/rulegraph.png`. To render a different sub-DAG, pass its target as the first argument. The default target is `results/test/small_test/scenario.done`.
