# Gene drop and tstrait phenotyping

The standard simACE pipeline draws additive genetic values $A$ from a Gaussian under an infinitesimal model. The gene-drop pipeline replaces that parametric $A$ with a genetic value computed from genotypes. It drops an ancestral tree sequence from SimHumanity through a simACE pedigree, sums causal-site effects per individual, and rescales the result to the configured $A$ variance. The phenotype, sampling, and plotting stages then run unchanged.

This page explains the design. The worked sweep over `alpha` and `num_causal` is in [Architecture sweep](../examples/gene-drop-architecture-sweep.md). The config keys are in [Configuration](../user-guide/configuration.md#gene-drop-with-tstrait).

## Why gene drop

The Gaussian $A$ model treats trait genetics as a sum of infinitely many tiny effects. The resulting distribution is smooth and says nothing about how heritability arises from a finite set of causal variants.

Gene drop simulates that mechanism:

1. Founder haplotypes carry SLiM-derived mutations from the SimHumanity ancestral simulation.
2. Mutations pass through the simACE pedigree by Mendelian inheritance, with recombination under the HapMapII GRCh38 rate map.
3. A finite subset of those mutations is declared causal. Each effect size is drawn from $\mathcal{N}(0, \sigma_\beta^2)$ and multiplied by $[2p(1-p)]^{\alpha}$. With $\alpha=-0.5$ this is the LDAK-thin form of Speed et al.
4. Each individual's genetic value is the sum of dosage times effect across causal sites.

The shape and tails of the realised $A$ distribution then follow from the genetic architecture. That is what you want when stress-testing methods that estimate variance components from pedigree data.

After computing $A$, the pipeline rescales it to match the configured `A1` variance. The phenotype models then see the same scale as a Gaussian $A$ run, and every downstream plot compares the two directly.

## Pipeline

The gene-drop branch sits between pedigree simulation and the phenotype stage. It has one preprocessing pass, run once per SimHumanity dataset, and four per-replicate steps:

```
Preprocessing (one-shot, scenario-independent)
──────────────────────────────────────────────
SimHumanity p2 .trees ──► tskit_preprocess_canonicalize_chrom (×22)
                                        │
                                        ▼
                          tskit_preprocess_concat
                                        │
                                        ▼
                          tstrait_site_catalog_chrom (×22)
                                        │
                                        ▼
                          tstrait_site_catalog_concat
                                        │
                                        ▼
                          <preprocessed>/site_catalog.parquet


Per replicate
─────────────
simulate_pedigree_liability ──► pedigree.full.parquet
              │
              └──► simulate_genotypes_chrom (×22)  ──► genotypes_chrom_{n}.trees
                            │
                            ▼
                  tstrait_assign_effects_{per_rep,shared}  ──► causal_effects.parquet
                            │
                            ▼
                  tstrait_gv_chrom (×22)  ──► gv_chrom_{n}.parquet
                            │
                            ▼
                  tstrait_augment_pedigree  ──► pedigree.full.tstrait.parquet
                            │
                            ▼
                  pedigree_dropout (reads .tstrait. when use_gene_drop=true)
                            │
                            ▼
                  phenotype → censor → ascertainment → stats → plot → atlas
```

### Step 1: preprocess the SimHumanity ancestry

The `tskit_preprocess` rules canonicalise the per-chromosome `.trees` files from SimHumanity and concatenate them. Canonicalising keeps one mutation per site and sorts samples into a fixed order. The `tstrait_site_catalog` rules then build the site catalog, `site_catalog.parquet`, which records the founder allele frequency of every site. Every gene-drop scenario reads that one catalog.

The preprocessing rules need the dedicated tskit conda environment. Run them once:

```bash
pixi run snakemake --use-conda --cores 4 tskit_preprocess
```

### Step 2: drop founders through the simACE pedigree

`simulate_genotypes_chrom` builds an msprime fixed pedigree from the simACE pedigree, tagging the latest `G_pheno` generations as samples. It runs `msprime.sim_ancestry(model="fixed_pedigree")` with the per-chromosome SimHumanity recombination map. It then grafts the dropped tree onto the preprocessed ancestral tree at the founder generation. The result is a chromosome-scale tree sequence whose present-day samples carry alleles inherited from the SimHumanity founder haplotypes.

This is the slowest step. To reuse one set of drops across several architectures, see [Sharing drops across variants](#sharing-drops-across-variants).

### Step 3: assign causal effects and compute genetic values

`tstrait_assign_effects_per_rep`, or `tstrait_assign_effects_shared` when `share_architecture` is true, applies the MAF filter to the site catalog and samples `num_causal` sites from what remains. If `frac_causal` is set instead, it samples that fraction of the eligible sites. For each site it draws a raw effect $\beta \sim \mathcal{N}(\mu, \sigma_\beta^2)$ and multiplies by $[2p(1-p)]^{\alpha}$.

`tstrait_gv_chrom` computes each individual's genetic value with a single-pass kernel compiled by numba on `tskit.jit.numba.NumbaTreeSequence`. The kernel walks each chromosome's trees once, keeps `left_child` and `right_sib` arrays up to date as it goes, and for each causal site does a depth-first walk from the mutation node to add the effect to every descendant sample. Its output matches `tstrait.genetic_value` up to floating-point summation order.

Only sample-tagged individuals, the generations covered by `G_pheno`, get a genetic value. Older generations have no sample nodes in the dropped tree.

### Step 4: rescale and overwrite A in the pedigree

`tstrait_augment_pedigree` reads `pedigree.full.parquet` and sums the per-chromosome genetic values into one genome-wide value per sample individual. It then:

1. Centres the values at zero.
2. Rescales them so that $\mathrm{Var}(A_\text{new})$ equals the configured `A1`. The variance is the sample variance with `ddof=0`.
3. Overwrites the `A1` column for sample individuals.
4. Recomputes `liability1` as $A_\text{new} + C + E$.
5. Writes `pedigree.full.tstrait.parquet` next to the original. The original is unchanged.

Older ancestors keep their parametric `A1`. That asymmetry is deliberate. Only sample-generation individuals reach the downstream model fits that read the phenotype.

`A2` is untouched because gene drop is single-trait. Sharing the gene-drop $A$ between both traits would need an extension to the augment script.

### Step 5: feed the standard pipeline

When a scenario sets `use_gene_drop: true`, the `pedigree_dropout` rule reads `pedigree.full.tstrait.parquet` instead of `pedigree.full.parquet`. The phenotype model, censoring, ascertainment, stats, plots, and atlas all run as they would for a Gaussian $A$ scenario, on the gene-drop $A$ column.

## Sharing drops across variants

The drop and graft step is expensive. To vary the architecture while holding the genotypes fixed, set `drop_from: <base_scenario>` on the variant scenario. The architecture keys are `tstrait.alpha`, `tstrait.num_causal`, and the other keys under `tstrait`. The `tstrait_gv_chrom` and `tstrait_augment_pedigree` rules then read the base scenario's `.trees` and `pedigree.full.parquet`. Only the tstrait steps run per variant.

A six-scenario sweep over `num_causal` in {100, 1k, 10k, 100k, 1m} and `alpha` in {0, -0.5} therefore runs the drop once instead of six times.

## Heritability

There is no `tstrait.h2` config key. The pipeline derives $h^2$ from the standard simACE variance components:

$$h^2 = \frac{A_1}{A_1 + C_1 + E_1}$$

Gene-drop scenarios therefore sit on the same heritability scale as Gaussian $A$ scenarios. The `h2_realized` field in `tstrait_phenotype_meta.json` reports how close each replicate came to that value.

## Comparison

| | Gaussian $A$ (standard simACE) | Gene drop |
|---|---|---|
| $A$ comes from | $\mathcal{N}(0, A_1)$ per individual | sum of dosage times effect over causal sites, rescaled to $A_1$ |
| Architecture | infinitesimal | finite causal variants under an $\alpha$-MAF model |
| Cost per replicate | seconds | minutes, dominated by the drop and graft step |
| Scale | same by construction | same after the rescale step |
| Multi-trait | yes, trait1 and trait2 | trait1 only. trait2 stays parametric |
| Parameters | `A1`, `C1`, `E1`, `assort1`, and the other pedigree keys | the same, plus the `tstrait` block, `drop_from`, and `use_gene_drop` |
