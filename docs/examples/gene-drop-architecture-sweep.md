# Gene-drop architecture sweep

How does the genetic architecture (`alpha` and `num_causal`) shape the
realized $A$ distribution under [gene drop](../concepts/gene-drop.md)? This
example walks through six scenarios that vary one architecture knob at a
time against a fixed pedigree+drop, showing both the raw genetic-value
distributions and what they look like after the `tstrait_augment_pedigree`
rescale that puts them on the same scale as Gaussian-$A$ scenarios.

## Scenarios

All six share an identical pedigree+drop via `drop_from: p2_full`:

- **N = 35,951** founders (the full SimHumanity p2 panel)
- **G_ped = 10**, **G_pheno = 3** (107,853 sample inds across the latest 3 generations)
- Variance composition **A=0.4, C=0.1, E=0.5** (so $h^2 = 0.4$) for both traits
- No assortative mating, no MZ twins
- Same trait1 and trait2 phenotype model (frailty / weibull, scale=2160, rho=0.8, beta=1.0, prevalence=0.10)

The variants differ only in the tstrait architecture:

| Scenario | `alpha` | `num_causal` | What it explores |
|---|---:|---:|---|
| `p2_full` | -0.5 | 1,000 | anchor (defaults) |
| `p2_full_a0` | **0.0** | 1,000 | no MAF dependence in effect sizes |
| `p2_full_c100` | -0.5 | **100** | very sparse architecture |
| `p2_full_c10k` | -0.5 | **10,000** | moderate polygenicity |
| `p2_full_c100k` | -0.5 | **100,000** | high polygenicity |
| `p2_full_c1m` | -0.5 | **1,000,000** | near-infinitesimal |

Defined in `config/genotype_drop.yaml`. The variants use YAML anchors to
inherit everything from `p2_full` and override only the tstrait knobs.

## Run

The drop+graft step (`simulate_genotypes_chrom`) only runs once for
`p2_full`; the variants reuse those trees:

```bash
# Drop the founders through the simACE pedigree (one-shot, ~5 min wall at -j 8)
snakemake --use-conda --cores 8 \
  results/genotype_drop/p2_full/rep1/.simulate_genotypes.done

# Run all 5 variants (each only the tstrait sub-pipeline + augment + atlas)
snakemake --use-conda --rerun-triggers mtime -j 8 \
  results/genotype_drop/p2_full_a0/plots/atlas.pdf \
  results/genotype_drop/p2_full_c100/plots/atlas.pdf \
  results/genotype_drop/p2_full_c10k/plots/atlas.pdf \
  results/genotype_drop/p2_full_c100k/plots/atlas.pdf \
  results/genotype_drop/p2_full_c1m/plots/atlas.pdf
```

## Raw GV variance scales linearly with `num_causal`

Before the rescale, the variance of the genome-wide GV is roughly the sum
of per-site contributions, $\sigma_\beta^2 \cdot \sum_i [2 p_i (1 - p_i)]^{2\alpha + 1}$.
With $\sigma_\beta = 1$ and identically-distributed AFs, this scales linearly
in `num_causal`:

| Scenario | n_causal | realized var(GV) | scale factor (to var=0.4) |
|---|---:|---:|---:|
| `p2_full_c100` | 100 | 98 | 6.4e-2 |
| `p2_full` | 1,000 | 966 | 2.0e-2 |
| `p2_full_c10k` | 10,000 | 10,059 | 6.3e-3 |
| `p2_full_c100k` | 100,000 | 97,629 | 2.0e-3 |
| `p2_full_c1m` | 1,000,000 | 1,018,392 | 6.3e-4 |

Each 10× in `num_causal` gives ~10× in raw var(GV) and ~$\sqrt{10}^{-1}$ in
the rescale factor.

## `alpha` amplifies rare-allele contribution

At fixed `num_causal=1000`, switching from $\alpha=0$ (no MAF dependence)
to $\alpha=-0.5$ (LDAK-thin: rare alleles get larger effects) inflates raw
var(GV) about 4-fold:

| Scenario | alpha | realized var(GV) |
|---|---:|---:|
| `p2_full_a0` | 0.0 | 219 |
| `p2_full` | -0.5 | 966 |

The same 1000 sites and effects, weighted differently across the AF
spectrum.

## After rescale, all six are on the same scale

The augment rule centers each scenario's GV at zero and rescales sample
variance to exactly the configured `A1`:

| Scenario | var(A1) before (parametric) | var(A1) after (gene drop) | $h^2$ realized |
|---|---:|---:|---:|
| `p2_full_a0` | 0.4 (Gaussian draw) | **0.4000** | 0.3999 |
| `p2_full_c100` | 0.4 | **0.4000** | 0.3999 |
| `p2_full` | 0.4 | **0.4000** | (anchor) |
| `p2_full_c10k` | 0.4 | **0.4000** | 0.3999 |
| `p2_full_c100k` | 0.4 | **0.4000** | 0.3999 |
| `p2_full_c1m` | 0.4 | **0.4000** | 0.3999 |

The realized $h^2$ matches $A_1 / (A_1 + C_1 + E_1) = 0.4 / 1.0 = 0.4$
exactly — the tstrait `sim_env` step targets that derived value.

## What's left to compare in the atlas

After the augment, every scenario's `pedigree.full.tstrait.parquet` has
$A_1$ with var = 0.4. What still differs between scenarios is the
**distribution shape** of $A_1$ across the sample population: a 100-causal
architecture has fatter tails (a few large-effect carriers dominate) than a
1M-causal architecture (which converges to Gaussian by CLT).

The per-scenario atlas (`results/genotype_drop/{scenario}/plots/atlas.pdf`)
runs the full simACE phenotype → censor → sample → stats chain on the
augmented pedigree, so the kinship-based heritability estimates,
relative-pair correlations, and prevalence plots all reflect the gene-drop
$A$.

Compare the atlases pairwise to see how the architecture affects:

- **Phenotype prevalence per generation** — should be identical to within
  sampling noise, since the rescale puts every $A_1$ at the same variance
  and the threshold is calibrated against unit-variance liability.
- **Heritability estimates** (Falconer, regression-on-relatives) — should
  also be ~0.4 across scenarios, but estimator variance may differ as a
  function of the underlying architecture (sparse architectures with rare
  large-effect variants are harder to estimate).
- **Relative-pair correlations vs theoretical kinship** — the line should
  pass through the expected slope $h^2 = 0.4$ in all cases.

If a method's heritability estimate degrades under sparse architectures,
this sweep is what surfaces it.

## Cost

On the local box (single workstation):

- Pre-pipeline (one-shot): ~10 min for `tskit_preprocess`, ~10 min for
  `site_catalog`. Reusable across all gene-drop scenarios forever.
- `simulate_genotypes_chrom` × 22: ~2 hr CPU total, ~30 min wall at -j 8
  (chr1 alone takes ~16 min on one core). Done once for `p2_full`; variants
  reuse via `drop_from`.
- Per variant: tstrait sub-pipeline + augment + phenotype + atlas — a few
  minutes each, dominated by `tstrait_gv_chrom` for `c1m` (~50 s/chrom × 22 / 8 cores ≈ 2-3 min wall).
