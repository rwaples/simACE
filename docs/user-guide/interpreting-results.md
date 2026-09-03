# Interpreting results

Each scenario produces one atlas, `results/{folder}/{scenario}/plots/atlas.html`,
and each folder produces a validation atlas, `results/{folder}/plots/atlas.html`.
This page tells you which figures to check first and what a healthy scenario
looks like. The [Plot catalog](plot-catalog.md) lists every figure with its
caption.

The analyze stage computes the statistics behind the plots once per replicate
and stores them in `report.yaml` and `plot_payload.yaml`. The plotting rules
read those files and a downsampled `plotting_sample.parquet`, never the full
trait files.

To rebuild an atlas after a plotting change, run:

```bash
pixi run snakemake --cores 4 -f results/{folder}/{scenario}/plots/atlas.html
```

## Check a scenario atlas

Open the six figures below in this order. Each name is the file basename in
the scenario `plots/` directory.

1. **`pedigree_counts.ped` and `family_structure`.** Confirm that the
   relationship pair counts and family sizes match what you configured. A
   scenario with dropout or `N_sample` shows the reduced counts in
   `pedigree_counts`, the phenotyped-population version.
2. **`heritability.by_generation`.** The blue and orange points sit on their
   dashed lines in every generation. A drift across generations means the
   simulation is not holding the configured A and C.
3. **`cross_trait`.** The Pearson r in each panel matches the configured
   `rA`, `rC`, and `rE`. The liability panel matches their variance-weighted
   sum.
4. **`liability_violin.phenotype`.** The affected half of each violin sits
   above the unaffected half. If the two halves overlap fully, the phenotype
   model is not using liability.
5. **`cumulative_incidence.phenotype` and `censoring`.** The observed curve
   lies below the true curve, and the gap is explained by the censoring
   percentages in the text box. A gap that grows in the oldest generations
   comes from death censoring. A gap only in the youngest generation comes
   from the right end of its observation window.
6. **`tetrachoric.phenotype`.** The violins sit below the black dashed
   liability correlation and above zero, in the same rank order across
   relationship types. The gap between a violin and its dashed line is the
   cost of censoring plus dichotomization. The green dash-dot line, when
   present, shows how much of that gap is censoring alone.

## Check a validation atlas

The validation atlas compares every scenario in a folder to its configured or
theoretical expectation. In each strip plot, blue dots are per-replicate
observed values and orange dashes are the expected value.

1. **`variance_components`.** Every dot sits on its dash. This is the first
   thing to look at after changing the simulator.
2. **`correlations_A` and `correlations_phenotype`.** MZ, full-sibling,
   half-sibling, and parent-offspring correlations match `2 * kinship * A`
   plus C where the pair shares a household. Under assortative mating, read
   the purple marker instead of the orange dash.
3. **`summary_bias`.** Every strip is centred on the red zero line. A strip
   that sits to one side is the scenario to investigate.
4. **`runtime` and `memory`.** Both scale roughly linearly with N on the
   log-log axes. A point off the line is a scenario worth profiling.

## Read the binary trait

Case status lives in `trait.parquet` as `affected1` and `affected2`. The
phenotype model and the censoring stage together decide it. For a trait that
is a pure liability threshold, choose the `simple_ltm` model, described in
[Phenotype models](phenotype-models.md). fitACE computes the descriptive
statistics on case status, meaning prevalence by generation, tetrachoric
correlations, and Falconer heritability, from `trait.parquet`.
