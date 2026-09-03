# Interpreting results

Each scenario produces one atlas, `results/{folder}/{scenario}/plots/atlas.html`.
Each folder produces a validation atlas, `results/{folder}/plots/atlas.html`.
The checks below name the figures to open first and what a healthy scenario
shows in each. The [Plot catalog](plot-catalog.md) lists every figure with
its caption.

[Output structure](output-structure.md) describes the files behind the
plots. To rebuild an atlas, follow
[Running the pipeline, Rebuild one output](running-the-pipeline.md#rebuild-one-output).

## Check a scenario atlas

Open the figures below in this order. Each name is the file basename in the
scenario `plots/` directory.

1. **`pedigree_counts.ped` and `family_structure`.** Confirm that the
   relationship pair counts and family sizes match what you configured. A
   scenario with dropout or `N_sample` shows the reduced counts in
   `pedigree_counts`, the phenotyped-population version.
2. **`heritability.by_generation`.** Confirm that the blue and orange points
   sit on their dashed lines in every generation. A drift across generations means that A
   and C move away from their configured values.
3. **`cross_trait`.** Confirm that the Pearson r in each panel matches the
   configured `rA`, `rC`, and `rE`. Confirm that the liability panel matches
   their variance-weighted sum.
4. **`liability_violin.phenotype`.** Confirm that the affected half of each
   violin sits above the unaffected half. If the two halves overlap fully, the phenotype
   model is not using liability.
5. **`cumulative_incidence.phenotype` and `censoring`.** Confirm that the
   observed curve lies below the true curve. The censoring percentages in
   the text box account for the gap. A gap that grows in the oldest generations
   comes from death censoring. A gap only in the youngest generation comes
   from the right end of its observation window.
6. **`tetrachoric.phenotype`.** Confirm that the violins sit below the black
   dashed liability correlation and above zero. Confirm that the
   relationship types keep the same rank order as their dashed lines. The gap between a violin and its dashed line is how
   much censoring and dichotomization lower the correlation. The green dash-dot line, when
   present, shows how much of that gap is censoring alone.

## Check a validation atlas

The validation atlas compares every scenario in a folder to its configured or
theoretical expectation. In each strip plot, blue dots are per-replicate
observed values and orange dashes are the expected value.

1. **`variance_components`.** Confirm that every dot sits on its dash. Check
   this figure first after changing the simulator.
2. **`correlations_A` and `correlations_phenotype`.** Confirm that the MZ,
   full-sibling, half-sibling, and parent-offspring correlations match
   `2 * kinship * A`, plus C where the pair shares a household. Under assortative mating, read
   the purple marker instead of the orange dash.
3. **`summary_bias`.** Confirm that every strip is centered on the red zero
   line. A strip
   that sits to one side is the scenario to investigate.
4. **`runtime` and `memory`.** Confirm that both scale roughly linearly with
   N on the log-log axes. A point off the line is a scenario worth profiling.
