# simACE

simACE simulates multi-generational pedigrees with **A** (additive genetic), **C** (common environment), and **E** (unique environment) variance components, then runs phenotype → censor → ascertainment → analyze → plot. It exists as a controlled testbed for evaluating family-study and twin-study statistical methods, where the ground truth is known.

This file is a disambiguation glossary for AI agents and future maintainers — it picks canonical names for concepts that have competing aliases in the code, docs, and conversation. It is **not** a methods write-up (see `docs/concepts/methods.md` for that) and contains no implementation details.

## Language

### Variance-component model

**Liability** ($L$):
The continuous per-individual, per-trait quantity $L = A + C + E$. May refer to either the raw value (as stored in `liability1` / `liability2` parquet columns) or its standardized form ($\tilde L$, produced by the `standardize` step). Which one is meant is determined by context — see Flagged ambiguities.
_Avoid_: phenotype (means something else), score, risk score, trait value, latent.

**Additive genetic component** ($A$):
The heritable component of liability, following the infinitesimal model. Transmitted parent → offspring as midparent average plus Mendelian sampling noise. Stored in `A1` / `A2` columns.
_Avoid_: genetic value, breeding value, polygenic score, heritability (heritability is the *variance proportion* $\sigma_A^2$, not the component itself).

**Common environment component** ($C$):
The per-household environmental effect shared by all offspring of the same mother. Drawn fresh per household per generation; **not** transmitted from parental $C$. Stored in `C1` / `C2` columns.
_Avoid_: shared environment, household effect, family effect, sibship effect, maternal effect. ("Household" is the implementation primitive that holds $C$ — see the **Household** entry — but $C$ the component is named "common environment".)

**Unique environment component** ($E$):
The per-individual residual component, independent across siblings, twins, and traits. Stored in `E1` / `E2` columns.
_Avoid_: residual, noise, individual environment, idiosyncratic component, error.

**Trait**:
The canonical word for *both* (a) one of the two correlated axes the simulation models jointly, and (b) the per-individual, per-trait observable result of phenotype-model evaluation (binary affection status + age-at-onset where applicable). **Always exactly two traits** — this is a hard invariant of simACE, baked into the cross-trait correlation structure (`rA`, `rC`, `assort1`, `assort2`). Identified numerically as **Trait 1** and **Trait 2** (in code: `trait1` / `trait2`). Each trait carries its own $A_k$, $C_k$, $E_k$, liability $L_k$, phenotype model, and observable outcome.
_Avoid_: outcome, label, response, target, disease, variable, "trait A" / "trait B" (indexing is numeric, never alphabetic). Note that "phenotype" is **not** an alias for trait — see **Phenotype model** / **Phenotype stage**.

### Phenotype

**Phenotype model**:
The model family that maps a trait's liability → its observable outcome. One of `frailty`, `cure_frailty`, `adult` (with `method: ltm` or `cox`), or `first_passage`. Each is a frozen dataclass subclassing `PhenotypeModel`. Configured per-trait under `phenotype.trait1.model` / `phenotype.trait2.model`.
_Avoid_: phenotype family, liability-to-onset map, hazard model (only a subset of families have a hazard step).

**Phenotype stage**:
The pipeline stage that runs each trait's phenotype model against the simulated pedigree to produce observable per-trait outcomes. Always the **noun form** — never "phenotyping". Lives under `simace/phenotype/` (package) and `workflow/rules/simace/phenotype.smk` (rule).
_Avoid_: phenotyping (gerund form is suppressed across the codebase — see Flagged ambiguities).

### Trait outcomes

**Affected** / **Unaffected**:
An individual's *ground-truth* binary status for a given trait. Stored as `affected1` / `affected2` (boolean). "Affected" is a truth about the individual produced by the phenotype model + censoring; it is not contingent on study enrollment.
_Avoid_: case (reserved for the sampling/ascertainment context — see **Case** / **Control**), diagnosed, positive, sick, affection, "affected status" (just say "affected"). (Caption exception: "true cases" is permitted for ground-truth affecteds at the **censoring** stage — see **Plot captions & annotations**.)

**Case** / **Control**:
Roles a sampled individual occupies in a study-design context: a **case** is an affected individual drawn into the sample; a **control** is an unaffected one. Only meaningful after the sample stage and only with case-ascertainment weighting in play. The split between "affected" (truth) and "case" (study role) is load-bearing because simACE exists to study how case ascertainment biases estimates relative to the underlying truth.
_Avoid_: using "case" / "control" for un-sampled individuals or for the raw `affected` column.

**Onset** (age-at-onset):
The age at which an affected individual experiences the trait event, as produced by time-to-event phenotype models (`frailty`, `cure_frailty`, `adult` with `method: cox`, `first_passage`) and clipped by censoring. Stored as `onset1` / `onset2`. "Onset" is canonical for column names and short-form references; "age-at-onset" is acceptable in prose.
_Avoid_: event time, time-to-event (these are *method* names, not column names), age of diagnosis, age of disease. (Caption exception: bare "event time" is permitted for the raw pre-censoring simulated time, distinct from the clipped `onset` column — see **Plot captions & annotations**.)

### Generations

**Generation** ($g$):
A discrete cohort of $N$ individuals in the simulation, indexed sequentially. Population size $N$ is constant across generations (hard invariant). Within the recorded pedigree, generations are zero-indexed (`generation` column in `pedigree.parquet`).
_Avoid_: cohort (ambiguous), age group (different concept).

**$G_{\text{sim}}$** (simulated generations):
Total number of generations the simulation runs *internally*, including burn-in. Config key: `G_sim`.

**$G_{\text{ped}}$** (recorded generations):
The trailing $G_{\text{ped}}$ generations written to `pedigree.parquet`. Earlier generations are burn-in (simulated but discarded). Config key: `G_ped`. Note $G_{\text{ped}} \leq G_{\text{sim}}$ and the difference is the burn-in length.

**$G_{\text{pheno}}$** (phenotyped generations):
The trailing $G_{\text{pheno}}$ generations *of the recorded pedigree* that receive phenotype-model evaluation and contribute observable trait outcomes. Config key: `G_pheno`. Note $G_{\text{pheno}} \leq G_{\text{ped}}$.

**Founder**:
An individual whose $A$, $C$, $E$ values are *drawn from scratch* rather than inherited from parents. Founders exist only at generation 0 of $G_{\text{sim}}$ — i.e., the very start of the internal simulation. **When burn-in is present ($G_{\text{sim}} > G_{\text{ped}}$), founders are upstream of the recorded pedigree and never appear in `pedigree.parquet`.** Generation 0 of the *recorded* pedigree is **not** a founder generation unless burn-in is zero.
_Avoid_: progenitor, ancestor, starting individuals, "generation 0" (only synonymous when burn-in = 0).

**Burn-in**:
The $G_{\text{sim}} - G_{\text{ped}}$ earliest generations of the simulation, run to randomize ancestry but not written to disk. Their purpose is to break dependence on the founder draw before recording begins.
_Avoid_: discarded generations, warmup, pre-recording.

**Pedigree-only generations**:
The earliest **recorded** generations that contribute pedigree structure (so cousins, grandparents, etc. are computable) but are *not* phenotyped. There are $G_{\text{ped}} - G_{\text{pheno}}$ such generations, sitting between the burn-in and the phenotyped generations.
_Avoid_: structure-only, unphenotyped (technically correct but less specific), background generations.

### Pedigree structure

**Pedigree**:
The full multi-generational graph stored in `pedigree.parquet`, covering all $G_{\text{ped}}$ recorded generations. "The pedigree" without qualification means the recorded pedigree (burn-in is *not* part of the pedigree from the consumer's perspective).
_Avoid_: family tree, genealogy.

**Mating**:
An event linking one male and one female parent, producing zero or more offspring. Under the **standard mating model** each individual may participate in multiple matings (mating count drawn from a zero-truncated Poisson). Under the **Wright-Fisher mating model** there is no persistent mating-pair structure: each offspring's two parents are drawn independently, so each offspring is conceptually its own degenerate one-offspring mating event. "Mating pair" is the same thing in prose.
_Avoid_: couple, partnership, union, pair (ambiguous — see **Relationship pair**).

**Household**:
The unit within which the common-environment component ($C$) is shared. **One household per mother**, not per mating pair: all offspring of the same mother — including maternal half-siblings (same mother, different father) — share a single household and a single $C$ draw. Paternal half-siblings (same father, different mother) do **not** share a household. Stored as `household_id`.
_Avoid_: family (overloaded), sibship (wrong cardinality — paternal half-sibs are siblings but not householdmates), home, nest.

**MZ twin** (monozygotic twin):
An offspring pair within the same mating sharing identical $A$ and identical sex. Always exactly 2 individuals per twin event (no triplets). Linked via the `twin` column (sentinel −1 = no twin). MZ twins automatically share a household (same mother) and share $C$ as well as $A$.
_Avoid_: identical twin (colloquial), DZ twin (**simACE has no concept of DZ twins** — same-mating non-MZ offspring are simply full siblings), twin (alone — always qualified as "MZ twin").

**Sibship**:
**Not a domain term in simACE.** Do not introduce it. The cardinality is wrong (paternal half-sibs would belong to a sibship but not a household, breaking the C-sharing intuition). Use **household**, **siblings**, or **mating** instead, depending on which structural fact you actually mean.

**Family size**:
The number of offspring per parent — counted per **mother** or per **father** — not household size and not pedigree "family" size. The `family_size` validation plot (mean offspring per mother/father among parents with ≥1 child) and the `family_size_variance` effective-size diagnostic both use this offspring-count sense. simACE has no household-size or pedigree-family-size concept; when a plot or stat says "family size" it means offspring count per parent.
_Avoid_: household size, sibship size, pedigree family size (all wrong cardinality — see **Household**, **Sibship**), kids per couple (see **Mating**).

### Prevalence and incidence

**Prevalence** ($K$):
The proportion of a population designated as affected under the phenotype model — i.e., the *lifetime / asymptotic* risk of being a case. Config key: `prevalence` inside `phenotype.trait{N}.params`, accepted as a scalar, per-generation dict, sex-specific dict, or sex × generation dict. Only meaningful for the threshold-bearing families (`adult`, `cure_frailty`); `frailty` and `first_passage` reject `prevalence` outright because case fraction emerges from the hazard for those families.
_Avoid_: rate (overloaded — could mean hazard), case rate, K-value (just say "prevalence" or "$K$"). Note that **incidence is not an alias** for prevalence — see below.

**Observed prevalence**:
The affected fraction visible in output data, after censoring (and after sampling, if applied). This is what a downstream model fit on the simulated output would estimate. May differ from the configured prevalence due to censoring (some events fall outside the age window or are pre-empted by death) and ascertainment.
_Avoid_: empirical prevalence (ambiguous), sample prevalence (reserve "sample" for explicit subsampling).

**Cumulative incidence** (CIF, cumulative incidence function):
The proportion of a cohort that has experienced the trait event by a given age. A function of age, not a single scalar — distinct from prevalence, which is the lifetime / asymptotic limit. Used directly by the ADuLT phenotype models (`adult` family), where `cip_x0` and `cip_k` (legacy parameter names — see Flagged ambiguities) parameterize the logistic CIF curve that maps liability rank to age-at-onset. CIF is also the canonical word in [fitACE_epimight](../fitACE/fitACE_epimight) — adopted from EPIMIGHT v2.0's survival/competing-risks framing.
_Avoid_: CIP, "cumulative incidence proportion", "cumulative incidence fraction" (deprecated — CIF is the cross-repo canonical); incidence rate (a per-time hazard, different concept); case rate.

**Cure fraction**:
In the mixture cure model (`cure_frailty`), the proportion of the population that will *never* experience the trait event regardless of follow-up. Equal to $1 - K$. Implemented by assigning non-susceptible individuals a sentinel onset of $10^6$ so they are right-censored under any realistic window.
_Avoid_: immune fraction, unsusceptible fraction, never-affected fraction.

### Censoring

**Censoring**:
Umbrella term for any mechanism that prevents a trait event from being observed — the individual was too young, too old, dead, or outside the study window. A censored individual has `affected = False` for a reason *other than* being genuinely below threshold. simACE does **not** store a per-individual censor-reason column; the reason is reconstructible from `onset`, the per-generation age window, and the (unstored) latent death age — agents that need the reason must infer it from the field values.
_Avoid_: missing (that's **ascertainment** — a different mechanism that removes the individual from the analysis dataset entirely, rather than failing to observe their event), excluded.

**Age-window censoring**:
The deterministic mechanism applied per generation via the observation window $[a_g^L, a_g^R]$. Covers both **left-censoring** (true onset would be before the window started) and **right-censoring** (true onset would be after the window ended). Generations with zero-width windows (e.g. $[80, 80]$) are fully censored and contribute pedigree structure but no observable events.
_Avoid_: window censoring (the "age-" qualifier matters), observational censoring.

**Death censoring** (competing-risk censoring):
The random mechanism where a per-individual mortality age, drawn from a Weibull, preempts trait onset. If $t_{\text{death}} < t_{\text{onset}}$, the observed time is set to $t_{\text{death}}$ and the individual is marked unaffected.
_Avoid_: mortality censoring (use "death"), competing event censoring.

**Left-censored** / **Right-censored**:
Adjective sub-types of age-window censoring. Left-censored = true onset was before the individual entered the observation window. Right-censored = true onset would be after the window ended (or after death).

### Ascertainment

**Ascertainment**:
The unified post-censor pipeline stage that determines which individuals are in the final analysis dataset. Replaces the older two-stage design (pedigree dropout before phenotype + subsampling after censor). The stage has two independent knobs — a random-removal **dropout rate** and a trait-weighted **case-ascertainment ratio** — plus a target sample size $N_{\text{sample}}$. Models how a real study's enrollment process distorts the recoverable population from the underlying truth.
_Avoid_: sampling stage (subsumed), sample selection, subsampling (the old word; see Flagged ambiguities), pedigree dropout (also subsumed).

**Dropout rate**:
The fraction of the population removed uniformly at random during ascertainment. Removal is independent of trait status, sex, generation, or pedigree position. Config key: `dropout_rate` (formerly `pedigree_dropout_rate`).
_Avoid_: attrition, pruning, missingness rate.

**Case-ascertainment ratio** ($\alpha$):
The weight applied to affected individuals during ascertainment selection, relative to unaffected. $\alpha = 1$ → uniform selection; $\alpha > 1$ → cases enriched; $\alpha = 0$ → only controls. Config key: `case_ascertainment_ratio`.
_Avoid_: ascertainment bias (that's the *result* of $\alpha \neq 1$, not the parameter itself), oversampling rate, enrichment factor.

**$N_{\text{sample}}$**:
Target size of the post-ascertainment analysis dataset. Config key: `N_sample`. The ascertainment step draws $N_{\text{sample}}$ individuals with weights determined by `dropout_rate` and `case_ascertainment_ratio`.

### Configuration

**Scenario**:
A named parameter set defined under `scenarios:` in a config YAML. Has its own seed, variance components, phenotype models, ascertainment settings, and other overrides on top of `defaults:`. Identified by its YAML key (e.g. `baseline10K`, `high_heritability`). Outputs live at `results/{folder}/{scenario}/`.
_Avoid_: experiment (killed entirely — do not use, even in prose), run, config (overloaded), parameter set, simulation (acceptable as a verb, not as a synonym for scenario).

**Folder**:
A grouping of related scenarios under one config YAML file. Folder name = YAML basename without extension (e.g. `config/heritability.yaml` → folder `heritability`). The top level of the `results/{folder}/{scenario}/rep{N}/` output tree. Defined by the `folder:` key in defaults; overridable per-scenario.
_Avoid_: group, suite, family, batch, scenario file (means the file, not this concept).

**Replicate**:
A single seeded run of a scenario, identified by `rep{N}` (e.g. `rep1`, `rep2`). Multiple replicates per scenario sample independent random draws; the seed for replicate $N$ is `seed + N`. Outputs at `results/{folder}/{scenario}/rep{N}/`. Replicates always exist (default 3, override per-scenario); a replicate-less scenario is not a valid configuration.
_Avoid_: trial, run, iteration, draw, sample (already taken — see **Ascertainment**).

**Config**:
The *merged runtime parameter set* for a specific scenario — i.e., what comes out of `simace.config.load_config` after defaults are overridden by scenario keys. Distinct from a *config YAML file*, which is the on-disk source. In prose "the config" usually means the merged dict; when the file is meant, say "the config YAML" or "the scenario file".
_Avoid_: settings, spec, parameter dict (acceptable internally but not the canonical name).

**Folder / scenario / replicate** are a strict 3-level hierarchy: every replicate belongs to exactly one scenario, every scenario to exactly one folder. There are no cross-folder scenarios and no scenarios without replicates.

**Mating model**:
The algorithm simACE uses to sample mating pairs and offspring counts each generation. One of `standard` (default) or `wright_fisher`. Config key: `pedigree.mating_model`. The choice is global to a scenario — there is no per-generation switching. Recorded faithfully in `params.yaml` so downstream stages (`validate`, `stats`, `plot`) can branch on it.
_Avoid_: mating algorithm, mating scheme, pedigree model.

**Wright-Fisher mating model** (`mating_model: wright_fisher`):
Sex-structured idealized Wright-Fisher: two sexes are retained; for each of the $N$ offspring next generation, one mother is drawn uniformly at random from the prior generation's females and one father uniformly from the males (both with replacement). Multinomial offspring counts per parent, no persistent mating-pair structure, no MZ twins. Households-per-mother (and therefore $C$) still apply because each offspring has a well-defined mother. Under random mating with 50/50 sex ratio, $N_e \to N$ by construction (Crow-Kimura). The standard-model knobs `mating_lambda`, `p_mztwin`, `assort1`, `assort2`, and `assort_matrix` are no-ops under WF; explicit non-no-op overrides of these in a WF scenario are rejected at config-load. See `docs/adr/0002-wright-fisher-mating-model.md`.
_Avoid_: hermaphroditic WF (this is *sex-structured* WF — a deliberate divergence from textbook hermaphroditic Wright-Fisher; see ADR 0002).

### Relatedness, kinship, and relationship types

**Relatedness** ($r$, coefficient of relationship):
The expected fraction of alleles two individuals share by descent — equivalently, $r_{ij} = 2\phi_{ij}$ for non-inbred pairs. The colloquial "siblings are 50% related" number. Used in prose and high-level discussion of how related two individuals are.
_Avoid_: relationship coefficient (use "coefficient of relationship"), genetic distance, sharing fraction.

**Kinship** ($\phi$, kinship coefficient):
The probability that an allele drawn at random from individual $i$ is IBD to an allele drawn at random from individual $j$. The specific per-pair coefficient stored and computed throughout simACE. For non-inbred pairs, $\phi = \tfrac{1}{2}r$, and follows the formula $\phi = n_{\text{ancestors}} \times (1/2)^{(\text{up} + \text{down} + 1)}$ from the `pedigree-graph` registry. With `estimate_inbreeding: true`, computed exactly via sparse propagation.
_Avoid_: coancestry (suppressed in canonical vocabulary — see Flagged ambiguities), relatedness (different concept — see above; relatedness = $2\phi$), kinship coefficient (acceptable, but "kinship" alone is preferred).

**Relationship type**:
The categorical label for a related pair — one of the 23 codes in the `REL_REGISTRY` from `pedigree-graph` (e.g., `MZ`, `FS`, `MHS`, `PHS`, `MO`, `GP`, `1C`, ...). Used in plotting, stats, validation, and analysis. The full table — codes, kinship values, and (up, down, n_ancestors) tuples — lives in `docs/concepts/simulation-design.md`; do not duplicate here.
_Avoid_: pair type (the legacy code name was `pair_type`; renamed to `relationship_type` — do not reintroduce), pair category, rel_type, kinship class.

**Relationship pair** (or **relative pair**):
An unordered $(i, j)$ tuple of two individuals together with a known **relationship type**. Distinct from **mating pair** (a male-female parental event) and from **MZ twin** (a specific relationship type).
_Avoid_: pair (alone — always qualify: relationship pair, mating pair).

**IBD** / **identity by descent**:
NOT a simACE concept. simACE does not simulate genotypes; kinship is the IBD-probability between individuals but IBD itself never manifests in code. If you see "IBD" in field-standard literature, mentally translate to "kinship" for simACE purposes.
_Avoid_: introducing IBD as a code identifier or stats output.

### Effective size

**Population size** ($N$):
The actual number of individuals per generation. Constant across generations by simACE design (fixed-$N$ is a hard invariant — see **Generation**). Config key: `N`.
_Avoid_: census size (quant-gen literature uses this to disambiguate from $N_e$; simACE does not — just say "population size" or "$N$").

**Effective size** ($N_e$, effective population size):
The size of an idealized Wright-Fisher population that would produce the same rate of drift / inbreeding / mean-kinship accumulation as the simulated pedigree. Distinct from the population size $N$. Canonical compound form: `effective_size` (matches code: `effective_size.yaml`, `validate_effective_size`, `simace.analysis.stats.effective_size`). "Effective population size" is acceptable in prose.
_Avoid_: effective number, Ne (acceptable as a math symbol but the word form is "effective size"), drift size.

**$N_e$ estimator**:
One of several methods simACE computes for inferring effective size from the pedigree. Each is identified by its code name (e.g., `ne_coancestry`, `ne_caballero_toro`). The estimator names are fixed identifiers tied to published methods — do not rename. The full set lives in `simace.analysis.stats.effective_size`; not enumerated here.
_Avoid_: Ne method (use "estimator"), drift estimator (subset only), inbreeding estimator (subset only).

### Pipeline stages

The pipeline runs the following stages in order. Stage names match the Snakemake rule files (`workflow/rules/simace/{stage}.smk`). Form is verb where natural, noun where natural — don't try to retroactively uniformize.

1. **Simulate** — generate the multi-generational pedigree with ACE variance components. Package: `simace/simulation/`.
2. **Phenotype** — apply the phenotype model per trait to produce binary affection + onset. Package: `simace/phenotype/` (noun form, **not** "phenotyping"). The only stage where noun-form is enforced because "phenotype" the noun is also a domain word.
3. **Censor** — apply age-window and death censoring to event times. Package: `simace/censoring/`.
4. **Ascertainment** — unified dropout + case-ascertainment + $N_{\text{sample}}$ selection (per ADR 0001). Noun form. Replaces the older `dropout` + `sample` two-stage split.
5. **Analyze** — combined production of ground-truth sanity checks and descriptive stats. Verb form. Folds the former validate and stats steps into one pipeline stage that emits a single curated per-replicate report plus a plot payload (ADR 0008).
6. **Plot** — render each scope's plot **atlas**. Package: `simace/plotting/`. The default, always-built rendering is a self-contained **HTML atlas** (`atlas.html`); a multi-page **PDF atlas** (`atlas.pdf`) is an on-demand export (ADR 0010).

_Avoid_: "phenotyping" (killed), "subsampling" / "dropout stage" (killed — see **Ascertainment**), "validation stage" / "stats stage" as separate pipeline stages (use **Analyze** for the combined stage; use **Per-replicate scientific report** / **Plot payload** when referring to artifacts), "statistics" (use "stats"), "simulation stage" (just say "the simulate stage").

**Per-replicate scientific report**:
The curated Analyze-stage report for one replicate. It summarizes quality checks, ground truth, observed post-ascertainment summaries, and estimator outputs, with every quantity labeled by the population scope it describes. Report scopes are **recorded pedigree** (full pre-ascertainment recorded pedigree), **phenotyped population** (full pre-ascertainment phenotyped/censored rows), **analysis sample** (final ascertained trait rows), and **analysis pedigree** (ancestor-closure pedigree supporting the analysis sample). It is not a plot cache and not a cross-replicate aggregate.
_Avoid_: phenotype statistics, stats dump, summary YAML, plot payload.

**Plot payload**:
A durable companion artifact for dense arrays needed only to render plots, such as age grids and full curve values. It is derived from the same replicate outputs as the scientific report but is not itself the scientific report.
_Avoid_: report, stats report, scientific summary.

**Plotting sample**:
A downsampled set of trait rows used only to draw dense scatter and histogram plots. It is distinct from the post-ascertainment analysis dataset and must not be used as an analysis sample.
_Avoid_: phenotype sample, stats sample, subsample.

**Atlas**:
The ordered set of figures, captions, and section breaks assembled for one **scope** into a single navigable document. An atlas is defined by its *manifest* — the figure order, caption text, and section dividers — and is independent of how it is rendered. Each atlas has two **renderings**: a self-contained **HTML atlas** (`atlas.html`) — the primary, default-built artifact, one portable file with figures embedded and equations as inline SVG — and a **PDF atlas** (`atlas.pdf`), an on-demand export (ADR 0010). Atlas scopes: the per-**scenario** atlas (phenotype figures plus a parameter overview and Table 1), the per-folder **validation** atlas (cross-scenario figures), and the fitACE-side **EPIMIGHT**, EPIMIGHT-bias, and onset-censoring atlases. The PA-FGRS atlas is a separate bespoke artifact, not part of this shared atlas family.
_Avoid_: "the PDF" / "the atlas PDF" as a synonym for the atlas (the PDF is now just one rendering — the HTML atlas is primary); "report" (an atlas is figures, not the **Per-replicate scientific report**).

### simACE, fitACE, and the ACE model

**simACE**:
This package — the simulation pipeline. Generates pedigrees, applies phenotype models, censors, ascertains, validates, computes descriptive stats, and plots. Does **not** fit variance-component models. Outputs are consumed downstream by fitACE.
_Avoid_: "the simulator", "the framework" (overloaded), "ACE" alone (ACE is the model, not this package).

**fitACE**:
The model-fitting sister repo — a **core + Snakemake orchestrator** (`fitace`) whose inferential methods (EPIMIGHT, PA-FGRS, sparseREML, iter_reml, Stan, PCGC, frailty) live in `fitACE_<x>` **method sisters** (see below). Consumes simACE outputs (`trait.parquet`, `pedigree.parquet`, `report.yaml`) to estimate variance components and recover ground-truth parameters. The boundary is **one-way**: simACE → fitACE, with no feedback loop into simACE.
_Avoid_: "the fitter", "the estimator suite" (subset), "the analysis package" (simACE also has `analysis/`).

**Public surface** (simACE → downstream):
The set of simACE modules that fitACE and its method sisters may import. Downstream imports only these *public* modules — never an underscore-private module such as `simace.core._numba_utils`. When a downstream package needs a private primitive, it is **promoted** into a public module rather than reached into (e.g. the bivariate-normal / tetrachoric numba kernels now exported from `simace.core.numerics`).
_Avoid_: treating "underscore = private" as advisory across the repo boundary; importing `simace.core._*` from fitACE.

**Method sister** (`fitACE_<x>` repo):
A repo holding exactly one fitting method's implementation, private helpers, Snakemake rule, and tests — `fitACE_epimight`, `fitACE_pcgc`, `fitACE_iter_reml`, `fitACE_tetraher`, `fitACE_pafgrs`, `fitACE_stan`, `fitACE_frailty`. Each depends on `fitace` + `simace`, never on another method sister. (See fitACE ADR 0001 for the core-vs-sister placement rule.)
_Avoid_: plugin, git submodule (they are gitignored sibling checkouts, not submodules), "fitACE module" (the methods are no longer inside `fitace/` core).

**Dormant** (method sister):
A method sister whose Snakemake rule file is **not** `include:`d in the core fitACE `Snakefile` — installed, importable, embedded for cross-repo search, and tested, but not wired into the pipeline DAG. *Active* = its rule file is included; activation is a one-line include. (`fitACE_stan` and `fitACE_frailty` are dormant; dormancy is about the DAG, not code coupling.)
_Avoid_: disabled, deprecated, inactive, retired (dormant code is live and tested — it is simply not orchestrated).

**ACE** (the model):
The variance-component decomposition $L = A + C + E$. The conceptual subject of both repos; **not** a package or a piece of software. When somebody says "the ACE model" they mean the math, not the code.
_Avoid_: ACE-model package, ACE framework, ACE-package.

**Ground truth**:
The known, configured-and-realized parameter values used to generate a scenario — the values that fitACE estimates will be benchmarked against. Includes the $A$, $C$, $E$ component values per individual, the cross-trait correlations, the per-individual trait status, the per-pair relationship type, and the configured prevalences. Distinct from any *estimate* fitACE produces.
_Avoid_: true values (acceptable in prose but less specific), simulated values, target values (target prevalence is a config *input* — it's only ground truth once realized).

### Versioning

**Lockstep family**:
The set of repos released under one shared version, tagged together each release — simACE, fitACE core, the seven method sisters, and the `ace_iter_reml` binary. External dependencies (`pedigree-graph`, `pedsum`, `tetraher_simace`) are **not** members and keep independent versions.
_Avoid_: "fitACE family" (excludes simACE by name), monorepo (separate repos, separate origins), submodule set.

**Family version**:
The single CalVer (`vYYYY.MM[.patch]`) every Lockstep family repo carries. Identical across repos at a tagged release; between releases each repo's dev build diverges only by its setuptools-scm commit-distance suffix.
_Avoid_: "the simACE version" / "the fitACE version" (under lockstep there is no per-repo version), build number.

**Family floor**:
The single minimum compatible Family version — the source of truth in `fitace._deps` that fitACE core and the method sisters pin (`>=`) and runtime-guard. simACE is upstream of the floor and does not import it.
_Avoid_: separate simACE / fitACE floors (collapsed into one under lockstep), version pin, minimum requirement.

- Each **Lockstep family** repo carries the same **Family version** at a tagged release.
- A **Method sister** is a **Lockstep family** member; an external dependency is not.
- fitACE core and the **Method sisters** pin and guard the single **Family floor**; simACE does not.

### Descriptive vs inferential analysis

simACE-side `simace.analysis.stats` produces **observed summaries** and simple **estimators**. Observed summaries are quantities directly measured from a scoped output population: prevalence, person-years, relationship counts, liability/affected/tetrachoric correlations, and mate correlations. Estimators are values derived from observed summaries to estimate a target quantity, such as naive observed-scale heritability from affected-status relationship correlations.

fitACE-side methods are **inferential**: they fit a full variance-component model (often with explicit pedigree structure, censoring, and ascertainment correction) and estimate $\sigma^2_A$, $\sigma^2_C$, $\sigma^2_E$, $r_A$, $r_C$, and related parameters with uncertainty.

When in doubt: if it directly describes a scoped simACE output population, it is an observed summary. If it transforms observed summaries into a target quantity such as $h^2$, it is an estimator. If it comes from fitting a variance-component model with uncertainty, it is inferential fitACE output.

### Plot captions & annotations

Conventions for the text rendered in the plot atlas — both the `PlotEntry` `title` / `body` captions in `simace/plotting/atlas_manifest.py` and the inline matplotlib annotations (panel titles, axis labels, the plotting-sample note) in the `plot_*.py` modules. The goal: a reader of any atlas page sees the same canonical terms this glossary defines. Python identifiers, stats-output keys, and docstrings are **exempt** — they may retain legacy words (e.g. `dz_sibling_A1_corr`, the `left_truncated` stats key, `subsample_note`, `mean_self_coancestry`).

- **Population scope.** Name a population using the canonical report scopes — **recorded pedigree**, **phenotyped population**, **analysis sample**, **analysis pedigree** (see **Per-replicate scientific report**) — never ad-hoc phrases like "full (non-subsampled) data". Quantities measured before the ascertainment stage are scoped to the **phenotyped population**; add "(pre-ascertainment)" where the contrast matters.
- **Plotting sample.** The downsample note on dense scatter/histogram plots says "plotting sample" (see **Plotting sample**), never "subsampled".
- **Observed prevalence.** A post-censoring affected fraction shown on a plot is **observed prevalence** (see the entry); reserve bare "prevalence" / $K$ for the configured target.
- **Trait, not disease.** The event of interest in a time-to-event / competing-risks caption is the **trait event** (as already used in the **Onset** entry); write "AJ trait CIF" vs "AJ death CIF", "terminal AJ trait F(∞)". Do not call it "disease" — simACE traits are abstract (see **Trait**, _Avoid_: disease).
- **Blessed caption compounds.** Two survival-analysis terms are permitted in caption / annotation prose as a narrow carve-out from the bare-word _Avoid_ rules (which target running prose): **"true cases"** for ground-truth affecteds at the **censoring** stage (pre-ascertainment, where the study-role meaning of "case" is not in play), and bare **"event time"** for the raw pre-censoring simulated time (distinct from the clipped `onset` column). Both stay confined to censoring-stage captions.
- **Retired / corrected labels.** "survival model" is no longer used as a plot qualifier — it is inaccurate for the threshold `adult` / `ltm` family, and the model-aware section break already names the configured family. In caption prose, age-window censoring before the window opens is **left-censored**, never "left-truncated" (which survives only as a stats key); "delayed entry" remains correct in the Aalen-Johansen captions, where it names how per-generation observation windows are honoured. "coancestry" stays out of caption prose (use **kinship** / **mean kinship**); only the fixed estimator identifiers `Ne_coancestry` / `mean_self_coancestry` / `Ne_caballero_toro` keep the word.

## Flagged ambiguities

- **"fitACE family"** reads as fitACE-only, but the **Lockstep family** (the versioned set) also contains simACE and the `ace_iter_reml` binary. Use **Lockstep family** when you mean the repos that share a **Family version**; reserve "fitACE core + method sisters" for the fit-only subset.

- **"liability"** is intentionally polysemous: it can refer to the raw $L = A + C + E$ or to its standardized form $\tilde L$. Both readings are legitimate; the right one is inferable from context (raw inside `pedigree.parquet` columns; standardized inside phenotype-model code that consumes it). **Do not** "fix" this by renaming — the dual usage is load-bearing.

- **"phenotype"** is never used bare for the *observable outcome* — that's called a **trait** (per-individual instance). "Phenotype" appears only in qualified form: **phenotype model** (the family) or **phenotype stage** (the pipeline step). The canonical post-ascertainment output file is `trait.parquet` (renamed from the legacy `phenotype.parquet`). `simple_ltm` is a phenotype **model** (liability threshold + fixed/normal onset), not a separate output; the former parallel `trait.simple_ltm.parquet` was retired (ADR 0011 amendment). The descriptive binary stats that consumed it are now fitACE's **observed-binary** outputs, computed from `trait.parquet` for every scenario.

- **Noun, not gerund** for the phenotype stage: the canonical package is `simace/phenotype/` (noun). The legacy `simace/phenotyping/` gerund form was renamed in lockstep. Do not reintroduce "phenotyping" identifiers.

- **"trait"** carries both an *axis* sense ("the two traits") and a *per-individual instance* sense ("individual $i$'s trait 1"). Unlike liability, both senses always co-occur in the same code/text and context makes the right reading unambiguous — no special care needed.

- **"pedigree dropout" and "subsampling"** are *deprecated names* — both have been folded into the unified **ascertainment** stage (see ADR 0001). The config key `pedigree_dropout_rate` is renamed to `dropout_rate` and the YAML block `sampling:` is renamed to `ascertainment:`. The legacy `dropout.smk` / `sample.smk` rules and `simace/sampling/` package have been removed. Do not reintroduce these names.

- **"coancestry"** is suppressed in canonical simACE vocabulary — use **kinship** for per-pair coefficients and **mean kinship** for the generation-aggregate form. The legacy estimator name `ne_coancestry` (and column names like `mean_self_coancestry`) survive in code because they match published-literature names for the coancestry-rate $N_e$ estimator; treat these as fixed external identifiers, not as a glossary term to extend.

- **"pair type"** is the deprecated code name for **relationship type**. The `pair_type` identifiers in `simace.plotting`, `simace.analysis.stats`, and `simace.core.relationships.PAIR_TYPES` were renamed to `relationship_type` / `RELATIONSHIP_TYPES` across both simACE and fitACE. Do not reintroduce them.

- **`cip_x0` / `cip_k`** are *legacy parameter names* in `simace.phenotype.models.adult` — the *concept* renamed CIP → CIF (see **Cumulative incidence**), but the parameter identifiers retain the `cip_` prefix to avoid a config-migration sweep across scenario YAMLs. Don't rename these to `cif_x0`/`cif_k` in code; do say "CIF" in prose, comments, and plot labels.

## Example dialogue

> **Agent:** "Two individuals share a `household_id` — are they full siblings?"
> **Maintainer:** "Not necessarily. A **household** is grouped by *mother*, not by *mating pair*. So maternal half-siblings (same mother, different father) share a household and a $C$ draw, but they're not full sibs. To check full vs. half, compare both `mother` and `father`."

> **Agent:** "Should I subtract midparent from each offspring's `liability1` to get the genetic deviation?"
> **Maintainer:** "Careful — `liability1` is the raw $L = A + C + E$, not the additive component. You want `A1` if you're after the genetic deviation. And remember that 'liability' can mean either the raw value or the standardized form $\tilde L$ depending on context — phenotype models consume $\tilde L$, but the parquet stores raw $L$."

> **Agent:** "I want to count cases per generation."
> **Maintainer:** "Be precise about what you mean. If you want the *ground-truth affected* — every individual the phenotype model + censoring marked as positive — sum `affected1`. If you want **cases** in the study-sample sense, that's only meaningful after the **ascertainment** stage with case-ascertainment weighting; before that step, individuals are *affected* or *unaffected*, not *cases* or *controls*."
