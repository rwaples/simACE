# tstrait quantitative phenotyping on per-chrom drop+graft tree sequences.
#
# NOTE: requires `--use-conda` (uses workflow/envs/tskit.yaml).
#
# Pipeline (per (folder, scenario, rep)):
#
#     22 canonicalized chrom_{n}.trees ──► tstrait_site_catalog_chrom (×22)
#                                                  │
#                                                  ▼
#                                  tstrait_site_catalog_concat
#                                                  │
#                                                  ▼
#               (one of, depending on share_architecture flag)
#               ┌────────────────────────────┬────────────────────────────┐
#         per-rep: tstrait_assign_effects_per_rep    shared: tstrait_assign_effects_shared
#               └──────────────┬──────────────┴──────────────┬─────────────┘
#                              │                              │
#                              ▼                              ▼
#                  per-rep grafted .trees + chosen causal_effects.parquet
#                                              │
#                                              ▼
#                                  tstrait_gv_chrom (×22)
#                                              │
#                                              ▼
#                                  tstrait_phenotype  (echoes assign-time
#                                              │     params from causal_effects_meta.json)
#                                              ▼
#                                  tstrait_phenotype_all  (alias)
#
# `share_architecture` mechanic
#   Two assign-effects rules with disjoint output paths are defined
#   unconditionally. The gv_chrom and phenotype rules' `effects` /
#   `effects_meta` inputs are dispatched by input functions so only the
#   matching rule is reachable. Caveat: flipping the flag mid-run leaves
#   stale files at the other path — clean them up manually if needed.
#
# Targets
#   results/{folder}/{scenario}/rep{rep}/tstrait_phenotype.parquet
#   results/{folder}/{scenario}/rep{rep}/.tstrait_phenotype.done   (alias)
#
# Seed namespaces (no overlap with genotype_drop's seed + (rep-1) + 100*n)
#   tstrait_assign_effects_per_rep:  scenario_seed + (rep-1) + _SEED_OFFSET_ASSIGN_EFFECTS
#   tstrait_assign_effects_shared:   scenario_seed +             _SEED_OFFSET_ASSIGN_EFFECTS
#   tstrait_phenotype (sim_env):     scenario_seed + (rep-1) + _SEED_OFFSET_PHENOTYPE


_TSTRAIT_PREPROCESSED = config["tskit_preprocess"]["output_dir"]
_TSTRAIT_CANON = f"{_TSTRAIT_PREPROCESSED}/canonicalized"
_TSTRAIT_CATALOG_DIR = f"{_TSTRAIT_PREPROCESSED}/site_catalog"
_TSTRAIT_CATALOG = f"{_TSTRAIT_PREPROCESSED}/site_catalog.parquet"
_TSTRAIT_CATALOG_SUMMARY = f"{_TSTRAIT_PREPROCESSED}/site_catalog_summary.json"
_TSTRAIT_CHROMS = list(range(1, 23))

# Seed-offset namespaces, kept disjoint from genotype_drop's `+ 100*n` shift
# so a rep / chrom / sub-stage collision is impossible up to thousands of reps.
_SEED_OFFSET_ASSIGN_EFFECTS = 10_000
_SEED_OFFSET_PHENOTYPE = 20_000


def _tstrait_effects_path(wildcards):
    """Return scenario-shared or per-rep causal_effects.parquet path."""
    if get_param(config, wildcards.scenario, "tstrait_share_architecture"):
        return f"results/{wildcards.folder}/{wildcards.scenario}/causal_effects.parquet"
    return f"results/{wildcards.folder}/{wildcards.scenario}/rep{wildcards.rep}/causal_effects.parquet"


def _tstrait_effects_meta_path(wildcards):
    """Return scenario-shared or per-rep causal_effects_meta.json path."""
    if get_param(config, wildcards.scenario, "tstrait_share_architecture"):
        return (
            f"results/{wildcards.folder}/{wildcards.scenario}/causal_effects_meta.json"
        )
    return f"results/{wildcards.folder}/{wildcards.scenario}/rep{wildcards.rep}/causal_effects_meta.json"


def _drop_source(w):
    """Resolve which scenario provides pedigree+drop outputs.

    Returns the scenario named in `drop_from` if set, else the current
    scenario. Variants that share genotypes with a base scenario set
    `drop_from: <base>` so this rule's input function points the gv_chrom and
    augment rules at the base scenario's `pedigree.full.parquet` and
    `genotypes_chrom_*.trees`.
    """
    src = get_param(config, w.scenario, "drop_from")
    return src if src else w.scenario


def _drop_trees_path(w):
    return (
        f"results/{w.folder}/{_drop_source(w)}/rep{w.rep}/genotypes_chrom_{w.n}.trees"
    )


def _drop_pedigree_path(w):
    return f"results/{w.folder}/{_drop_source(w)}/rep{w.rep}/pedigree.full.parquet"


def _trait1_h2(w):
    """h2 = A1 / (A1 + C1 + E1) for trait 1 — derived from the standard simACE
    A/C/E components so the tstrait branch is on the same scale and there is
    no separate h2 knob to keep in sync."""
    a = float(get_param(config, w.scenario, "A1"))
    c = float(get_param(config, w.scenario, "C1"))
    e = float(get_param(config, w.scenario, "E1"))
    total = a + c + e
    if total <= 0:
        raise ValueError(f"scenario '{w.scenario}': A1+C1+E1 must be > 0; got {total}")
    return a / total


def _tstrait_assign_common_params():
    """Shared params for both assign_effects rules — only the seed differs."""
    return dict(
        num_causal=lambda w: get_param(config, w.scenario, "tstrait_num_causal"),
        frac_causal=lambda w: get_param(config, w.scenario, "tstrait_frac_causal"),
        maf_threshold=lambda w: get_param(config, w.scenario, "tstrait_maf_threshold"),
        alpha=lambda w: get_param(config, w.scenario, "tstrait_alpha"),
        effect_mean=lambda w: get_param(config, w.scenario, "tstrait_effect_mean"),
        effect_var=lambda w: get_param(config, w.scenario, "tstrait_effect_var"),
        trait_id=lambda w: get_param(config, w.scenario, "tstrait_trait_id"),
    )


rule tstrait_site_catalog_chrom:
    input:
        trees=f"{_TSTRAIT_CANON}/chromosome_{{n}}.trees",
    output:
        catalog=f"{_TSTRAIT_CATALOG_DIR}/chrom_{{n}}.parquet",
    log:
        "logs/tstrait/site_catalog_chrom_{n}.log",
    benchmark:
        "benchmarks/tstrait/site_catalog_chrom_{n}.tsv"
    wildcard_constraints:
        n=r"\d+",
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=8000,
        runtime=15,
    script:
        "../../scripts/simace/tskit/tstrait_site_catalog_chrom.py"


rule tstrait_site_catalog_concat:
    input:
        catalogs=expand(
            f"{_TSTRAIT_CATALOG_DIR}/chrom_{{n}}.parquet", n=_TSTRAIT_CHROMS
        ),
    output:
        catalog=_TSTRAIT_CATALOG,
        summary=_TSTRAIT_CATALOG_SUMMARY,
    log:
        "logs/tstrait/site_catalog_concat.log",
    benchmark:
        "benchmarks/tstrait/site_catalog_concat.tsv"
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    script:
        "../../scripts/simace/tskit/tstrait_site_catalog_concat.py"


rule tstrait_assign_effects_per_rep:
    input:
        catalog=_TSTRAIT_CATALOG,
    output:
        effects="results/{folder}/{scenario}/rep{rep}/causal_effects.parquet",
        meta="results/{folder}/{scenario}/rep{rep}/causal_effects_meta.json",
    log:
        "logs/{folder}/{scenario}/rep{rep}/tstrait_assign_effects.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/tstrait_assign_effects.tsv"
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=15,
    params:
        **_tstrait_assign_common_params(),
        seed=lambda w: get_param(config, w.scenario, "seed")
        + (int(w.rep) - 1)
        + _SEED_OFFSET_ASSIGN_EFFECTS,
    script:
        "../../scripts/simace/tskit/tstrait_assign_effects.py"


rule tstrait_assign_effects_shared:
    input:
        catalog=_TSTRAIT_CATALOG,
    output:
        effects="results/{folder}/{scenario}/causal_effects.parquet",
        meta="results/{folder}/{scenario}/causal_effects_meta.json",
    log:
        "logs/{folder}/{scenario}/tstrait_assign_effects_shared.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/tstrait_assign_effects_shared.tsv"
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=15,
    params:
        **_tstrait_assign_common_params(),
        seed=lambda w: get_param(config, w.scenario, "seed")
        + _SEED_OFFSET_ASSIGN_EFFECTS,
    script:
        "../../scripts/simace/tskit/tstrait_assign_effects.py"


rule tstrait_gv_chrom:
    input:
        trees=_drop_trees_path,
        effects=_tstrait_effects_path,
    output:
        gv="results/{folder}/{scenario}/rep{rep}/gv_chrom_{n}.parquet",
        meta="results/{folder}/{scenario}/rep{rep}/gv_chrom_{n}_meta.json",
    log:
        "logs/{folder}/{scenario}/rep{rep}/tstrait_gv_chrom_{n}.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/tstrait_gv_chrom_{n}.tsv"
    wildcard_constraints:
        n=r"\d+",
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=45,
    params:
        trait_id=lambda w: get_param(config, w.scenario, "tstrait_trait_id"),
    script:
        "../../scripts/simace/tskit/tstrait_gv_chrom.py"


rule tstrait_phenotype:
    input:
        gv=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/gv_chrom_{n}.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=w.rep,
            n=_TSTRAIT_CHROMS,
        ),
        meta=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/gv_chrom_{n}_meta.json",
            folder=w.folder,
            scenario=w.scenario,
            rep=w.rep,
            n=_TSTRAIT_CHROMS,
        ),
        effects=_tstrait_effects_path,
        effects_meta=_tstrait_effects_meta_path,
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/tstrait_phenotype.parquet",
        meta="results/{folder}/{scenario}/rep{rep}/tstrait_phenotype_meta.json",
    log:
        "logs/{folder}/{scenario}/rep{rep}/tstrait_phenotype.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/tstrait_phenotype.tsv"
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=15,
    params:
        h2=_trait1_h2,
        trait_id=lambda w: get_param(config, w.scenario, "tstrait_trait_id"),
        share_architecture=lambda w: get_param(
            config, w.scenario, "tstrait_share_architecture"
        ),
        seed=lambda w: get_param(config, w.scenario, "seed")
        + (int(w.rep) - 1)
        + _SEED_OFFSET_PHENOTYPE,
    script:
        "../../scripts/simace/tskit/tstrait_phenotype.py"


rule tstrait_augment_pedigree:
    """Overwrite pedigree A1 with rescaled tstrait GV; recompute liability1.

Produces a sibling pedigree.full.tstrait.parquet so downstream simACE
phenotype models (frailty, threshold, etc.) can run on the augmented
pedigree and use the standard A+C+E variance composition with a
realistic A column derived from real genotypes.
"""
    input:
        pedigree=_drop_pedigree_path,
        gv=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/gv_chrom_{n}.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=w.rep,
            n=_TSTRAIT_CHROMS,
        ),
    output:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.full.tstrait.parquet",
        meta="results/{folder}/{scenario}/rep{rep}/pedigree.full.tstrait_meta.json",
    log:
        "logs/{folder}/{scenario}/rep{rep}/tstrait_augment_pedigree.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/tstrait_augment_pedigree.tsv"
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=15,
    params:
        target_var_A=lambda w: get_param(config, w.scenario, "A1"),
    script:
        "../../scripts/simace/tskit/tstrait_augment_pedigree.py"


rule tstrait_phenotype_all:
    """Alias: drop+graft+phenotype the full genome for one (folder, scenario, rep)."""
    input:
        "results/{folder}/{scenario}/rep{rep}/tstrait_phenotype.parquet",
        "results/{folder}/{scenario}/rep{rep}/pedigree.full.tstrait.parquet",
    output:
        touch("results/{folder}/{scenario}/rep{rep}/.tstrait_phenotype.done"),
