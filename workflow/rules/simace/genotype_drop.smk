# Per-chromosome pedigree drop + ancestry graft.
#
# For each (folder, scenario, rep, chrom) the rule:
#   1. Loads the simACE pedigree (from `simulate_pedigree_liability`).
#   2. Builds an msprime fixed-pedigree.
#   3. Drops it under msprime with the SimHumanity HapMapII_GRCh38
#      per-chromosome recombination rate map.
#   4. Grafts the canonicalized p2 ancestry (from `tskit_preprocess`) onto
#      the founder lineages.
#
# Prerequisites:
#   - Run `tskit_preprocess` first to produce
#     <preprocessed>/canonicalized/chromosome_{n}.trees.
#   - `git -C external/SimHumanity checkout main` to populate the SimHumanity
#     working tree (the rate map files live under
#     external/SimHumanity/stdpopsim extraction/extracted/).
#   - Pipeline must be invoked with `--use-conda`.
#
# Targets:
#   results/{folder}/{scenario}/rep{rep}/genotypes_chrom_{n}.trees
#   results/{folder}/{scenario}/rep{rep}/.simulate_genotypes.done   (alias)


_GD_PREPROCESSED = config["tskit_preprocess"]["output_dir"]
_GD_SIMHUMANITY = "external/SimHumanity"


rule build_pedigree_tables:
    """Build the msprime fixed-pedigree TableCollection once per rep.

    Saves ~80s/rep over the previous per-chrom rebuild on baseline100K.
    Output is a tskit .trees file with placeholder sequence_length=1; each
    chrom-drop rule mutates sequence_length to its per-chrom value.
    """
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.full.parquet",
    output:
        tables="results/{folder}/{scenario}/rep{rep}/pedigree_tables.trees",
    log:
        "logs/{folder}/{scenario}/rep{rep}/build_pedigree_tables.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/build_pedigree_tables.tsv"
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=15,
    params:
        G_ped=lambda w: get_param(config, w.scenario, "G_ped"),
        G_pheno=lambda w: get_param(config, w.scenario, "G_pheno"),
    script:
        "../../scripts/simace/tskit/build_pedigree_tables.py"


rule simulate_genotypes_chrom:
    input:
        pedigree_tables="results/{folder}/{scenario}/rep{rep}/pedigree_tables.trees",
        trees=f"{_GD_PREPROCESSED}/canonicalized/chromosome_{{n}}.trees",
        recomb=f"{_GD_SIMHUMANITY}/stdpopsim extraction/extracted/chr{{n}}_recombination.txt",
    output:
        trees="results/{folder}/{scenario}/rep{rep}/genotypes_chrom_{n}.trees",
    log:
        "logs/{folder}/{scenario}/rep{rep}/simulate_genotypes_chrom_{n}.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/simulate_genotypes_chrom_{n}.tsv"
    wildcard_constraints:
        n=r"\d+",
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=8000,
        runtime=60,
    params:
        G_ped=lambda w: get_param(config, w.scenario, "G_ped"),
        seed=lambda w: get_param(config, w.scenario, "seed")
        + (int(w.rep) - 1)
        + 100 * int(w.n),
        simhumanity_dir=_GD_SIMHUMANITY,
    script:
        "../../scripts/simace/tskit/genotype_drop_chrom.py"


rule simulate_genotypes:
    """Alias: drop+graft all 22 autosomes for one (folder, scenario, rep)."""
    input:
        lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/genotypes_chrom_{n}.trees",
            folder=w.folder,
            scenario=w.scenario,
            rep=w.rep,
            n=range(1, 23),
        ),
    output:
        touch("results/{folder}/{scenario}/rep{rep}/.simulate_genotypes.done"),
