# ---------------------------------------------------------------------------
# EPIMIGHT: heritability and genetic correlation from time-to-event data
# ---------------------------------------------------------------------------
#
# Three-step pipeline per replicate:
#   1. epimight_create_parquet  – phenotype.parquet → NDD/NDG parquets + truth JSON
#   2. epimight_guide_yob       – per relationship kind: CIF, h², GC analysis (R)
#   3. epimight_atlas           – assemble all kinds into a plot atlas PDF
#
# Output directory: results/{folder}/{scenario}/rep{rep}/epimight/
# ---------------------------------------------------------------------------

EPIMIGHT_KINDS = config["defaults"].get("epimight_kinds", ["PO", "FS", "HS"])


rule epimight_create_parquet:
    """Convert ACE phenotype to EPIMIGHT time-to-event format."""
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
    output:
        ndd="results/{folder}/{scenario}/rep{rep}/epimight/NDD.parquet",
        ndg="results/{folder}/{scenario}/rep{rep}/epimight/NDG.parquet",
        truth="results/{folder}/{scenario}/rep{rep}/epimight/true_parameters.json",
    log:
        "logs/{folder}/{scenario}/rep{rep}/epimight_create_parquet.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/epimight_create_parquet.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime = lambda w: _scale_runtime(config, w.scenario, "G_pheno")
    threads: 1
    script:
        "../scripts/epimight_create_parquet.py"


rule epimight_guide_yob:
    """Run EPIMIGHT CIF, heritability, and genetic correlation for one relationship kind."""
    input:
        ndd="results/{folder}/{scenario}/rep{rep}/epimight/NDD.parquet",
        ndg="results/{folder}/{scenario}/rep{rep}/epimight/NDG.parquet",
    output:
        h2_d1="results/{folder}/{scenario}/rep{rep}/epimight/tsv/h2_d1_{kind}.tsv",
        h2_d2="results/{folder}/{scenario}/rep{rep}/epimight/tsv/h2_d2_{kind}.tsv",
        gc="results/{folder}/{scenario}/rep{rep}/epimight/tsv/gc_full_{kind}.tsv",
        report="results/{folder}/{scenario}/rep{rep}/epimight/results_{kind}.md",
    log:
        "logs/{folder}/{scenario}/rep{rep}/epimight_guide_yob_{kind}.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/epimight_guide_yob_{kind}.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime = lambda w: _scale_runtime(config, w.scenario, "G_pheno")
    threads: 1
    shell:
        "conda run -n epimight "
        "Rscript epimight/guide-yob.R "
        "results/{wildcards.folder}/{wildcards.scenario}/rep{wildcards.rep}/epimight "
        "{wildcards.kind} "
        ">{log} 2>&1"


rule epimight_atlas:
    """Generate EPIMIGHT plot atlas comparing all relationship kinds."""
    input:
        tsv=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/epimight/tsv/h2_d1_{kind}.tsv",
            folder=w.folder, scenario=w.scenario, rep=w.rep,
            kind=get_param(config, w.scenario, "epimight_kinds"),
        ),
        truth="results/{folder}/{scenario}/rep{rep}/epimight/true_parameters.json",
    output:
        atlas="results/{folder}/{scenario}/rep{rep}/epimight/plots/atlas.pdf",
    log:
        "logs/{folder}/{scenario}/rep{rep}/epimight_atlas.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/epimight_atlas.tsv"
    resources:
        mem_mb  = 4000,
        runtime = 10
    threads: 1
    script:
        "../scripts/epimight_atlas.py"


rule epimight_folder:
    """Run EPIMIGHT analysis for all scenarios in a folder."""
    input:
        lambda w: [f"results/{w.folder}/{s}/rep{r}/epimight/plots/atlas.pdf"
                   for s in get_scenarios_for_folder(config, w.folder)
                   for r in range(1, get_param(config, s, "replicates") + 1)]
    output:
        touch("results/{folder}/epimight.done")


rule epimight_all:
    """Run EPIMIGHT analysis for all scenarios and replicates."""
    input:
        [f"results/{get_folder(config, s)}/{s}/rep{r}/epimight/plots/atlas.pdf"
         for s in config["scenarios"]
         for r in range(1, get_param(config, s, "replicates") + 1)]
