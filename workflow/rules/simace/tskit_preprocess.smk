# Per-chromosome tskit canonicalize + multi-chrom streaming concat.
#
# NOTE: requires `--use-conda`. The rules use the dedicated tskit env in
# workflow/envs/tskit.yaml (tskit isn't in the simACE env). To avoid building
# a ~5 GB env per worktree, pass `--conda-prefix /data/.snakemake_conda`.
#
# Targets
#   tskit_preprocess          one-shot alias: 22x canonicalize + 1 concat
#   tskit_preprocess_verify   separate target: golden + sidecar fingerprint check
#
# Autosomes only (chr1..22). Sex chromosomes are an explicit non-goal — the
# canonicalize script raises if a non-autosome filename is fed in.


_TP = config["tskit_preprocess"]
_SRC = _TP["source_dir"]
_OUT = _TP["output_dir"]
_POP = _TP.get("pop")
_CHROMS = _TP.get("chroms", list(range(1, 23)))
_CANON = f"{_OUT}/canonicalized"


rule tskit_preprocess_canonicalize_chrom:
    input:
        src=f"{_SRC}/chromosome_{{n}}.trees",
    output:
        canon=f"{_CANON}/chromosome_{{n}}.trees",
        stats=f"{_CANON}/chromosome_{{n}}.stats.json",
    log:
        "logs/tskit_preprocess/canonicalize_chrom_{n}.log",
    benchmark:
        "benchmarks/tskit_preprocess/canonicalize_chrom_{n}.tsv"
    wildcard_constraints:
        n=r"\d+",
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=12000,
        runtime=30,
    params:
        pop_name=_POP,
    script:
        "../../scripts/simace/tskit/canonicalize_chrom.py"


rule tskit_preprocess_concat:
    input:
        canon=expand(f"{_CANON}/chromosome_{{n}}.trees", n=_CHROMS),
        stats=expand(f"{_CANON}/chromosome_{{n}}.stats.json", n=_CHROMS),
    output:
        trees=f"{_OUT}/all_chroms.trees",
        summary=f"{_OUT}/preprocess_summary.json",
        fingerprint=f"{_OUT}/all_chroms.trees.fingerprint",
    log:
        "logs/tskit_preprocess/concat.log",
    benchmark:
        "benchmarks/tskit_preprocess/concat.tsv"
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=8000,
        runtime=60,
    params:
        pop_name=_POP,
        source_dir=_SRC,
        chroms=_CHROMS,
    script:
        "../../scripts/simace/tskit/concat_chroms.py"


rule tskit_preprocess_verify:
    input:
        trees=f"{_OUT}/all_chroms.trees",
        fingerprint=f"{_OUT}/all_chroms.trees.fingerprint",
        summary=f"{_OUT}/preprocess_summary.json",
    output:
        touch(f"{_OUT}/.verify.done"),
    log:
        "logs/tskit_preprocess/verify.log",
    conda:
        "../../envs/tskit.yaml"
    threads: 1
    resources:
        mem_mb=8000,
        runtime=15,
    params:
        # Passed via params (not input) so first-run bootstrap with
        # SIMACE_TSKIT_WRITE_GOLDEN=1 works before the file exists.
        golden="tests/data/tskit_preprocess_fingerprint.json",
    script:
        "../../scripts/simace/tskit/verify.py"


rule tskit_preprocess:
    input:
        f"{_OUT}/all_chroms.trees",
        f"{_OUT}/preprocess_summary.json",
        f"{_OUT}/all_chroms.trees.fingerprint",
