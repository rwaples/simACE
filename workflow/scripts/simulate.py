"""ACE pedigree simulation - Snakemake wrapper with CLI fallback."""

import yaml

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.simulate import cli as _cli
from sim_ace.simulate import run_simulation


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    params = snakemake.params
    output_pedigree = snakemake.output.pedigree
    output_params = snakemake.output.params

    pedigree = run_simulation(
        seed=params.seed,
        N=params.N,
        G_ped=params.G_ped,
        mating_lambda=params.mating_lambda,
        p_mztwin=params.p_mztwin,
        A1=params.A1,
        C1=params.C1,
        A2=params.A2,
        C2=params.C2,
        rA=params.rA,
        rC=params.rC,
        G_sim=params.G_sim,
        assort1=params.assort1,
        assort2=params.assort2,
        assort_matrix=params.assort_matrix,
    )

    pedigree.to_parquet(output_pedigree, index=False)

    params_dict = {
        "seed": params.seed,
        "rep": params.rep,
        "A1": params.A1,
        "C1": params.C1,
        "E1": 1.0 - params.A1 - params.C1,
        "A2": params.A2,
        "C2": params.C2,
        "E2": 1.0 - params.A2 - params.C2,
        "rA": params.rA,
        "rC": params.rC,
        "N": params.N,
        "G_ped": params.G_ped,
        "G_sim": params.G_sim,
        "mating_lambda": params.mating_lambda,
        "p_mztwin": params.p_mztwin,
        "assort1": params.assort1,
        "assort2": params.assort2,
    }
    if params.assort_matrix is not None:
        params_dict["assort_matrix"] = params.assort_matrix
    with open(output_params, "w", encoding="utf-8") as f:
        yaml.dump(params_dict, f, default_flow_style=False)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
