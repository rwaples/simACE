"""ACE pedigree simulation - Snakemake wrapper with CLI fallback."""
from sim_ace import setup_logging
from sim_ace.simulate import run_simulation, cli as _cli

import yaml


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    params = snakemake.params
    output_pedigree = snakemake.output.pedigree
    output_params = snakemake.output.params

    pedigree = run_simulation(
        seed=params.seed,
        N=params.N,
        G_ped=params.G_ped,
        fam_size=params.fam_size,
        p_mztwin=params.p_mztwin,
        p_nonsocial_father=params.p_nonsocial_father,
        A1=params.A1,
        C1=params.C1,
        A2=params.A2,
        C2=params.C2,
        rA=params.rA,
        rC=params.rC,
        G_sim=params.G_sim,
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
        "fam_size": params.fam_size,
        "p_mztwin": params.p_mztwin,
        "p_nonsocial_father": params.p_nonsocial_father,
    }
    with open(output_params, "w", encoding="utf-8") as f:
        yaml.dump(params_dict, f, default_flow_style=False)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
