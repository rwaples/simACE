"""
Prepare Survival Kit input files for Weibull frailty model estimation.

Reads pedigree.parquet and phenotype.weibull.parquet, writes 5 files per trait:
  - data.dat        : processed data file (space-delimited)
  - codelist.txt     : categorical variable metadata
  - varlist.txt      : variable positions/types
  - pedigree.ped     : pedigree (id, sex, sire, dam)
  - weibull.txt      : model specification

File formats match Survival Kit prepare.exe output exactly:
  - Variable names truncated to 8 characters
  - Codelist uses Fortran format (1X, A8, I5, I2, I10, I10, I10, I8, I8)
  - All class variable codes are 1-based
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_data(df: pd.DataFrame, trait: str, path: str) -> None:
    """Write data.dat for a given trait.

    Columns: time code id sex generation animal
    All IDs are 1-based (simulation IDs + 1).
    """
    t_col = f"t_observed{trait}"
    affected_col = f"affected{trait}"

    out = pd.DataFrame()
    out["time"] = df[t_col]
    out["code"] = df[affected_col].astype(int)
    out["id"] = df["id"] + 1  # 0-based -> 1-based
    out["sex"] = df["sex"].map({0: 1, 1: 2})  # 0=F->1, 1=M->2
    out["generation"] = df["generation"] + 1  # 0-based -> 1-based
    out["animal"] = df["id"] + 1  # 1-based, same as id
    out["share_hh"] = df["household_id"] + 1  # 0-based -> 1-based

    out.to_csv(path, sep=" ", index=False, header=False)


def _codelist_line(name: str, num_levels: int, original_code: int, freq: int, new_code: int) -> str:
    """Format one codelist line matching Fortran (1X, A8, I5, I2, I10, I10, I10, I8, I8)."""
    return f" {name:<8s}{num_levels:5d}{1:2d}{original_code:10d}{0:10d}{0:10d}{freq:8d}{new_code:8d}"


def write_codelist(df: pd.DataFrame, G_ped: int, path: str) -> None:
    """Write codelist.txt with level metadata for each class variable."""
    sex_recoded = df["sex"].map({0: 1, 1: 2})
    gen_recoded = df["generation"] + 1

    lines = []

    # sex: 2 levels
    for code in [1, 2]:
        freq = int((sex_recoded == code).sum())
        lines.append(_codelist_line("sex", 2, code, freq, code))

    # generati (truncated to 8 chars): G_ped levels
    for code in range(1, G_ped + 1):
        freq = int((gen_recoded == code).sum())
        lines.append(_codelist_line("generati", G_ped, code, freq, code))

    # animal: N_total levels (1-based IDs)
    animal_ids = sorted(df["id"].unique())
    n_animals = len(animal_ids)
    for new_code, aid in enumerate(animal_ids, start=1):
        lines.append(_codelist_line("animal", n_animals, new_code, 1, new_code))

    # share_hh (truncated to 8 chars): N_households levels
    household_ids = sorted(df["household_id"].unique())
    n_households = len(household_ids)
    for new_code, hid in enumerate(household_ids, start=1):
        lines.append(_codelist_line("share_hh", n_households, new_code, 1, new_code))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_varlist(G_ped: int, N_total: int, N_households: int, path: str) -> None:
    """Write varlist.txt describing column positions and variable types.

    Variable names truncated to 8 characters to match Survival Kit.
    """
    lines = [
        "TIME    1;",
        "CODE    2;",
        "ID      3;",
        "COVARIATE",
        "/* class variables */",
        f"sex         {2:5d}   4   4",
        f"generati{G_ped:5d}   5   5",
        f"animal  {N_total:5d}   6   6",
        f"share_hh{N_households:5d}   7   7",
        ";",
        " FREE_FORMAT;",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_pedigree(pedigree: pd.DataFrame, path: str) -> None:
    """Write pedigree.ped with columns: animal_id sex sire dam.

    Sex: 1=male, 2=female. All IDs are 1-based. Missing parents (founders) -> 0.
    """
    ped = pd.DataFrame()
    ped["id"] = pedigree["id"] + 1  # 1-based
    ped["sex"] = pedigree["sex"].map({0: 2, 1: 1})  # 0=F->2, 1=M->1
    # Parents: -1 (missing) -> 0, otherwise +1 for 1-based
    ped["sire"] = np.where(pedigree["father"].values == -1, 0, pedigree["father"].values + 1)
    ped["dam"] = np.where(pedigree["mother"].values == -1, 0, pedigree["mother"].values + 1)

    ped.to_csv(path, sep=" ", index=False, header=False)


def write_weibull_config(scenario: str, trait: str, rep: str, ndimax: int, nrecmax: int, path: str) -> None:
    """Write weibull.txt model specification.

    Variable names truncated to 8 characters to match varlist/codelist.
    """
    lines = [
        f"NDIMAX {ndimax};",
        f"NRECMAX {nrecmax};",
        "FILES data.dat codelist.txt results.rwe pedigree.ped;",
        f"TITLE {scenario} trait{trait} rep{rep} - WEIBULL MODEL;",
        "MODEL sex generati animal share_hh;",
        "RANDOM animal ESTIMATE MULTINORMAL USUAL_RULES 0.001 1.0 0.01;",
        "RANDOM share_hh ESTIMATE LOGGAMMA 0.1 10 0.1;",
        "INTEGRATE_OUT;",
        "DENSE_HESSIAN;",
        "ITE_QUASI 200;",
        "CONSTRAINT LARGEST;",
        "TEST SEQUENTIAL;",
        "STD_ERROR;",
        "STORAGE IN_CORE;",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_prepare(pedigree: pd.DataFrame, phenotype: pd.DataFrame, scenario: str, rep: str, trait: str, outputs: dict[str, str]) -> None:
    """Orchestrate preparation of Survival Kit input files.

    Args:
        pedigree: pedigree DataFrame
        phenotype: phenotype DataFrame
        scenario: scenario name
        rep: replicate number
        trait: trait number
        outputs: dict with keys: data, codelist, varlist, pedigree_ped, weibull_config
    """
    logger.info("Preparing Survival Kit files: scenario=%s, rep=%s, trait=%s", scenario, rep, trait)
    # Drop columns already in pedigree to avoid _x/_y suffix collision
    pheno_only = phenotype[[c for c in phenotype.columns if c not in pedigree.columns or c == "id"]]
    df = pedigree.merge(pheno_only, on="id")

    G_ped = pedigree["generation"].nunique()
    N_total = len(pedigree)
    N_households = pedigree["household_id"].nunique()
    ndimax = N_total + N_households + G_ped + 1000

    write_data(df, trait, outputs["data"])
    write_codelist(df, G_ped, outputs["codelist"])
    write_varlist(G_ped, N_total, N_households, outputs["varlist"])
    write_pedigree(pedigree, outputs["pedigree_ped"])
    nrecmax = 2 * ndimax
    write_weibull_config(scenario, trait, rep, ndimax, nrecmax, outputs["weibull_config"])


def cli() -> None:
    """Command-line interface for preparing Survival Kit input files."""
    from sim_ace import setup_logging
    parser = argparse.ArgumentParser(description="Prepare Survival Kit input files")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG output")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING+ only")
    parser.add_argument("--pedigree", required=True, help="Pedigree parquet path")
    parser.add_argument("--phenotype", required=True, help="Phenotype parquet path")
    parser.add_argument("--scenario", required=True, help="Scenario name")
    parser.add_argument("--rep", required=True, help="Replicate identifier")
    parser.add_argument("--trait", required=True, help="Trait name (e.g. trait1, trait2)")
    parser.add_argument("--data", required=True, help="Output data.dat path")
    parser.add_argument("--codelist", required=True, help="Output codelist.txt path")
    parser.add_argument("--varlist", required=True, help="Output varlist.txt path")
    parser.add_argument("--pedigree-ped", required=True, help="Output pedigree.ped path")
    parser.add_argument("--weibull-config", required=True, help="Output weibull.txt path")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=level)

    pedigree = pd.read_parquet(args.pedigree)
    phenotype = pd.read_parquet(args.phenotype)
    outputs = {
        "data": args.data,
        "codelist": args.codelist,
        "varlist": args.varlist,
        "pedigree_ped": args.pedigree_ped,
        "weibull_config": args.weibull_config,
    }
    run_prepare(pedigree, phenotype, args.scenario, args.rep, args.trait, outputs)
