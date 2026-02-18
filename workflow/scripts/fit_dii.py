"""Fit the ACE pedigree model using the dii (non-centered) Stan model.

Usage:
    python workflow/scripts/fit_dii.py <pedigree.parquet> [--profile [W S]] [--advi]

Examples:
    # ADVI only (default)
    python workflow/scripts/fit_dii.py results/baseline10K/rep1/pedigree.parquet

    # Profile MCMC only (default 10+10 iters)
    python workflow/scripts/fit_dii.py results/baseline10K/rep1/pedigree.parquet --profile

    # Profile MCMC with custom warmup/sampling
    python workflow/scripts/fit_dii.py results/baseline10K/rep1/pedigree.parquet --profile 1000 1000

    # Both
    python workflow/scripts/fit_dii.py results/baseline10K/rep1/pedigree.parquet --profile --advi
"""

import argparse
import shutil
import os
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


def compute_dii(ped):
    """Compute Henderson's D-inverse diagonal (non-inbred pedigree).

    dii[i] = 1 / (1 - 0.25 * n_known_parents[i])
    Founders: 1.0, one parent known: 4/3, both parents known: 2.0
    """
    n_parents = (ped.mother.values != -1).astype(int) + (ped.father.values != -1).astype(int)
    dii = 1.0 / (1.0 - 0.25 * n_parents)
    return dii


def prepare_data(parquet_path):
    """Load pedigree and build Stan data dict."""
    ped = pd.read_parquet(parquet_path).sort_values("id").reset_index(drop=True)

    N = len(ped)

    # Fit the last 3 generations
    last_gen = ped.generation.max()
    fit_gens = 3
    fit_mask = ped.generation >= (last_gen - fit_gens + 1)
    Npheno = int(fit_mask.sum())
    Ndrop = N - Npheno

    # Parent index mapping: ID -> 1-based position (0 for unknown)
    idx_of_id = pd.Series(ped.index + 1, index=ped.id)  # 1-based
    mother = ped.mother.map(idx_of_id).fillna(0).astype(int).values
    father = ped.father.map(idx_of_id).fillna(0).astype(int).values

    # pid: 1-based indices of phenotyped individuals into full pedigree
    pid = np.arange(Ndrop + 1, N + 1, dtype=int)

    # Household indices for fit subset (1-based contiguous)
    fit_subset = ped.iloc[-Npheno:]
    _, hh_idx = np.unique(fit_subset.household_id.values, return_inverse=True)
    household = hh_idx + 1
    H = int(household.max())

    # Standardize phenotype
    y = fit_subset.liability1.values.copy()
    ystd = y.std()
    if ystd > 0:
        y = (y - y.mean()) / ystd
    else:
        y = y - y.mean()

    # Henderson's D-inverse diagonal
    dii = compute_dii(ped)

    # Tier boundaries for vectorized recursion
    unique_gens = np.sort(ped.generation.unique())
    G = len(unique_gens)
    gen_values = ped.generation.values
    tier_start = np.empty(G + 1, dtype=int)
    for k, gen in enumerate(unique_gens):
        tier_start[k] = int(np.searchsorted(gen_values, gen, side="left")) + 1  # 1-based
    tier_start[G] = N + 1

    print(f"N={N}, Npheno={Npheno}, Ndrop={Ndrop}, H={H}, G={G}")
    print(f"tier_start={tier_start.tolist()}")
    print(f"y mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"dii: {np.unique(dii, return_counts=True)}")

    stan_data = {
        "N": N,
        "mother": mother,
        "father": father,
        "dii": dii,
        "Npheno": Npheno,
        "pid": pid,
        "y": y,
        "H": H,
        "household": household,
        "G": G,
        "tier_start": tier_start,
    }

    return stan_data, os.path.dirname(parquet_path)


def run_profile(model, stan_data, inits, warmup, sampling):
    """Run short MCMC for profiling."""
    print(f"\nRunning MCMC profiling ({warmup} warmup + {sampling} sampling)...")
    profile_fit = model.sample(
        data=stan_data,
        seed=42,
        chains=1,
        iter_warmup=warmup,
        iter_sampling=sampling,
        inits=inits,
    )
    # Profile CSV is written to cwd by CmdStan (default: profile.csv)
    profile_csv = os.path.join(os.getcwd(), "profile.csv")
    if os.path.exists(profile_csv):
        print(f"\n=== Profiling (1 chain, {warmup}+{sampling} iters) ===")
        prof_df = pd.read_csv(profile_csv)
        print(prof_df.to_string())
        os.remove(profile_csv)
    else:
        print("\nNo profile.csv found")

    prof_dir = os.path.dirname(profile_fit.runset.csv_files[0])
    shutil.rmtree(prof_dir, ignore_errors=True)


def run_advi(model, stan_data, inits, output_dir):
    """Run ADVI and save parameter summary."""
    print("\nRunning ADVI...")
    fit = model.variational(
        data=stan_data,
        seed=42,
        draws=20,
        sig_figs=4,
        grad_samples=2,
        inits=inits,
        require_converged=False,
    )

    # Extract variational means for scalar parameters
    params = ["mu", "sigma_A", "sigma_H", "sigma_E", "V_A", "V_H", "V_E", "h2", "c2", "e2"]

    print("\n=== Parameter Estimates (ADVI) ===")
    results = []
    for p in params:
        try:
            val = fit.stan_variable(p, mean=True)
        except Exception:
            continue
        results.append({"param": p, "mean": float(val)})
        print(f"{p:10s}: mean={float(val):.4f}")

    param_summary = pd.DataFrame(results).set_index("param")

    # Clean up temp Stan output files
    runset_dir = os.path.dirname(fit.runset.csv_files[0])
    shutil.rmtree(os.path.dirname(runset_dir), ignore_errors=True)

    # Save summary
    output_path = os.path.join(output_dir, "dii_fit_summary.csv")
    param_summary.to_csv(output_path)
    print(f"\nSummary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit ACE pedigree model (dii parameterization)")
    parser.add_argument("parquet", help="Path to pedigree.parquet")
    parser.add_argument(
        "--profile", nargs="*", type=int, metavar=("WARMUP", "SAMPLING"),
        help="Run MCMC profiling. Optional: warmup and sampling counts (default: 10 10)",
    )
    parser.add_argument("--advi", action="store_true", help="Run ADVI inference")
    args = parser.parse_args()

    # Default to --advi if neither flag given
    if args.profile is None and not args.advi:
        args.advi = True

    # Parse profile warmup/sampling
    if args.profile is not None:
        if len(args.profile) == 0:
            warmup, sampling = 10, 10
        elif len(args.profile) == 2:
            warmup, sampling = args.profile
        else:
            parser.error("--profile takes 0 or 2 arguments (warmup sampling)")

    stan_data, output_dir = prepare_data(args.parquet)

    # Compile Stan model
    stan_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ace_dii.stan"
    )
    print(f"Compiling Stan model: {stan_file}")
    model = CmdStanModel(stan_file=stan_file)

    inits = {"sigma_A": 0.5, "sigma_H": 0.2, "sigma_E": 0.2}

    if args.profile is not None:
        run_profile(model, stan_data, inits, warmup, sampling)

    if args.advi:
        run_advi(model, stan_data, inits, output_dir)


if __name__ == "__main__":
    main()
