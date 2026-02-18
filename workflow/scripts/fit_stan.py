"""Fit the ACE pedigree model using CmdStanPy.

Usage:
    python workflow/scripts/fit_stan.py results/small_test/rep1/pedigree.parquet
"""

import shutil
import sys
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


def main(parquet_path):
    # Load and sort pedigree
    ped = pd.read_parquet(parquet_path).sort_values("id").reset_index(drop=True)

    Nped = len(ped)

    # Fit the last 3 generations
    last_gen = ped.generation.max()
    fit_gens = 3
    fit_mask = ped.generation >= (last_gen - fit_gens + 1)
    Nfit = fit_mask.sum()
    Ndrop = Nped - Nfit

    # Parent index mapping: ID -> 1-based position (0 for founders with id=-1)
    idx_of_id = pd.Series(ped.index + 1, index=ped.id)  # 1-based
    mother = ped.mother.map(idx_of_id).fillna(0).astype(int).values
    father = ped.father.map(idx_of_id).fillna(0).astype(int).values

    # Family indices for fit subset (1-based contiguous)
    fit_subset = ped.iloc[-Nfit:]
    _, fam_idx = np.unique(fit_subset.household_id.values, return_inverse=True)
    fam = fam_idx + 1
    Nfam = len(np.unique(fam))

    # Standardize phenotype
    y = fit_subset.liability1.values.copy()
    ystd = y.std()
    if ystd > 0:
        y = (y - y.mean()) / ystd
    else:
        y = y - y.mean()

    print(f"Nped={Nped}, Nfit={Nfit}, Ndrop={Ndrop}, Nfam={Nfam}")
    print(f"y mean={y.mean():.4f}, std={y.std():.4f}")

    # Henderson's D-inverse diagonal
    dii = compute_dii(ped)
    print(f"dii: {np.unique(dii, return_counts=True)}")

    pedigree_data = {
        "Nped": Nped,
        "Nfam": Nfam,
        "Nfit": Nfit,
        "Ndrop": Ndrop,
        "y": y,
        "fam": fam,
        "mother": mother,
        "father": father,
        "dii": dii,
    }

    # Compile Stan model
    stan_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fit_pedigree_ace.stan"
    )
    print(f"Compiling Stan model: {stan_file}")
    model = CmdStanModel(stan_file=stan_file)

    # ADVI — use draws=10 to minimize temp CSV size (4M params × draws rows)
    print("Running ADVI...")
    fit = model.variational(
        data=pedigree_data,
        seed=42,
        draws=10,
    )

    # Extract variational means for scalar parameters
    params = ["mu", "sigma_A", "sigma_C", "sigma_E", "V_A", "V_C", "V_E", "h2", "c2", "e2"]

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
    output_dir = os.path.dirname(parquet_path)
    output_path = os.path.join(output_dir, "stan_fit_summary.csv")
    param_summary.to_csv(output_path)
    print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fit_stan.py <path/to/pedigree.parquet>")
        sys.exit(1)
    main(sys.argv[1])
