"""Fit the ACE pedigree model using variational inference via CmdStanPy.

Uses a Stan model with profiled-out mean and REML correction term.
ADVI (Automatic Differentiation Variational Inference) approximates
the full posterior, properly integrating out the latent effects.

Usage:
    python workflow/scripts/fit_reml.py results/small_test/rep1/pedigree.parquet
"""

import sys
import os
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


def main(parquet_path):
    # Load and sort pedigree
    ped = pd.read_parquet(parquet_path).sort_values("id").reset_index(drop=True)

    Nped = len(ped)

    # Fit the last generation
    last_gen = ped.generation.max()
    Nfit = len(ped[ped.generation == last_gen])
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

    pedigree_data = {
        "Nped": Nped,
        "Nfam": Nfam,
        "Nfit": Nfit,
        "Ndrop": Ndrop,
        "y": y,
        "fam": fam,
        "mother": mother,
        "father": father,
    }

    # Compile Stan model
    stan_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fit_pedigree_ace_reml.stan"
    )
    print(f"Compiling Stan model: {stan_file}")
    model = CmdStanModel(stan_file=stan_file)

    # Run ADVI variational inference
    print("Running ADVI variational inference...")
    vb = model.variational(
        data=pedigree_data,
        seed=42,
        draws=4000,
        algorithm="meanfield",
    )

    # Extract variational draws
    draws = vb.variational_sample

    print("\n=== Variational Bayes Estimates ===")
    keys = ["sigma_A", "sigma_C", "sigma_E", "V_A", "V_C", "V_E", "V_T", "h2", "c2", "e2"]
    col_names = list(vb.column_names)
    results = {}
    for k in keys:
        if k in col_names:
            idx = col_names.index(k)
            vals = draws[:, idx]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            q5 = np.quantile(vals, 0.05)
            q50 = np.quantile(vals, 0.50)
            q95 = np.quantile(vals, 0.95)
            results[k] = mean_val
            print(f"  {k:>8s}: mean={mean_val:.4f}  std={std_val:.4f}  "
                  f"median={q50:.4f}  90%CI=[{q5:.4f}, {q95:.4f}]")

    # Also show variational mean (optimized q)
    print("\n=== Variational Mean ===")
    vmean = vb.variational_params_dict
    for k in keys:
        if k in vmean:
            print(f"  {k:>8s} = {vmean[k]:.4f}")

    # Save summary
    output_dir = os.path.dirname(parquet_path)
    output_path = os.path.join(output_dir, "reml_fit_summary.csv")
    results_df = pd.DataFrame({"parameter": list(results.keys()), "estimate": list(results.values())})
    results_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fit_reml.py <path/to/pedigree.parquet>")
        sys.exit(1)
    main(sys.argv[1])
