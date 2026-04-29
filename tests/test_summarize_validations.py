"""
Aggregate simACE validation.yaml files into a summary table.

Expected structure:
  results/base/{baseline}/{rep}/validation.yaml

Usage:
  python aggregate_validation.py
  python aggregate_validation.py --results_dir path/to/results
"""

import yaml
import pandas as pd
from pathlib import Path
import argparse


def extract_row(yaml_path: Path) -> dict:
    """Extract relevant values from a single validation.yaml file."""
    with open(yaml_path) as f:
        d = yaml.safe_load(f)

    params = d.get("parameters", {})
    stat   = d.get("statistical", {})
    her    = d.get("heritability", {})
    summ   = d.get("summary", {})
    pop    = d.get("population", {})

    # Identify baseline and rep from the directory structure
    parts = yaml_path.parts
    rep      = parts[-2]   # e.g. rep1
    baseline = parts[-3]   # e.g. baseline1M

    row = {
        "baseline":  baseline,
        "rep":        rep,
        "N":          params.get("N"),

        # True parameters
        "true_h2_1":  params.get("A1"),
        "true_h2_2":  params.get("A2"),
        "true_c2_1":  params.get("C1"),
        "true_c2_2":  params.get("C2"),
        "true_rA":    params.get("rA"),

        # Observed estimates (Falconer)
        "obs_h2_1":   her.get("falconer_estimate_trait1", {}).get("observed"),
        "obs_h2_2":   her.get("falconer_estimate_trait2", {}).get("observed"),

        # Variance components (founders)
        "obs_c2_1":   stat.get("variance_C1", {}).get("observed"),
        "obs_c2_2":   stat.get("variance_C2", {}).get("observed"),
        "obs_rA":     stat.get("cross_trait_rA", {}).get("observed"),

        # MZ / DZ correlations
        "r_MZ_1":     her.get("mz_twin_liability1_correlation", {}).get("observed"),
        "r_MZ_2":     her.get("mz_twin_liability2_correlation", {}).get("observed"),
        "r_DZ_1":     her.get("dz_sibling_liability1_correlation", {}).get("observed"),
        "r_DZ_2":     her.get("dz_sibling_liability2_correlation", {}).get("observed"),

        # Validation checks
        "checks_passed": summ.get("checks_passed"),
        "checks_total":  summ.get("checks_total"),
        "all_passed":    summ.get("passed"),
    }

    return row


def main(results_dir: str = "results"):
    results_path = Path(results_dir)
    yaml_files = sorted(results_path.glob("**/validation.yaml"))

    if not yaml_files:
        print(f"No validation.yaml files found in: {results_path.resolve()}")
        return

    rows = []
    for yf in yaml_files:
        try:
            rows.append(extract_row(yf))
        except Exception as e:
            print(f"  [WARN] Error processing {yf}: {e}")

    df = pd.DataFrame(rows)

    # Sort by baseline (numeric order) and replication
    baseline_order = {
        "baseline10K": 1,
        "baseline100K": 2,
        "baseline1M": 3,
        "baseline2M": 4,
    }
    df["_order"] = df["baseline"].map(baseline_order).fillna(99)
    df = df.sort_values(["_order", "rep"]).drop(columns="_order")

    # --- Paper table: mean ± SD across replications ---
    numeric_cols = [
        "obs_h2_1", "obs_h2_2",
        "obs_c2_1", "obs_c2_2",
        "obs_rA",
        "r_MZ_1", "r_MZ_2",
        "r_DZ_1", "r_DZ_2",
    ]

    summary = (
        df.groupby("baseline")[numeric_cols]
        .agg(["mean", "std"])
        .round(4)
    )

    # Format "mean ± SD" for each column
    paper_rows = []
    for baseline, group in df.groupby("baseline", sort=False):
        row = {"baseline": baseline}
        for col in numeric_cols:
            m = group[col].mean()
            s = group[col].std()
            row[col] = f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f}"
        paper_rows.append(row)

    paper_df = pd.DataFrame(paper_rows)

    # Add true parameter columns (identical across reps within a baseline)
    true_cols = [
        "true_h2_1", "true_h2_2",
        "true_c2_1", "true_c2_2",
        "true_rA", "N",
    ]
    true_vals = (
        df.groupby("baseline")[true_cols]
        .first()
        .reset_index()
    )
    paper_df = paper_df.merge(true_vals, on="baseline")

    # Reorder columns
    cols = [
        "baseline", "N",
        "true_h2_1", "obs_h2_1",
        "true_h2_2", "obs_h2_2",
        "true_c2_1", "obs_c2_1",
        "true_c2_2", "obs_c2_2",
        "true_rA",   "obs_rA",
    ]
    paper_df = paper_df[[c for c in cols if c in paper_df.columns]]

    # Save outputs
    df.to_csv("results/base/validation_all.csv", index=False)
    paper_df.to_csv("results/base/validation_paper_table.csv", index=False)

    print("\n=== Paper table (mean ± SD across replications) ===")
    print(paper_df.to_string(index=False))
    print("\nSaved files:")
    print("  validation_all.csv          — all replication-level values")
    print("  validation_paper_table.csv — baseline-level summary table")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Root directory containing validation.yaml files (default: results)",
    )
    args = parser.parse_args()
    main(args.results_dir)