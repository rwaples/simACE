"""Benchmark pipeline stages on baseline100K (rep1).

Times: simulation, phenotyping, stats (the numba-optimized hot path),
plot atlas generation, and PA-FGRS scoring.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import time

import numpy as np
import pandas as pd
import yaml

from sim_ace.core.utils import yaml_loader


def timed(label, func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"  {label:45s} {dt:8.2f}s")
    return result, dt


def bench_simulate():
    """Time pedigree simulation."""
    from sim_ace.simulation.simulate import run_simulation

    params = {
        "seed": 2042,
        "N": 100_000,
        "G_ped": 6,
        "G_sim": 8,
        "mating_lambda": 0.5,
        "p_mztwin": 0.02,
        "A1": 0.5,
        "C1": 0.1,
        "A2": 0.4,
        "C2": 0.2,
        "rA": 0.3,
        "rC": 0.2,
        "assort1": 0.0,
        "assort2": 0.0,
        "assort_matrix": None,
    }
    _, dt = timed("simulate (100K, G_ped=6, G_sim=8)", run_simulation, **params)
    return dt


def bench_phenotype():
    """Time phenotyping on existing pedigree."""
    from sim_ace.phenotyping.phenotype import phenotype_adult_ltm

    df = pd.read_parquet("results/base/baseline100K/rep1/pedigree.parquet")
    liability = df["liability1"].values
    sex = df["sex"].values

    _, dt = timed(
        "phenotype_adult_ltm (trait 1)",
        phenotype_adult_ltm,
        liability,
        prevalence=0.10,
        beta=1.0,
        cip_x0=50.0,
        cip_k=0.1,
        sex=sex,
        beta_sex=0.0,
    )
    return dt


def bench_stats():
    """Time the stats hot paths: pair extraction, tetrachoric, correlations."""
    from sim_ace.analysis.stats import (
        compute_cross_trait_tetrachoric,
        compute_liability_correlations,
        compute_parent_offspring_corr,
        compute_regression,
        compute_tetrachoric,
        compute_tetrachoric_by_generation,
    )
    from sim_ace.core.pedigree_graph import extract_relationship_pairs

    df = pd.read_parquet("results/base/baseline100K/rep1/phenotype.parquet")
    timings = {}

    _, timings["pair_extraction"] = timed("pair extraction", extract_relationship_pairs, df, seed=42)
    pairs = extract_relationship_pairs(df, seed=42)

    _, timings["regression"] = timed("compute_regression", compute_regression, df)
    _, timings["liability_corr"] = timed(
        "compute_liability_correlations", compute_liability_correlations, df, pairs=pairs
    )
    _, timings["parent_offspring"] = timed("compute_parent_offspring_corr", compute_parent_offspring_corr, df)
    _, timings["tetrachoric"] = timed("compute_tetrachoric (same-trait)", compute_tetrachoric, df, pairs=pairs)
    _, timings["tetrachoric_gen"] = timed(
        "compute_tetrachoric_by_generation", compute_tetrachoric_by_generation, df, pairs=pairs
    )
    _, timings["cross_trait"] = timed(
        "compute_cross_trait_tetrachoric", compute_cross_trait_tetrachoric, df, pairs=pairs
    )

    total = sum(timings.values())
    print(f"  {'stats total (excl frailty)':45s} {total:8.2f}s")
    return total


def bench_atlas():
    """Time plot generation via snakemake force-rebuild."""
    import subprocess
    import tempfile

    # Generate plots into a temp dir to avoid clobbering real results
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()
        result = subprocess.run(
            [
                "python",
                "-m",
                "sim_ace.plotting.plot_phenotype",
                "--stats",
                "results/base/baseline100K/rep1/phenotype_stats.yaml",
                "--samples",
                "results/base/baseline100K/rep1/phenotype.parquet",
                "--validation",
                "results/base/baseline100K/rep1/validation.yaml",
                "--output-dir",
                tmpdir,
                "--censor-age",
                "80",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        dt = time.perf_counter() - t0
        if result.returncode != 0:
            # Try alternative: just time the individual plot functions
            print("  plot via CLI failed, timing individual functions...")
            dt = _bench_plot_functions()
        else:
            print(f"  {'plot phenotype (all plots)':45s} {dt:8.2f}s")
    return dt


def _bench_plot_functions():
    """Time the key plot functions individually."""
    from sim_ace.plotting.plot_correlations import (
        plot_parent_offspring_liability,
        plot_tetrachoric_sibling,
    )
    from sim_ace.plotting.plot_distributions import plot_trait_regression

    with open("results/base/baseline100K/rep1/phenotype_stats.yaml") as f:
        stats = yaml.load(f, Loader=yaml_loader())
    df = pd.read_parquet("results/base/baseline100K/rep1/phenotype.parquet")
    # Subsample for plotting
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), min(10_000, len(df)), replace=False)
    df_sub = df.iloc[idx]

    import tempfile

    total = 0.0
    with tempfile.TemporaryDirectory() as tmpdir:
        _, dt = timed(
            "plot_tetrachoric_sibling", plot_tetrachoric_sibling, [stats], f"{tmpdir}/tet.png", "baseline100K"
        )
        total += dt
        _, dt = timed(
            "plot_trait_regression", plot_trait_regression, df_sub, [stats], f"{tmpdir}/reg.png", "baseline100K"
        )
        total += dt
        _, dt = timed(
            "plot_parent_offspring_liability",
            plot_parent_offspring_liability,
            df_sub,
            [stats],
            f"{tmpdir}/po.png",
            "baseline100K",
        )
        total += dt
    print(f"  {'plots total (3 key functions)':45s} {total:8.2f}s")
    return total


def bench_pafgrs():
    """Time PA-FGRS threshold computation + scoring."""
    from fit_ace.pafgrs.pafgrs import compute_thresholds_and_w

    df = pd.read_parquet("results/base/baseline100K/rep1/phenotype.parquet")
    if "affected1" not in df.columns:
        print("  PA-FGRS: skipped (no affected1)")
        return 0.0

    affected = df["affected1"].values.astype(bool)
    t_obs = df["t_observed1"].values
    # Dummy CIP table for benchmarking thresholds computation
    cip_ages = np.arange(0, 90, 1, dtype=float)
    cip_values = 0.10 * (1 - np.exp(-0.03 * cip_ages))

    # Warm up
    compute_thresholds_and_w(affected, t_obs, cip_ages, cip_values, 0.10)

    _, dt = timed(
        "compute_thresholds_and_w (100K)", compute_thresholds_and_w, affected, t_obs, cip_ages, cip_values, 0.10
    )
    return dt


def main():
    print("=" * 60)
    print("Pipeline benchmark: baseline100K (rep1)")
    print("=" * 60)

    timings = {}

    print("\n--- Simulation ---")
    timings["simulate"] = bench_simulate()

    print("\n--- Phenotyping ---")
    timings["phenotype"] = bench_phenotype()

    print("\n--- Stats ---")
    timings["stats"] = bench_stats()

    print("\n--- Plot Atlas ---")
    timings["atlas"] = bench_atlas()

    print("\n--- PA-FGRS ---")
    try:
        timings["pafgrs"] = bench_pafgrs()
    except Exception as e:
        print(f"  PA-FGRS: error ({e})")
        timings["pafgrs"] = 0.0

    print("\n" + "=" * 60)
    total = sum(timings.values())
    print(f"{'Total':47s} {total:8.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
