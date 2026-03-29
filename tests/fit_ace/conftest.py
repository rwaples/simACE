"""Shared fixtures for fit_ace tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def pafgrs_scores_dir(tmp_path_factory):
    """Write synthetic scores.parquet and metrics.tsv for plot_pafgrs tests."""
    d = tmp_path_factory.mktemp("pafgrs")
    rng = np.random.default_rng(99)
    n = 200

    data = {
        "id": np.arange(n),
        "generation": rng.choice([0, 1, 2], n),
    }
    for trait in ["trait1", "trait2"]:
        tnum = trait[-1]
        true_a = rng.normal(0, 1, n)
        affected = (true_a > np.percentile(true_a, 80)).astype(int)
        for cip in ["empirical", "true"]:
            for h2src in ["true", "estimated"]:
                tag = f"{trait}_{cip}_{h2src}"
                est = true_a + rng.normal(0, 0.5, n)
                data[f"est_{tag}"] = est
                data[f"var_{tag}"] = rng.uniform(0.1, 0.5, n)
                data[f"nrel_{tag}"] = rng.integers(0, 10, n)
        data[f"true_A{tnum}"] = true_a
        data[f"affected{tnum}"] = affected

    pd.DataFrame(data).to_parquet(d / "scores.parquet", index=False)

    # metrics.tsv in wide format
    metrics_rows = [
        {
            "trait": trait,
            "cip_source": cip,
            "h2_source": h2src,
            "r": rng.uniform(0.3, 0.8),
            "r2": rng.uniform(0.1, 0.6),
            "bias": rng.uniform(-0.1, 0.1),
            "auc": rng.uniform(0.6, 0.9),
            "var_calibration": rng.uniform(0.8, 1.2),
            "n_scored": n,
            "n_affected": int(n * 0.2),
        }
        for trait in ["trait1", "trait2"]
        for cip in ["empirical", "true"]
        for h2src in ["true", "estimated"]
    ]
    pd.DataFrame(metrics_rows).to_csv(d / "metrics.tsv", sep="\t", index=False)

    return d
