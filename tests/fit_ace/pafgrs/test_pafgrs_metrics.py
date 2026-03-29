"""Tests for PA-FGRS validation metrics."""

import numpy as np
import pandas as pd
import pytest

from fit_ace.pafgrs.pafgrs_metrics import (
    _fast_auc,
    compute_pafgrs_metrics,
    read_and_combine_metrics,
    write_metrics_tsv,
)

# ---------------------------------------------------------------------------
# _fast_auc
# ---------------------------------------------------------------------------


class TestFastAuc:
    def test_perfect_separation(self):
        scores = np.array([0.0, 0.1, 0.9, 1.0])
        labels = np.array([False, False, True, True])
        assert _fast_auc(scores, labels) == pytest.approx(1.0)

    def test_inverse_separation(self):
        scores = np.array([0.9, 1.0, 0.0, 0.1])
        labels = np.array([False, False, True, True])
        assert _fast_auc(scores, labels) == pytest.approx(0.0)

    def test_random_near_half(self):
        rng = np.random.default_rng(42)
        n = 10_000
        scores = rng.normal(0, 1, n)
        labels = rng.choice([True, False], n)
        auc = _fast_auc(scores, labels)
        assert abs(auc - 0.5) < 0.05

    def test_ties_handled(self):
        # All same score -> AUC should be 0.5
        scores = np.array([1.0, 1.0, 1.0, 1.0])
        labels = np.array([False, False, True, True])
        assert _fast_auc(scores, labels) == pytest.approx(0.5)

    def test_agrees_with_mannwhitneyu(self):
        from scipy.stats import mannwhitneyu

        rng = np.random.default_rng(7)
        scores = rng.normal(0, 1, 200)
        labels = np.zeros(200, dtype=bool)
        labels[:50] = True
        scores[labels] += 0.8  # shift positives up

        auc_ours = _fast_auc(scores, labels)
        u, _ = mannwhitneyu(scores[labels], scores[~labels], alternative="greater")
        auc_ref = u / (labels.sum() * (~labels).sum())
        assert auc_ours == pytest.approx(auc_ref, abs=1e-10)


# ---------------------------------------------------------------------------
# compute_pafgrs_metrics
# ---------------------------------------------------------------------------


class TestComputePafgrsMetrics:
    def test_output_keys(self):
        df = pd.DataFrame({
            "est": [1.0, 2.0, 3.0],
            "true_A": [1.0, 2.0, 3.0],
            "var": [0.01, 0.01, 0.01],
            "affected": [0, 0, 1],
        })
        m = compute_pafgrs_metrics(df)
        expected = {"r", "r2", "bias", "auc", "var_calibration",
                    "mean_reported_var", "mean_actual_mse", "n_scored", "n_affected"}
        assert set(m.keys()) == expected

    def test_perfect_estimator(self):
        rng = np.random.default_rng(42)
        n = 500
        true_a = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "est": true_a,
            "true_A": true_a,
            "var": np.full(n, 0.01),
            "affected": (true_a > 0).astype(int),
        })
        m = compute_pafgrs_metrics(df)
        assert m["r"] == pytest.approx(1.0, abs=1e-5)
        assert m["r2"] == pytest.approx(1.0, abs=1e-5)
        assert m["bias"] == pytest.approx(0.0, abs=1e-10)

    def test_random_estimator(self):
        rng = np.random.default_rng(42)
        n = 5000
        df = pd.DataFrame({
            "est": rng.normal(0, 1, n),
            "true_A": rng.normal(0, 1, n),
            "var": np.full(n, 0.5),
            "affected": rng.choice([0, 1], n),
        })
        m = compute_pafgrs_metrics(df)
        assert abs(m["r"]) < 0.05

    def test_constant_estimator(self):
        df = pd.DataFrame({
            "est": [0.0, 0.0, 0.0, 0.0],
            "true_A": [1.0, 2.0, 3.0, 4.0],
            "var": [0.01, 0.01, 0.01, 0.01],
            "affected": [0, 0, 1, 1],
        })
        m = compute_pafgrs_metrics(df)
        assert m["r"] == 0.0  # zero variance in est
        assert m["bias"] == pytest.approx(-2.5)

    def test_auc_discriminates(self):
        rng = np.random.default_rng(42)
        n = 1000
        true_a = rng.normal(0, 1, n)
        affected = (true_a > np.percentile(true_a, 80)).astype(int)
        df = pd.DataFrame({
            "est": true_a + rng.normal(0, 0.3, n),
            "true_A": true_a,
            "var": np.full(n, 0.1),
            "affected": affected,
        })
        m = compute_pafgrs_metrics(df)
        assert m["auc"] > 0.7

    def test_no_affected_auc_nan(self):
        df = pd.DataFrame({
            "est": [1.0, 2.0, 3.0],
            "true_A": [1.0, 2.0, 3.0],
            "var": [0.01, 0.01, 0.01],
            "affected": [0, 0, 0],
        })
        m = compute_pafgrs_metrics(df)
        assert np.isnan(m["auc"])

    def test_n_scored_matches(self):
        df = pd.DataFrame({
            "est": [1.0, 2.0],
            "true_A": [1.0, 2.0],
            "var": [0.01, 0.01],
            "affected": [0, 1],
        })
        m = compute_pafgrs_metrics(df)
        assert m["n_scored"] == 2
        assert m["n_affected"] == 1


# ---------------------------------------------------------------------------
# write_metrics_tsv / read_and_combine_metrics
# ---------------------------------------------------------------------------


class TestWriteMetricsTsv:
    def test_creates_file(self, tmp_path):
        metrics = {"r": 0.5, "auc": 0.7}
        path = tmp_path / "metrics.tsv"
        write_metrics_tsv(metrics, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_tsv_columns(self, tmp_path):
        metrics = {"r": 0.5}
        path = tmp_path / "metrics.tsv"
        write_metrics_tsv(metrics, path, trait="trait2", cip_source="true", h2_source="estimated")
        df = pd.read_csv(path, sep="\t")
        assert set(df.columns) == {"trait", "cip_source", "h2_source", "metric", "value"}
        assert df.iloc[0]["trait"] == "trait2"

    def test_parent_dirs_created(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "metrics.tsv"
        write_metrics_tsv({"r": 0.5}, path)
        assert path.exists()

    def test_all_metrics_written(self, tmp_path):
        metrics = {"r": 0.5, "r2": 0.25, "bias": 0.01}
        path = tmp_path / "metrics.tsv"
        write_metrics_tsv(metrics, path)
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 3


class TestReadAndCombineMetrics:
    def test_roundtrip(self, tmp_path):
        metrics = {"r": 0.5, "auc": 0.7}
        path = tmp_path / "metrics.tsv"
        write_metrics_tsv(metrics, path)
        df = read_and_combine_metrics([path])
        assert len(df) == 2
        assert "metric" in df.columns

    def test_combine_two_files(self, tmp_path):
        p1 = tmp_path / "m1.tsv"
        p2 = tmp_path / "m2.tsv"
        write_metrics_tsv({"r": 0.5}, p1, trait="trait1")
        write_metrics_tsv({"r": 0.8}, p2, trait="trait2")
        df = read_and_combine_metrics([p1, p2])
        assert len(df) == 2
        assert set(df["trait"]) == {"trait1", "trait2"}

    def test_missing_file_skipped(self, tmp_path):
        p1 = tmp_path / "exists.tsv"
        write_metrics_tsv({"r": 0.5}, p1)
        df = read_and_combine_metrics([p1, tmp_path / "nope.tsv"])
        assert len(df) == 1

    def test_all_missing_returns_empty(self, tmp_path):
        df = read_and_combine_metrics([tmp_path / "a.tsv", tmp_path / "b.tsv"])
        assert df.empty
