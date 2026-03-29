"""Tests for PA-FGRS diagnostic atlas."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from fit_ace.plotting.plot_pafgrs import _subsample, generate_atlas

# ---------------------------------------------------------------------------
# _subsample
# ---------------------------------------------------------------------------


class TestSubsample:
    def test_no_subsample_when_small(self):
        df = pd.DataFrame({"x": range(50)})
        result, note = _subsample(df, max_n=100, seed=42)
        assert len(result) == 50
        assert note == ""

    def test_subsamples_when_large(self):
        df = pd.DataFrame({"x": range(1000)})
        result, note = _subsample(df, max_n=100, seed=42)
        assert len(result) == 100
        assert "100" in note
        assert "1,000" in note

    def test_deterministic_with_seed(self):
        df = pd.DataFrame({"x": range(500)})
        r1, _ = _subsample(df, max_n=50, seed=42)
        r2, _ = _subsample(df, max_n=50, seed=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_different_seed_different_result(self):
        df = pd.DataFrame({"x": range(500)})
        r1, _ = _subsample(df, max_n=50, seed=42)
        r2, _ = _subsample(df, max_n=50, seed=99)
        assert not r1.reset_index(drop=True).equals(r2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# generate_atlas full smoke test
# ---------------------------------------------------------------------------


class TestGenerateAtlas:
    def test_creates_pdf(self, pafgrs_scores_dir, tmp_path):
        output = tmp_path / "atlas.pdf"
        before = plt.get_fignums()
        generate_atlas(str(pafgrs_scores_dir), str(output))
        assert output.exists()
        assert output.stat().st_size > 0
        assert plt.get_fignums() == before
