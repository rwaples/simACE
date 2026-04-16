"""Unit tests for sim_ace.analysis.export_tables."""

from __future__ import annotations

import json
import struct

import numpy as np
import pandas as pd
import pytest

from sim_ace.analysis.export_grm import ACE_SREML_MAGIC
from sim_ace.analysis.export_tables import (
    _min_max_degree_for_kinship,
    assign_founder_family_ids,
    export_cumulative_incidence,
    export_pairwise_relatedness,
    export_pgs,
    export_sparse_grm,
)
from sim_ace.core.pedigree_graph import PAIR_KINSHIP, count_relationship_pairs
from sim_ace.simulation.simulate import generate_correlated_components

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _three_family_pedigree() -> pd.DataFrame:
    """Three disconnected founder couples, each with a pair of full sibs.

    Founders (gen 0): ids 0,1 (couple A); 2,3 (couple B); 4,5 (couple C).
    Offspring (gen 1): ids 6,7 (from A); 8,9 (from B); 10,11 (from C).
    Expected FIDs by component: {0,1,6,7}, {2,3,8,9}, {4,5,10,11} — min ids
    0, 2, 4 respectively.
    """
    return pd.DataFrame(
        {
            "id": np.arange(12, dtype=np.int32),
            # 0=F, 1=M
            "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8),
            "mother": np.array([-1, -1, -1, -1, -1, -1, 0, 0, 2, 2, 4, 4], dtype=np.int32),
            "father": np.array([-1, -1, -1, -1, -1, -1, 1, 1, 3, 3, 5, 5], dtype=np.int32),
            "twin": np.full(12, -1, dtype=np.int32),
            "generation": np.array([0] * 6 + [1] * 6, dtype=np.int32),
        }
    )


def _read_grm_bin_header(path):
    with open(path, "rb") as fh:
        magic = fh.read(len(ACE_SREML_MAGIC))
        n, nnz = struct.unpack("<qq", fh.read(16))
    return magic, n, nnz


# ---------------------------------------------------------------------------
# Cumulative incidence
# ---------------------------------------------------------------------------


def _synthetic_phenotype(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """Minimal phenotype-shaped DataFrame sufficient for the incidence exporter.

    Mirrors only the columns read by
    ``compute_cumulative_incidence_by_sex_generation``: ``sex``,
    ``generation``, ``affected{1,2}``, ``t_observed{1,2}``.
    """
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, size=n, dtype=np.int8)
    generation = rng.integers(2, 5, size=n, dtype=np.int32)

    # Realistic-ish observation times: event times up to 80, some censored
    # at 80.  Affected status drives whether ``t_observed`` is an event
    # time or a censoring time.
    df = pd.DataFrame({"sex": sex, "generation": generation})
    for trait_num in (1, 2):
        affected = rng.random(n) < 0.3
        onset = rng.uniform(10, 75, size=n)
        censor_age = 80.0
        t_observed = np.where(affected, onset, censor_age)
        df[f"affected{trait_num}"] = affected
        df[f"t_observed{trait_num}"] = t_observed
    return df


class TestExportCumulativeIncidence:
    def test_long_shape_and_columns(self, tmp_path):
        phen = _synthetic_phenotype()
        out = export_cumulative_incidence(phen, censor_age=80.0, out_path=tmp_path / "ci.tsv", n_points=50)
        df = pd.read_csv(out, sep="\t")

        assert list(df.columns) == [
            "trait",
            "sex",
            "generation",
            "age",
            "cum_incidence",
            "n_at_risk",
        ]
        assert df["trait"].isin({1, 2}).all()
        assert df["sex"].isin({"F", "M"}).all()
        assert (df["cum_incidence"] >= 0).all()
        assert (df["cum_incidence"] <= 1).all()

        # Monotone non-decreasing cum_incidence and non-increasing n_at_risk
        # within each (trait, sex, generation) stratum.
        for _, sub in df.groupby(["trait", "sex", "generation"], sort=False):
            sub_sorted = sub.sort_values("age")
            assert sub_sorted["cum_incidence"].is_monotonic_increasing
            assert sub_sorted["n_at_risk"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Pairwise relatedness
# ---------------------------------------------------------------------------


class TestExportPairwiseRelatedness:
    def test_counts_match_extractor_at_zero_threshold(self, tmp_path, small_pedigree):
        # min_kinship=0 asks for every canonical pair up to the maximum
        # degree covered in REL_REGISTRY (degree 5).
        out = export_pairwise_relatedness(small_pedigree, out_path=tmp_path / "pairs.tsv", min_kinship=0.0)
        df = pd.read_csv(out, sep="\t")

        # Reference counts come from the untouched extractor API.
        expected = count_relationship_pairs(small_pedigree, max_degree=5)
        observed = df["rel_code"].value_counts().to_dict()
        for code, n_expected in expected.items():
            if n_expected == 0:
                continue
            assert observed.get(code, 0) == n_expected, code

    def test_kinship_filter(self, tmp_path, small_pedigree):
        thresholds = [0.0, 0.0625, 0.125, 0.25]
        previous_codes: set[str] | None = None
        for t in thresholds:
            out = export_pairwise_relatedness(small_pedigree, out_path=tmp_path / f"pairs_{t}.tsv", min_kinship=t)
            df = pd.read_csv(out, sep="\t")
            if not df.empty:
                assert (df["kinship"] >= t).all(), t
            codes_present = set(df["rel_code"].unique())
            # Raising threshold can only drop codes, never add them.
            if previous_codes is not None:
                assert codes_present.issubset(previous_codes), (t, codes_present - previous_codes)
            previous_codes = codes_present

    def test_kinship_matches_lookup_non_inbred(self, tmp_path):
        # Use the disconnected 3-family fixture — guaranteed non-inbred,
        # no duplicate relationship paths between the same pair.
        ped = _three_family_pedigree()
        out = export_pairwise_relatedness(ped, out_path=tmp_path / "pairs.tsv", min_kinship=0.0)
        df = pd.read_csv(out, sep="\t")
        assert not df.empty
        expected = df["rel_code"].map(PAIR_KINSHIP).to_numpy()
        np.testing.assert_allclose(df["kinship"].to_numpy(), expected, atol=1e-12)

    def test_empty_when_threshold_exceeds_all(self, tmp_path, small_pedigree):
        # Nothing exceeds MZ's self-kinship of 0.5; the file is empty.
        out = export_pairwise_relatedness(small_pedigree, out_path=tmp_path / "pairs.tsv", min_kinship=1.0)
        df = pd.read_csv(out, sep="\t")
        assert len(df) == 0
        assert list(df.columns) == ["id1", "id2", "rel_code", "kinship"]

    def test_id_ordering(self, tmp_path, small_pedigree):
        out = export_pairwise_relatedness(small_pedigree, out_path=tmp_path / "pairs.tsv", min_kinship=0.125)
        df = pd.read_csv(out, sep="\t")
        if not df.empty:
            assert (df["id1"] < df["id2"]).all()

    def test_min_max_degree_helper(self):
        # kinship 0 → all codes → max degree in registry
        assert _min_max_degree_for_kinship(0.0) >= 5
        # 1/16 → up to degree 3
        assert _min_max_degree_for_kinship(0.0625) == 3
        # 1/8 → up to degree 2
        assert _min_max_degree_for_kinship(0.125) == 2
        # threshold above MZ (0.5) → no codes → 0
        assert _min_max_degree_for_kinship(1.0) == 0


# ---------------------------------------------------------------------------
# Founder-family IDs
# ---------------------------------------------------------------------------


class TestFounderFamilyIds:
    def test_disconnected_families(self):
        ped = _three_family_pedigree()
        fids = assign_founder_family_ids(ped)

        assert len(fids) == len(ped)
        # Three components → three distinct FIDs.
        assert fids.nunique() == 3

        # Members of each couple share an FID with their children.
        assert fids.iloc[0] == fids.iloc[1] == fids.iloc[6] == fids.iloc[7]
        assert fids.iloc[2] == fids.iloc[3] == fids.iloc[8] == fids.iloc[9]
        assert fids.iloc[4] == fids.iloc[5] == fids.iloc[10] == fids.iloc[11]
        # Distinct families.
        assert fids.iloc[0] != fids.iloc[2]
        assert fids.iloc[2] != fids.iloc[4]

        # FID = min id within each component.
        assert fids.iloc[0] == 0
        assert fids.iloc[2] == 2
        assert fids.iloc[4] == 4

    def test_marriage_merges_families(self):
        """A gen-1 child from couple A marrying a gen-1 child from couple B
        collapses both founder couples into one family."""
        ped = _three_family_pedigree().copy()
        # Add a shared child (id=12) of individual 7 (from A) and 8 (from B).
        ped = pd.concat(
            [
                ped,
                pd.DataFrame(
                    {
                        "id": [12],
                        "sex": [0],
                        "mother": [8],
                        "father": [7],
                        "twin": [-1],
                        "generation": [2],
                    }
                ),
            ],
            ignore_index=True,
        )
        fids = assign_founder_family_ids(ped)
        # Couples A and B now merged; C remains separate.
        assert fids.iloc[0] == fids.iloc[2]
        assert fids.iloc[0] != fids.iloc[4]
        assert fids.nunique() == 2

    def test_real_pedigree_runs(self, small_pedigree):
        fids = assign_founder_family_ids(small_pedigree)
        assert len(fids) == len(small_pedigree)
        assert fids.dtype == np.int64
        assert (fids >= 0).all()


# ---------------------------------------------------------------------------
# Sparse GRM
# ---------------------------------------------------------------------------


class TestExportSparseGrm:
    def test_files_written(self, tmp_path, small_pedigree):
        prefix = tmp_path / "grm" / "sparse"
        bin_path, id_path = export_sparse_grm(small_pedigree, prefix=prefix, threshold=0.05)
        assert bin_path.exists()
        assert id_path.exists()

        magic, n, nnz = _read_grm_bin_header(bin_path)
        assert magic == ACE_SREML_MAGIC
        assert n == len(small_pedigree)
        assert nnz > 0

    def test_id_file_has_founder_fids(self, tmp_path, small_pedigree):
        prefix = tmp_path / "sparse"
        _, id_path = export_sparse_grm(small_pedigree, prefix=prefix, threshold=0.05)

        expected_fids = assign_founder_family_ids(small_pedigree).to_numpy()
        expected_iids = small_pedigree["id"].to_numpy()

        with id_path.open() as fh:
            lines = [line.rstrip("\n").split("\t") for line in fh]
        assert len(lines) == len(small_pedigree)

        actual_fids = np.array([int(fid) for fid, _ in lines], dtype=np.int64)
        actual_iids = np.array([int(iid) for _, iid in lines], dtype=np.int64)

        np.testing.assert_array_equal(actual_fids, expected_fids)
        np.testing.assert_array_equal(actual_iids, expected_iids)

        # At least some individuals share a FID — otherwise we haven't
        # actually applied founder-family grouping.
        assert len(np.unique(actual_fids)) < len(actual_fids)


# ---------------------------------------------------------------------------
# Proxy polygenic score
# ---------------------------------------------------------------------------


def _synthetic_A_frame(n: int, var_A: tuple[float, float], rA: float, seed: int) -> pd.DataFrame:
    """Pedigree-shaped frame with analytically-known A1, A2 draws."""
    rng = np.random.default_rng(seed)
    sd1 = float(np.sqrt(var_A[0]))
    sd2 = float(np.sqrt(var_A[1]))
    A1, A2 = generate_correlated_components(rng, n, sd1, sd2, rA)
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int32),
            "sex": rng.integers(0, 2, size=n, dtype=np.int8),
            "generation": rng.integers(0, 4, size=n, dtype=np.int32),
            "A1": A1.astype(np.float32),
            "A2": A2.astype(np.float32),
        }
    )


class TestExportPgs:
    def test_shape_and_columns(self, tmp_path):
        ped = _synthetic_A_frame(500, var_A=(1.0, 1.0), rA=0.0, seed=1)
        out, meta = export_pgs(
            ped,
            r2=(0.2, 0.2),
            rA=0.0,
            var_A=(1.0, 1.0),
            sub_seed=123,
            out_path=tmp_path / "pgs.parquet",
        )
        assert out.exists()
        assert meta.exists()
        df = pd.read_parquet(out)
        assert list(df.columns) == ["id", "sex", "generation", "A1", "A2", "PGS1", "PGS2"]
        assert len(df) == len(ped)
        np.testing.assert_array_equal(df["id"].to_numpy(), ped["id"].to_numpy())

    def test_variance_matches_nominal(self, tmp_path):
        # With large n, realized Var(PGS_t) should be close to nominal Var(A_t).
        var_A = (1.0, 2.5)
        ped = _synthetic_A_frame(20_000, var_A=var_A, rA=0.3, seed=2)
        out, _ = export_pgs(
            ped,
            r2=(0.25, 0.5),
            rA=0.3,
            var_A=var_A,
            sub_seed=42,
            out_path=tmp_path / "pgs.parquet",
        )
        df = pd.read_parquet(out)
        # Var(PGS_t) = r²·Var(A_t) + (1-r²)·Var(A_t) = Var(A_t) analytically;
        # 3% slack covers finite-sample fluctuation at n=20k.
        assert abs(df["PGS1"].var(ddof=0) - var_A[0]) < 0.03 * var_A[0]
        assert abs(df["PGS2"].var(ddof=0) - var_A[1]) < 0.03 * var_A[1]

    def test_realized_accuracy_matches_r2(self, tmp_path):
        # Realized cor(PGS_t, A_t)² should be close to the nominal r²_t.
        var_A = (1.0, 1.0)
        r2 = (0.25, 0.5)
        ped = _synthetic_A_frame(20_000, var_A=var_A, rA=0.2, seed=3)
        out, _ = export_pgs(
            ped,
            r2=r2,
            rA=0.2,
            var_A=var_A,
            sub_seed=7,
            out_path=tmp_path / "pgs.parquet",
        )
        df = pd.read_parquet(out)
        r1 = np.corrcoef(df["PGS1"], df["A1"])[0, 1] ** 2
        r2_realized = np.corrcoef(df["PGS2"], df["A2"])[0, 1] ** 2
        assert abs(r1 - r2[0]) < 0.02
        assert abs(r2_realized - r2[1]) < 0.02

    def test_cross_trait_correlation_formula(self, tmp_path):
        # Expected Cor(PGS_1, PGS_2) = rA·[sqrt(r1·r2) + sqrt((1-r1)(1-r2))].
        var_A = (1.0, 1.0)
        r2 = (0.2, 0.8)
        rA = 0.4
        ped = _synthetic_A_frame(30_000, var_A=var_A, rA=rA, seed=4)
        out, _ = export_pgs(
            ped,
            r2=r2,
            rA=rA,
            var_A=var_A,
            sub_seed=11,
            out_path=tmp_path / "pgs.parquet",
        )
        df = pd.read_parquet(out)
        realized = np.corrcoef(df["PGS1"], df["PGS2"])[0, 1]
        expected = rA * (np.sqrt(r2[0] * r2[1]) + np.sqrt((1 - r2[0]) * (1 - r2[1])))
        assert abs(realized - expected) < 0.02

    def test_determinism(self, tmp_path):
        ped = _synthetic_A_frame(1_000, var_A=(1.0, 1.0), rA=0.3, seed=5)
        out_a, _ = export_pgs(
            ped,
            r2=(0.3, 0.4),
            rA=0.3,
            var_A=(1.0, 1.0),
            sub_seed=99,
            out_path=tmp_path / "a.parquet",
        )
        out_b, _ = export_pgs(
            ped,
            r2=(0.3, 0.4),
            rA=0.3,
            var_A=(1.0, 1.0),
            sub_seed=99,
            out_path=tmp_path / "b.parquet",
        )
        df_a = pd.read_parquet(out_a)
        df_b = pd.read_parquet(out_b)
        np.testing.assert_array_equal(df_a["PGS1"].to_numpy(), df_b["PGS1"].to_numpy())
        np.testing.assert_array_equal(df_a["PGS2"].to_numpy(), df_b["PGS2"].to_numpy())

        # Different seed → different draws.
        out_c, _ = export_pgs(
            ped,
            r2=(0.3, 0.4),
            rA=0.3,
            var_A=(1.0, 1.0),
            sub_seed=100,
            out_path=tmp_path / "c.parquet",
        )
        df_c = pd.read_parquet(out_c)
        assert not np.array_equal(df_a["PGS1"].to_numpy(), df_c["PGS1"].to_numpy())

    def test_r2_unity_returns_A(self, tmp_path):
        # r² = 1 means PGS = A exactly (noise term vanishes).
        ped = _synthetic_A_frame(500, var_A=(1.0, 2.0), rA=0.1, seed=6)
        out, _ = export_pgs(
            ped,
            r2=(1.0, 1.0),
            rA=0.1,
            var_A=(1.0, 2.0),
            sub_seed=1,
            out_path=tmp_path / "pgs.parquet",
        )
        df = pd.read_parquet(out)
        # float32 round-trip tolerance.
        np.testing.assert_allclose(df["PGS1"].to_numpy(), df["A1"].to_numpy(), atol=1e-6)
        np.testing.assert_allclose(df["PGS2"].to_numpy(), df["A2"].to_numpy(), atol=1e-6)

    def test_r2_zero_is_pure_noise(self, tmp_path):
        # r² = 0 means PGS is independent of A (pure rescaled noise).
        ped = _synthetic_A_frame(20_000, var_A=(1.0, 1.0), rA=0.3, seed=7)
        out, _ = export_pgs(
            ped,
            r2=(0.0, 0.0),
            rA=0.3,
            var_A=(1.0, 1.0),
            sub_seed=2,
            out_path=tmp_path / "pgs.parquet",
        )
        df = pd.read_parquet(out)
        cor_a = np.corrcoef(df["PGS1"], df["A1"])[0, 1]
        assert abs(cor_a) < 0.03

    def test_metadata_sidecar(self, tmp_path):
        ped = _synthetic_A_frame(2_000, var_A=(1.0, 1.0), rA=0.25, seed=8)
        _, meta_path = export_pgs(
            ped,
            r2=(0.2, 0.3),
            rA=0.25,
            var_A=(1.0, 1.0),
            sub_seed=321,
            out_path=tmp_path / "pgs.parquet",
        )
        with open(meta_path) as fh:
            meta = json.load(fh)
        assert meta["pgs_r2"] == [0.2, 0.3]
        assert meta["rA"] == 0.25
        assert meta["var_A"] == [1.0, 1.0]
        assert meta["sub_seed"] == 321
        assert meta["n"] == 2_000
        assert len(meta["realized_cor_pgs_a"]) == 2
        assert isinstance(meta["realized_cor_pgs1_pgs2"], float)

    def test_rejects_bad_r2(self, tmp_path):
        ped = _synthetic_A_frame(10, var_A=(1.0, 1.0), rA=0.0, seed=9)
        with pytest.raises(ValueError, match="r2"):
            export_pgs(
                ped,
                r2=(1.5, 0.5),
                rA=0.0,
                var_A=(1.0, 1.0),
                sub_seed=1,
                out_path=tmp_path / "pgs.parquet",
            )
        with pytest.raises(ValueError, match="r2"):
            export_pgs(
                ped,
                r2=(-0.1, 0.5),
                rA=0.0,
                var_A=(1.0, 1.0),
                sub_seed=1,
                out_path=tmp_path / "pgs.parquet",
            )
