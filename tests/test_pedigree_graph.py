"""Tests for sim_ace.pedigree_graph relationship extraction."""

import logging

import numpy as np
import pandas as pd
import pytest

from sim_ace.pedigree_graph import count_sib_pairs, extract_relationship_pairs
from sim_ace.validate import _count_sib_pairs_legacy

logger = logging.getLogger(__name__)


def _extract_relationship_pairs_legacy(df: pd.DataFrame, seed: int = 42) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Legacy implementation kept for golden testing."""
    ids_arr = df["id"].values.astype(np.int64)
    id_to_row = np.full(ids_arr.max() + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)

    def resolve_rows(ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        mask = (ids >= 0) & (ids < len(id_to_row))
        result = np.full(len(ids), -1, dtype=np.int32)
        result[mask] = id_to_row[ids[mask]]
        return result

    pairs = {}

    twins = df[df["twin"] != -1]
    ta = twins["id"].values.astype(int)
    tb = twins["twin"].values.astype(int)
    mask = ta < tb
    pairs["MZ twin"] = (resolve_rows(ta[mask]), resolve_rows(tb[mask]))

    non_twin_nf = df[(df["mother"] != -1) & (df["twin"] == -1)].copy()
    non_twin_nf["_row"] = non_twin_nf.index.values

    full_rows_1, full_rows_2 = [], []
    mat_half_rows_1, mat_half_rows_2 = [], []

    sib_counts = non_twin_nf.groupby("mother").size()
    multi_mothers = sib_counts[sib_counts >= 2].index
    mat_sib = non_twin_nf[non_twin_nf["mother"].isin(multi_mothers)]

    if len(mat_sib) > 0:
        mat_pairs = mat_sib[["mother", "father", "_row"]].merge(
            mat_sib[["mother", "father", "_row"]],
            on="mother",
            suffixes=("_1", "_2"),
        )
        mat_pairs = mat_pairs[mat_pairs["_row_1"] < mat_pairs["_row_2"]]
        same_father = mat_pairs["father_1"] == mat_pairs["father_2"]
        full_rows_1.append(mat_pairs.loc[same_father, "_row_1"].values)
        full_rows_2.append(mat_pairs.loc[same_father, "_row_2"].values)
        mat_half_rows_1.append(mat_pairs.loc[~same_father, "_row_1"].values)
        mat_half_rows_2.append(mat_pairs.loc[~same_father, "_row_2"].values)

    pat_half_rows_1, pat_half_rows_2 = [], []
    pat_counts = non_twin_nf.groupby("father").size()
    multi_fathers = pat_counts[pat_counts >= 2].index
    pat_sib = non_twin_nf[non_twin_nf["father"].isin(multi_fathers)]

    if len(pat_sib) > 0:
        pat_pairs = pat_sib[["mother", "father", "_row"]].merge(
            pat_sib[["mother", "father", "_row"]],
            on="father",
            suffixes=("_1", "_2"),
        )
        pat_pairs = pat_pairs[pat_pairs["_row_1"] < pat_pairs["_row_2"]]
        diff_mother = pat_pairs["mother_1"] != pat_pairs["mother_2"]
        pat_half_rows_1.append(pat_pairs.loc[diff_mother, "_row_1"].values)
        pat_half_rows_2.append(pat_pairs.loc[diff_mother, "_row_2"].values)

    pairs["Full sib"] = (
        np.concatenate(full_rows_1) if full_rows_1 else np.array([], dtype=int),
        np.concatenate(full_rows_2) if full_rows_1 else np.array([], dtype=int),
    )
    pairs["Maternal half sib"] = (
        np.concatenate(mat_half_rows_1) if mat_half_rows_1 else np.array([], dtype=int),
        np.concatenate(mat_half_rows_2) if mat_half_rows_1 else np.array([], dtype=int),
    )
    pairs["Paternal half sib"] = (
        np.concatenate(pat_half_rows_1) if pat_half_rows_1 else np.array([], dtype=int),
        np.concatenate(pat_half_rows_2) if pat_half_rows_1 else np.array([], dtype=int),
    )

    all_nf = df[df["mother"] != -1]
    child_rows = all_nf.index.values
    mother_rows = resolve_rows(all_nf["mother"].values.astype(int))
    father_rows = resolve_rows(all_nf["father"].values.astype(int))

    m_valid = mother_rows >= 0
    f_valid = father_rows >= 0
    pairs["Mother-offspring"] = (child_rows[m_valid], mother_rows[m_valid])
    pairs["Father-offspring"] = (child_rows[f_valid], father_rows[f_valid])

    child_ids = all_nf["id"].values.astype(np.int64)
    mother_ids = all_nf["mother"].values.astype(np.int64)
    father_ids = all_nf["father"].values.astype(np.int64)
    n_children = len(child_ids)

    df_mothers_col = df["mother"].values.astype(np.int64)
    df_fathers_col = df["father"].values.astype(np.int64)
    mother_row = resolve_rows(mother_ids)
    father_row = resolve_rows(father_ids)

    gp_ids = np.full((n_children, 4), -1, dtype=np.int64)
    m_ok = mother_row >= 0
    gp_ids[m_ok, 0] = df_mothers_col[mother_row[m_ok]]
    gp_ids[m_ok, 1] = df_fathers_col[mother_row[m_ok]]
    f_ok = father_row >= 0
    gp_ids[f_ok, 2] = df_mothers_col[father_row[f_ok]]
    gp_ids[f_ok, 3] = df_fathers_col[father_row[f_ok]]

    gp_child = np.tile(child_ids, 4)
    gp_parent = np.concatenate([mother_ids, mother_ids, father_ids, father_ids])
    gp_gp = np.concatenate([gp_ids[:, 0], gp_ids[:, 1], gp_ids[:, 2], gp_ids[:, 3]])

    valid_gp = gp_gp >= 0
    gp_child = gp_child[valid_gp]
    gp_parent = gp_parent[valid_gp]
    gp_gp = gp_gp[valid_gp]

    unique_gp_arr = np.unique(gp_gp)
    if len(unique_gp_arr) > 100000:
        logger.info(
            "extract_relationship_pairs: %d grandparents exceed 100K cap, sampling subset",
            len(unique_gp_arr),
        )
        rng = np.random.default_rng(seed)
        selected_gp = rng.choice(unique_gp_arr, 100000, replace=False)
        gp_mask = np.isin(gp_gp, selected_gp)
        gp_child = gp_child[gp_mask]
        gp_parent = gp_parent[gp_mask]
        gp_gp = gp_gp[gp_mask]

    sort_idx = np.argsort(gp_gp, kind="mergesort")
    gp_child = gp_child[sort_idx]
    gp_parent = gp_parent[sort_idx]
    gp_gp = gp_gp[sort_idx]

    _, group_starts, group_counts = np.unique(gp_gp, return_index=True, return_counts=True)

    multi = group_counts >= 2
    group_starts = group_starts[multi]
    group_counts = group_counts[multi]

    pair_i_parts = []
    pair_j_parts = []
    for size in np.unique(group_counts):
        gs = group_starts[group_counts == size]
        ii, jj = np.triu_indices(size, k=1)
        all_i = (gs[:, np.newaxis] + ii[np.newaxis, :]).ravel()
        all_j = (gs[:, np.newaxis] + jj[np.newaxis, :]).ravel()
        pair_i_parts.append(all_i)
        pair_j_parts.append(all_j)

    pair_i = np.concatenate(pair_i_parts)
    pair_j = np.concatenate(pair_j_parts)

    diff_parent = gp_parent[pair_i] != gp_parent[pair_j]
    c1_raw = gp_child[pair_i[diff_parent]]
    c2_raw = gp_child[pair_j[diff_parent]]

    c1 = np.minimum(c1_raw, c2_raw)
    c2 = np.maximum(c1_raw, c2_raw)

    max_id = int(c2.max()) + 1
    pair_keys = c1.astype(np.int64) * max_id + c2.astype(np.int64)
    unique_keys = np.unique(pair_keys)
    c1_final = unique_keys // max_id
    c2_final = unique_keys % max_id

    c_idx1 = resolve_rows(c1_final)
    c_idx2 = resolve_rows(c2_final)
    c_valid = (c_idx1 >= 0) & (c_idx2 >= 0)
    pairs["1st cousin"] = (c_idx1[c_valid], c_idx2[c_valid])

    return pairs


def _pairs_to_set(idx1, idx2):
    """Convert pair arrays to a set of sorted tuples for comparison."""
    return {(min(a, b), max(a, b)) for a, b in zip(idx1.tolist(), idx2.tolist(), strict=True)}


class TestGoldenComparison:
    """New implementation must produce identical pair sets as legacy for original 7 categories."""

    def test_golden_pairs_match(self, small_pedigree):
        """Run old and new on the same pedigree; assert identical pairs for original 7 keys.

        For cousins, the legacy version applies a 100K grandparent cap (subsamples),
        so we only check that legacy is a subset of new. For the small fixture
        (N=1000, G=3), the cap shouldn't trigger, so they should be equal.
        """
        df = small_pedigree
        legacy = _extract_relationship_pairs_legacy(df, seed=42)
        new = extract_relationship_pairs(df, seed=42)

        exact_keys = [
            "MZ twin",
            "Full sib",
            "Maternal half sib",
            "Paternal half sib",
            "Mother-offspring",
            "Father-offspring",
        ]
        for key in exact_keys:
            legacy_set = _pairs_to_set(*legacy[key])
            new_set = _pairs_to_set(*new[key])
            assert legacy_set == new_set, (
                f"{key}: legacy has {len(legacy_set)} pairs, new has {len(new_set)} pairs, "
                f"diff: {legacy_set.symmetric_difference(new_set)}"
            )

        # Cousins: legacy may subsample, produce self-pairs, or
        # misclassify siblings as cousins (pairs that share a parent AND a
        # grandparent are siblings, not cousins — legacy doesn't filter these).
        # New implementation should be a superset of legacy minus buggy pairs.
        legacy_cousins = _pairs_to_set(*legacy["1st cousin"])
        new_cousins = _pairs_to_set(*new["1st cousin"])
        mother = df["mother"].values
        father = df["father"].values
        # Filter out self-pairs and sibling-pairs from legacy
        legacy_proper = set()
        for a, b in legacy_cousins:
            if a == b:
                continue
            # Skip pairs that share a parent (they're siblings, not cousins)
            if mother[a] == mother[b] or father[a] == father[b]:
                continue
            legacy_proper.add((a, b))
        assert legacy_proper <= new_cousins, (
            f"1st cousin: legacy has {len(legacy_proper - new_cousins)} pairs not in new: {legacy_proper - new_cousins}"
        )

    def test_golden_sib_counts_match(self, small_pedigree):
        """count_sib_pairs must match _count_sib_pairs_legacy."""
        df = small_pedigree
        non_founders = df[df["mother"] != -1]
        non_twin_sibs = non_founders[non_founders["twin"] == -1][["id", "mother", "father"]]

        legacy = _count_sib_pairs_legacy(non_twin_sibs)
        new = count_sib_pairs(non_twin_sibs)

        for key in legacy:
            assert legacy[key] == new[key], f"{key}: legacy={legacy[key]}, new={new[key]}"


class TestNewRelationships:
    """Test the 3 new relationship categories."""

    def test_has_new_keys(self, small_pedigree):
        pairs = extract_relationship_pairs(small_pedigree)
        assert "Grandparent-grandchild" in pairs
        assert "Avuncular" in pairs
        assert "2nd cousin" in pairs

    def test_grandparent_grandchild_structure(self, small_pedigree):
        """Each grandparent-grandchild pair must be separated by 2 generations."""
        df = small_pedigree
        pairs = extract_relationship_pairs(df)
        gc, gp = pairs["Grandparent-grandchild"]
        if len(gc) == 0:
            pytest.skip("No grandparent-grandchild pairs found")

        gen = df["generation"].values
        gen_diff = np.abs(gen[gc] - gen[gp])
        assert np.all(gen_diff == 2), f"Generation diffs: {np.unique(gen_diff)}"

    def test_grandparent_grandchild_ancestry(self, small_pedigree):
        """Each grandchild must have the grandparent as a parent of a parent."""
        df = small_pedigree
        pairs = extract_relationship_pairs(df)
        gc_arr, gp_arr = pairs["Grandparent-grandchild"]
        mother = df["mother"].values
        father = df["father"].values

        for gc, gp in zip(gc_arr[:100], gp_arr[:100], strict=True):
            m = mother[gc]
            f = father[gc]
            grandparents = set()
            if m >= 0:
                if mother[m] >= 0:
                    grandparents.add(mother[m])
                if father[m] >= 0:
                    grandparents.add(father[m])
            if f >= 0:
                if mother[f] >= 0:
                    grandparents.add(mother[f])
                if father[f] >= 0:
                    grandparents.add(father[f])
            assert gp in grandparents, f"gc={gc}, gp={gp}, actual grandparents={grandparents}"

    def test_avuncular_structure(self, small_pedigree):
        """Avuncular pairs span exactly 1 generation."""
        df = small_pedigree
        pairs = extract_relationship_pairs(df)
        a1, a2 = pairs["Avuncular"]
        if len(a1) == 0:
            pytest.skip("No avuncular pairs found")

        gen = df["generation"].values
        gen_diff = np.abs(gen[a1] - gen[a2])
        # Avuncular: uncle/aunt is same gen as parent, so 1 gen from nephew/niece
        assert np.all(gen_diff <= 1), f"Generation diffs: {np.unique(gen_diff)}"


class TestStructuralCorrectness:
    """Verify structural properties of extracted pairs."""

    def test_full_sibs_share_both_parents(self, small_pedigree):
        df = small_pedigree
        pairs = extract_relationship_pairs(df)
        idx1, idx2 = pairs["Full sib"]
        if len(idx1) == 0:
            pytest.skip("No full sib pairs")

        mother = df["mother"].values
        father = df["father"].values
        assert np.all(mother[idx1] == mother[idx2])
        assert np.all(father[idx1] == father[idx2])

    def test_half_sibs_share_exactly_one_parent(self, small_pedigree):
        df = small_pedigree
        pairs = extract_relationship_pairs(df)

        mother = df["mother"].values
        father = df["father"].values

        for key in ["Maternal half sib", "Paternal half sib"]:
            idx1, idx2 = pairs[key]
            if len(idx1) == 0:
                continue

            same_m = mother[idx1] == mother[idx2]
            same_f = father[idx1] == father[idx2]

            if key == "Maternal half sib":
                assert np.all(same_m), "Maternal half sibs must share mother"
                assert np.all(~same_f), "Maternal half sibs must NOT share father"
            else:
                assert np.all(same_f), "Paternal half sibs must share father"
                assert np.all(~same_m), "Paternal half sibs must NOT share mother"

    def test_cousins_share_grandparent(self, small_pedigree):
        """Every 1st cousin pair must share at least one grandparent and not be a self-pair."""
        df = small_pedigree
        pairs = extract_relationship_pairs(df)
        idx1, idx2 = pairs["1st cousin"]
        if len(idx1) == 0:
            pytest.skip("No cousin pairs")

        mother = df["mother"].values
        father = df["father"].values

        for i in range(min(200, len(idx1))):
            a, b = idx1[i], idx2[i]
            assert a != b, f"Self-pair found: ({a}, {b})"

            # Share a grandparent
            gp_a = set()
            for p in [mother[a], father[a]]:
                if p >= 0:
                    if mother[p] >= 0:
                        gp_a.add(mother[p])
                    if father[p] >= 0:
                        gp_a.add(father[p])
            gp_b = set()
            for p in [mother[b], father[b]]:
                if p >= 0:
                    if mother[p] >= 0:
                        gp_b.add(mother[p])
                    if father[p] >= 0:
                        gp_b.add(father[p])
            assert gp_a & gp_b, f"Cousins {a},{b} share no grandparent"

    def test_no_pair_overlap_within_sibling_types(self, small_pedigree):
        """Full sib, maternal half sib, paternal half sib should be mutually exclusive."""
        pairs = extract_relationship_pairs(small_pedigree)
        sib_keys = ["Full sib", "Maternal half sib", "Paternal half sib"]
        sib_sets = {k: _pairs_to_set(*pairs[k]) for k in sib_keys}

        for i in range(len(sib_keys)):
            for j in range(i + 1, len(sib_keys)):
                overlap = sib_sets[sib_keys[i]] & sib_sets[sib_keys[j]]
                assert len(overlap) == 0, f"Overlap between '{sib_keys[i]}' and '{sib_keys[j]}': {len(overlap)} pairs"

    def test_no_pair_overlap_twins_and_siblings(self, small_pedigree):
        """MZ twins should not also appear as siblings."""
        pairs = extract_relationship_pairs(small_pedigree)
        twin_set = _pairs_to_set(*pairs["MZ twin"])
        for key in ["Full sib", "Maternal half sib", "Paternal half sib"]:
            sib_set = _pairs_to_set(*pairs[key])
            overlap = twin_set & sib_set
            assert len(overlap) == 0, f"Overlap between 'MZ twin' and '{key}': {len(overlap)} pairs"


class TestNoSubsamplingLoss:
    """The new implementation should not subsample cousins."""

    def test_exact_cousin_count(self, small_pedigree):
        """Verify cousin count is deterministic (no RNG-dependent cap)."""
        pairs1 = extract_relationship_pairs(small_pedigree, seed=1)
        pairs2 = extract_relationship_pairs(small_pedigree, seed=999)
        assert len(pairs1["1st cousin"][0]) == len(pairs2["1st cousin"][0])


class TestEdgeCases:
    def test_founders_only(self):
        """DataFrame with only founders should produce no pairs except possibly twins."""
        df = pd.DataFrame(
            {
                "id": np.arange(100),
                "mother": np.full(100, -1),
                "father": np.full(100, -1),
                "twin": np.full(100, -1),
                "sex": np.random.default_rng(42).binomial(1, 0.5, 100),
                "generation": np.zeros(100, dtype=int),
            }
        )
        pairs = extract_relationship_pairs(df)
        for key in [
            "Full sib",
            "Maternal half sib",
            "Paternal half sib",
            "Mother-offspring",
            "Father-offspring",
            "1st cousin",
            "Grandparent-grandchild",
            "Avuncular",
            "2nd cousin",
        ]:
            assert len(pairs[key][0]) == 0, f"{key} should be empty for founders-only"

    def test_single_child_families(self):
        """No sibling pairs when every family has exactly 1 child."""
        n_founders = 50
        n_children = 25
        ids = np.arange(n_founders + n_children)
        mothers = np.full(n_founders + n_children, -1, dtype=int)
        fathers = np.full(n_founders + n_children, -1, dtype=int)
        twins = np.full(n_founders + n_children, -1, dtype=int)
        sex = np.zeros(n_founders + n_children, dtype=int)
        gen = np.zeros(n_founders + n_children, dtype=int)

        # Assign unique parents to each child
        females = [i for i in range(n_founders) if i % 2 == 0]
        males = [i for i in range(n_founders) if i % 2 == 1]
        sex[:n_founders:2] = 0  # females
        sex[1:n_founders:2] = 1  # males

        for i in range(n_children):
            child_id = n_founders + i
            mothers[child_id] = females[i]
            fathers[child_id] = males[i]
            gen[child_id] = 1

        df = pd.DataFrame(
            {
                "id": ids,
                "mother": mothers,
                "father": fathers,
                "twin": twins,
                "sex": sex,
                "generation": gen,
            }
        )
        pairs = extract_relationship_pairs(df)
        assert len(pairs["Full sib"][0]) == 0
        assert len(pairs["Maternal half sib"][0]) == 0
        assert len(pairs["Paternal half sib"][0]) == 0


class TestKnownTinyPedigree:
    """Hand-built 3-generation pedigree with manually counted expected pairs."""

    @pytest.fixture
    def tiny_pedigree(self):
        """3-generation pedigree:
        Gen 0: 4 founders (0=F, 1=M, 2=F, 3=M)
        Gen 1: 4 offspring
          - 4,5 children of (0,1) — full sibs
          - 6   child of (2,3)
          - 7   child of (2,3) — full sib with 6
        Gen 2: 3 offspring
          - 8   child of (4=F, 6=M) — 4 is female, 6 is male
          - 9   child of (5=F, 7=M) — 5 is female, 7 is male
          - 10  child of (4=F, 6=M) — full sib with 8

        Expected:
          Full sibs: (4,5), (6,7), (8,10) = 3 pairs
          1st cousins: (8,9), (9,10) = 2 pairs (parents 4&5 are full sibs, parents 6&7 are full sibs)
          Mother-offspring: (4,0), (5,0), (6,2), (7,2), (8,4), (9,5), (10,4) = 7
          Father-offspring: (4,1), (5,1), (6,3), (7,3), (8,6), (9,7), (10,6) = 7
          Grandparent-grandchild: 8→{0,1,2,3}, 9→{0,1,2,3}, 10→{0,1,2,3} = 12
          Avuncular: 5 is aunt of 8,10; 4 is aunt of 9; 7 is uncle of 8,10; 6 is uncle of 9
                     = 6 pairs
        """
        n = 11
        data = {
            "id": np.arange(n),
            "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2, 4, 5, 4]),
            "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3, 6, 7, 6]),
            "twin": np.full(n, -1),
            "sex": np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]),
            "generation": np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
        }
        return pd.DataFrame(data)

    def test_full_sib_count(self, tiny_pedigree):
        pairs = extract_relationship_pairs(tiny_pedigree)
        sib_set = _pairs_to_set(*pairs["Full sib"])
        expected = {(4, 5), (6, 7), (8, 10)}
        assert sib_set == expected, f"Got {sib_set}"

    def test_mother_offspring_count(self, tiny_pedigree):
        pairs = extract_relationship_pairs(tiny_pedigree)
        mo_set = _pairs_to_set(*pairs["Mother-offspring"])
        assert len(mo_set) == 7

    def test_father_offspring_count(self, tiny_pedigree):
        pairs = extract_relationship_pairs(tiny_pedigree)
        fo_set = _pairs_to_set(*pairs["Father-offspring"])
        assert len(fo_set) == 7

    def test_cousin_count(self, tiny_pedigree):
        pairs = extract_relationship_pairs(tiny_pedigree)
        cousin_set = _pairs_to_set(*pairs["1st cousin"])
        # 8's parents: (4, 6). 9's parents: (5, 7).
        # 4 & 5 share grandparents 0,1. 6 & 7 share grandparents 2,3.
        # So 8 and 9 are double 1st cousins (share all 4 grandparents).
        # 10's parents: (4, 6) same as 8, so 10 is full sib of 8.
        # 10 and 9: parents (4,6) vs (5,7) — same as 8 vs 9
        expected = {(8, 9), (9, 10)}
        assert cousin_set == expected, f"Got {cousin_set}"

    def test_grandparent_grandchild_count(self, tiny_pedigree):
        pairs = extract_relationship_pairs(tiny_pedigree)
        gp_set = _pairs_to_set(*pairs["Grandparent-grandchild"])
        # 8 → grandparents 0,1,2,3
        # 9 → grandparents 0,1,2,3
        # 10 → grandparents 0,1,2,3
        expected = {
            (0, 8),
            (1, 8),
            (2, 8),
            (3, 8),
            (0, 9),
            (1, 9),
            (2, 9),
            (3, 9),
            (0, 10),
            (1, 10),
            (2, 10),
            (3, 10),
        }
        assert gp_set == expected, f"Got {gp_set}, expected {expected}"

    def test_avuncular_count(self, tiny_pedigree):
        pairs = extract_relationship_pairs(tiny_pedigree)
        avunc_set = _pairs_to_set(*pairs["Avuncular"])
        # 5 is full sib of 4 (mother of 8, 10) → 5 is aunt of 8, 10
        # 4 is full sib of 5 (mother of 9) → 4 is aunt of 9
        # 7 is full sib of 6 (father of 8, 10) → 7 is uncle of 8, 10
        # 6 is full sib of 7 (father of 9) → 6 is uncle of 9
        expected = {(5, 8), (5, 10), (4, 9), (7, 8), (7, 10), (6, 9)}
        assert avunc_set == expected, f"Got {avunc_set}"
