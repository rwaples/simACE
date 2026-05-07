"""Synthetic msprime fixture exercising the tstrait phenotyping pipeline
(catalog -> assign_effects -> per-chrom GV -> aggregate). Skipped if
tskit/msprime/tstrait are not available — the simACE env doesn't ship them;
the workflow uses the dedicated tskit conda env."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

tskit = pytest.importorskip("tskit")
msprime = pytest.importorskip("msprime")
tstrait = pytest.importorskip("tstrait")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "workflow" / "scripts" / "simace" / "tskit"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def catalog_mod():
    return _load_module("simace_tstrait_catalog", SCRIPT_DIR / "tstrait_site_catalog_chrom.py")


@pytest.fixture(scope="module")
def effects_mod():
    return _load_module("simace_tstrait_effects", SCRIPT_DIR / "tstrait_assign_effects.py")


@pytest.fixture(scope="module")
def gv_chrom_mod():
    return _load_module("simace_tstrait_gv_chrom", SCRIPT_DIR / "tstrait_gv_chrom.py")


@pytest.fixture(scope="module")
def augment_mod():
    return _load_module("simace_tstrait_augment", SCRIPT_DIR / "tstrait_augment_pedigree.py")


def _build_chrom(seqlen: float, seed: int, n_diploid: int = 30) -> tskit.TreeSequence:
    """Tiny single-pop ts with biallelic mutations.

    Mirrors the real pipeline's canonicalize-step filter that drops sites
    with != 1 mutation (recurrent / back / multi-allelic), so build_catalog's
    precondition is satisfied.
    """
    ts = msprime.sim_ancestry(
        samples=n_diploid,
        sequence_length=seqlen,
        recombination_rate=1e-7,
        population_size=1000,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=1e-6, random_seed=seed + 1)
    mut_counts = np.bincount(ts.tables.mutations.site, minlength=ts.num_sites)
    bad_sites = np.where(mut_counts != 1)[0]
    if bad_sites.size:
        ts = ts.delete_sites(bad_sites)
    return ts


@pytest.fixture(scope="module")
def chroms():
    """Three small per-chrom tree sequences."""
    return {n: _build_chrom(seqlen=2e5, seed=100 * n) for n in (1, 2, 3)}


@pytest.fixture(scope="module")
def catalog(catalog_mod, chroms):
    """Concatenated multi-chrom catalog as a single DataFrame."""
    parts = [catalog_mod.build_catalog(ts, n) for n, ts in chroms.items()]
    return pd.concat(parts, ignore_index=True).sort_values(["CHR", "site_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


def test_catalog_schema_and_af_range(catalog):
    assert list(catalog.columns) == ["CHR", "site_id", "POS", "AC", "AF", "causal_allele"]
    assert len(catalog) > 0
    # private singletons can give AF=1/n; doubletons still well below 1
    assert catalog["AF"].between(0.0, 1.0, inclusive="neither").all()
    assert (catalog["AC"] > 0).all()
    # site_id is per-chrom local
    for _chrom_n, sub in catalog.groupby("CHR"):
        assert sub["site_id"].min() == 0
        assert sub["site_id"].max() == len(sub) - 1


def test_catalog_ac_matches_variant_genotypes(catalog_mod, chroms):
    """AC computed via tree.num_samples == AC computed by counting derived alleles."""
    ts = chroms[1]
    df = catalog_mod.build_catalog(ts, 1)
    for var in ts.variants():
        derived_count = int((var.genotypes == 1).sum())
        assert int(df.loc[df["site_id"] == var.site.id, "AC"].iloc[0]) == derived_count


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------


def test_assign_effects_basic_shape(effects_mod, catalog):
    out, meta = effects_mod.assign_effects(
        catalog,
        num_causal=20,
        frac_causal=None,
        maf_threshold=0.0,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=1,
    )
    assert len(out) == 20
    assert list(out.columns) == ["site_id", "POS", "effect_size", "causal_allele", "trait_id", "CHR", "AF"]
    assert (out["trait_id"] == 0).all()
    # sorted by (CHR, site_id)
    sorted_check = out[["CHR", "site_id"]].apply(tuple, axis=1).tolist()
    assert sorted_check == sorted(sorted_check)
    assert meta["n_causal"] == 20
    assert meta["n_eligible_after_maf"] == len(catalog)


def test_assign_effects_frac_causal(effects_mod, catalog):
    out, meta = effects_mod.assign_effects(
        catalog,
        num_causal=None,
        frac_causal=0.1,
        maf_threshold=0.0,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=2,
    )
    assert len(out) == round(0.1 * len(catalog))
    assert meta["n_eligible_after_maf"] == len(catalog)


def test_assign_effects_maf_filter(effects_mod, catalog):
    """maf_threshold drops both rare and very common variants."""
    out, meta = effects_mod.assign_effects(
        catalog,
        num_causal=10,
        frac_causal=None,
        maf_threshold=0.1,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=3,
    )
    chosen_af = out["AF"].to_numpy()
    chosen_maf = np.minimum(chosen_af, 1.0 - chosen_af)
    assert (chosen_maf > 0.1).all()
    assert meta["n_eligible_after_maf"] < len(catalog)


def test_assign_effects_num_and_frac_both_set_raises(effects_mod, catalog):
    with pytest.raises(ValueError, match="both are set"):
        effects_mod.assign_effects(
            catalog,
            num_causal=10,
            frac_causal=0.1,
            maf_threshold=0.0,
            alpha=0.0,
            effect_mean=0.0,
            effect_var=1.0,
            trait_id=0,
            seed=4,
        )


def test_assign_effects_neither_set_raises(effects_mod, catalog):
    with pytest.raises(ValueError, match="both are null"):
        effects_mod.assign_effects(
            catalog,
            num_causal=None,
            frac_causal=None,
            maf_threshold=0.0,
            alpha=0.0,
            effect_mean=0.0,
            effect_var=1.0,
            trait_id=0,
            seed=5,
        )


def test_assign_effects_alpha_scales_betas(effects_mod, catalog):
    """alpha=-0.5 scales raw beta by 1/sqrt(2p(1-p)). Re-derive raw and verify."""
    out, _ = effects_mod.assign_effects(
        catalog,
        num_causal=30,
        frac_causal=None,
        maf_threshold=0.05,
        alpha=-0.5,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=6,
    )
    p = out["AF"].to_numpy()
    af_factor = (2.0 * p * (1.0 - p)) ** -0.5
    raw_implied = out["effect_size"].to_numpy() / af_factor
    # raw betas should look standard normal-ish (small N, just sanity)
    assert abs(np.mean(raw_implied)) < 1.0
    assert 0.2 < np.var(raw_implied) < 5.0


def test_assign_effects_n_causal_too_large_raises(effects_mod, catalog):
    with pytest.raises(ValueError, match="exceeds n_eligible"):
        effects_mod.assign_effects(
            catalog,
            num_causal=len(catalog) + 1,
            frac_causal=None,
            maf_threshold=0.0,
            alpha=0.0,
            effect_mean=0.0,
            effect_var=1.0,
            trait_id=0,
            seed=7,
        )


# --------------------------------------------------------------------------
# Per-chrom GV sum equals one-shot GV (the linchpin of the per-chrom design)
# --------------------------------------------------------------------------


def _slice_effects_to_chrom(effects: pd.DataFrame, chrom_n: int) -> pd.DataFrame:
    """Slice effects to a chrom for tstrait — site_id is canonical here so the
    test's per-chrom tree sequences (which use the same canonical site_ids)
    can be passed straight through. The pipeline does a position remap in
    `tstrait_gv_chrom` before tstrait sees them."""
    return (
        effects[effects["CHR"] == chrom_n][["site_id", "effect_size", "causal_allele", "trait_id"]]
        .sort_values("site_id")
        .reset_index(drop=True)
    )


def _build_concat_ts(chroms: dict) -> tuple[tskit.TreeSequence, pd.DataFrame]:
    """Concatenate the per-chrom tree sequences into one ts with a global
    site_id mapping back to (CHR, local_site_id)."""
    ordered = sorted(chroms.items())
    ts0 = ordered[0][1]
    tables = ts0.dump_tables()
    cum = float(ts0.sequence_length)

    site_map_records: list[dict] = [
        {"CHR": ordered[0][0], "local_site_id": s.id, "global_site_id": s.id} for s in ts0.sites()
    ]

    n_initial = ts0.num_samples
    for chrom_n, ts in ordered[1:]:
        sn, se, ss, sm = ts.tables.nodes, ts.tables.edges, ts.tables.sites, ts.tables.mutations
        node_offset = tables.nodes.num_rows - n_initial
        site_id_offset = tables.sites.num_rows
        mut_id_offset = tables.mutations.num_rows

        if ts.num_nodes > n_initial:
            nm_off = sn.metadata_offset
            tables.nodes.append_columns(
                flags=sn.flags[n_initial:],
                time=sn.time[n_initial:],
                population=sn.population[n_initial:],
                individual=sn.individual[n_initial:],
                metadata=sn.metadata[nm_off[n_initial] :],
                metadata_offset=nm_off[n_initial:] - nm_off[n_initial],
            )

        new_parent = np.where(se.parent < n_initial, se.parent, se.parent + node_offset)
        new_child = np.where(se.child < n_initial, se.child, se.child + node_offset)
        tables.edges.append_columns(
            left=se.left + cum,
            right=se.right + cum,
            parent=new_parent,
            child=new_child,
            metadata=se.metadata,
            metadata_offset=se.metadata_offset,
        )
        tables.sites.append_columns(
            position=ss.position + cum,
            ancestral_state=ss.ancestral_state,
            ancestral_state_offset=ss.ancestral_state_offset,
            metadata=ss.metadata,
            metadata_offset=ss.metadata_offset,
        )
        new_mut_node = np.where(sm.node < n_initial, sm.node, sm.node + node_offset)
        new_mut_parent = np.where(sm.parent == tskit.NULL, tskit.NULL, sm.parent + mut_id_offset)
        tables.mutations.append_columns(
            site=sm.site + site_id_offset,
            node=new_mut_node,
            derived_state=sm.derived_state,
            derived_state_offset=sm.derived_state_offset,
            parent=new_mut_parent,
            metadata=sm.metadata,
            metadata_offset=sm.metadata_offset,
            time=sm.time,
        )
        site_map_records.extend(
            {"CHR": chrom_n, "local_site_id": s.id, "global_site_id": s.id + site_id_offset} for s in ts.sites()
        )
        cum += float(ts.sequence_length)

    tables.sequence_length = cum
    tables.sort()
    return tables.tree_sequence(), pd.DataFrame(site_map_records)


def test_per_chrom_gv_sum_equals_oneshot(effects_mod, catalog, chroms):
    """Sum of per-chrom GVs == one-shot GV computed on the concatenated ts.

    This is the linearity property that makes the per-chrom decomposition
    safe. We allow tiny float drift because pandas groupby-sum and tstrait's
    internal accumulation may differ in summation order.
    """
    effects, _ = effects_mod.assign_effects(
        catalog,
        num_causal=15,
        frac_causal=None,
        maf_threshold=0.05,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=99,
    )

    # per-chrom GV
    per_chrom_gvs: list[pd.DataFrame] = []
    for chrom_n, ts in chroms.items():
        sub = _slice_effects_to_chrom(effects, chrom_n)
        if len(sub) == 0:
            continue
        per_chrom_gvs.append(tstrait.genetic_value(ts, sub))
    summed = (
        pd.concat(per_chrom_gvs, ignore_index=True)
        .groupby(["trait_id", "individual_id"], as_index=False)["genetic_value"]
        .sum()
        .sort_values("individual_id")
        .reset_index(drop=True)
    )

    # one-shot GV on concat
    ts_all, site_map = _build_concat_ts(chroms)
    effects_global = effects.merge(
        site_map.rename(columns={"local_site_id": "site_id"}),
        on=["CHR", "site_id"],
        how="left",
    )
    assert effects_global["global_site_id"].notna().all()
    one_shot_input = (
        effects_global[["global_site_id", "effect_size", "causal_allele", "trait_id"]]
        .rename(columns={"global_site_id": "site_id"})
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    one_shot = tstrait.genetic_value(ts_all, one_shot_input).sort_values("individual_id").reset_index(drop=True)

    np.testing.assert_allclose(
        summed["genetic_value"].to_numpy(),
        one_shot["genetic_value"].to_numpy(),
        atol=1e-9,
        rtol=1e-9,
    )


def test_empty_chrom_zero_gv_path(effects_mod, catalog, chroms):
    """Force a chrom with no causal sites by restricting effects to chrom 1 only."""
    effects, _ = effects_mod.assign_effects(
        catalog[catalog["CHR"] == 1].reset_index(drop=True),
        num_causal=5,
        frac_causal=None,
        maf_threshold=0.0,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=42,
    )
    # chr2 should now contribute no causal sites — emulate the gv_chrom guard.
    chrom2 = chroms[2]
    sub = _slice_effects_to_chrom(effects, 2)
    assert len(sub) == 0
    # Replicate the gv_chrom empty-path
    sample_inds = chrom2.tables.nodes.individual[chrom2.samples()]
    sample_inds = sample_inds[sample_inds != tskit.NULL]
    inds = np.unique(sample_inds)
    gv = pd.DataFrame(
        {
            "trait_id": np.full(len(inds), 0, dtype=np.int64),
            "individual_id": inds.astype(np.int64),
            "genetic_value": np.zeros(len(inds), dtype=np.float64),
        }
    )
    assert (gv["genetic_value"] == 0).all()
    assert len(gv) == chrom2.num_individuals


# --------------------------------------------------------------------------
# Heritability target via sim_env
# --------------------------------------------------------------------------


def test_sim_env_realized_h2_within_tolerance(effects_mod, catalog, chroms):
    """tstrait.sim_env should produce phenotype ~ Var(GV)/(Var(GV)+Var(E)) ≈ h2."""
    effects, _ = effects_mod.assign_effects(
        catalog,
        num_causal=40,
        frac_causal=None,
        maf_threshold=0.05,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=11,
    )
    per_chrom: list[pd.DataFrame] = []
    for chrom_n, ts in chroms.items():
        sub = _slice_effects_to_chrom(effects, chrom_n)
        if len(sub) == 0:
            continue
        per_chrom.append(tstrait.genetic_value(ts, sub))
    summed = (
        pd.concat(per_chrom, ignore_index=True)
        .groupby(["trait_id", "individual_id"], as_index=False)["genetic_value"]
        .sum()
    )

    target_h2 = 0.4
    pheno = tstrait.sim_env(summed, h2=target_h2, random_seed=12)
    var_gv = float(pheno["genetic_value"].var(ddof=0))
    var_env = float(pheno["environmental_noise"].var(ddof=0))
    realized = var_gv / (var_gv + var_env)
    # Loose tolerance — small N (30 inds) makes sampling noise large.
    assert abs(realized - target_h2) < 0.15


# --------------------------------------------------------------------------
# Smoke: assign_effects determinism
# --------------------------------------------------------------------------


def test_assign_effects_seed_determinism(effects_mod, catalog):
    a, _ = effects_mod.assign_effects(
        catalog,
        num_causal=20,
        frac_causal=None,
        maf_threshold=0.0,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=777,
    )
    b, _ = effects_mod.assign_effects(
        catalog,
        num_causal=20,
        frac_causal=None,
        maf_threshold=0.0,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=777,
    )
    pd.testing.assert_frame_equal(a, b)


def test_remap_effects_by_position_drops_missing(gv_chrom_mod, chroms, effects_mod, catalog):
    """Causal sites whose POS is not in the grafted ts get dropped, others remap."""
    effects, _ = effects_mod.assign_effects(
        catalog,
        num_causal=20,
        frac_causal=None,
        maf_threshold=0.0,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=123,
    )
    ts = chroms[1]
    chrom1 = effects[effects["CHR"] == 1].copy()
    # Drop a few sites from the ts to simulate the drop-and-graft loss
    # (delete sites at half of the chosen positions).
    chosen_pos = chrom1["POS"].to_numpy().astype(np.int64)
    ts_pos = np.asarray(ts.tables.sites.position).astype(np.int64)
    drop_pos = chosen_pos[: len(chosen_pos) // 2]
    drop_site_ids = np.where(np.isin(ts_pos, drop_pos))[0]
    ts_lossy = ts.delete_sites(drop_site_ids)

    remapped, n_dropped = gv_chrom_mod.remap_effects_by_position(chrom1, ts_lossy)
    # Confirm survivors point to valid grafted-ts site_ids
    assert n_dropped == len(drop_pos)
    new_pos = np.asarray(ts_lossy.tables.sites.position).astype(np.int64)[remapped["site_id"].to_numpy()]
    assert np.array_equal(np.sort(new_pos), np.sort(remapped["POS"].to_numpy().astype(np.int64)))
    # site_ids are sorted ascending after the remap
    assert remapped["site_id"].is_monotonic_increasing


def test_remap_effects_empty_input(gv_chrom_mod, chroms):
    """Empty input returns empty + 0 dropped without erroring."""
    empty = pd.DataFrame(
        {
            "site_id": pd.Series([], dtype=np.int64),
            "POS": pd.Series([], dtype=np.int64),
            "effect_size": pd.Series([], dtype=np.float64),
            "causal_allele": pd.Series([], dtype=object),
            "trait_id": pd.Series([], dtype=np.int32),
            "CHR": pd.Series([], dtype=np.int8),
            "AF": pd.Series([], dtype=np.float64),
        }
    )
    out, n = gv_chrom_mod.remap_effects_by_position(empty, chroms[1])
    assert len(out) == 0
    assert n == 0


def test_compute_gv_matches_tstrait(gv_chrom_mod, effects_mod, catalog, chroms):
    """The numba kernel produces byte-equivalent (up to float order) per-individual
    GV vs tstrait.genetic_value on the same effects + ts."""
    effects, _ = effects_mod.assign_effects(
        catalog,
        num_causal=20,
        frac_causal=None,
        maf_threshold=0.0,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=314,
    )
    ts = chroms[1]
    chrom_effects = effects[effects["CHR"] == 1].copy()
    chrom_effects, _ = gv_chrom_mod.remap_effects_by_position(chrom_effects, ts)
    if len(chrom_effects) == 0:
        pytest.skip("no causal sites on chr1 for this seed")

    # Numba kernel output (sample-filtered)
    gv_kernel, _ = gv_chrom_mod.compute_gv(ts, chrom_effects, trait_id=0)

    # tstrait baseline (sample-filter to same set)
    trait_df = chrom_effects[["site_id", "effect_size", "causal_allele", "trait_id"]]
    gv_tstrait_full = tstrait.genetic_value(ts, trait_df)
    sample_inds = gv_chrom_mod._individuals_from_samples(ts)
    gv_tstrait = (
        gv_tstrait_full[gv_tstrait_full["individual_id"].isin(sample_inds)]
        .sort_values("individual_id")
        .reset_index(drop=True)
    )
    gv_kernel_sorted = gv_kernel.sort_values("individual_id").reset_index(drop=True)

    np.testing.assert_array_equal(
        gv_kernel_sorted["individual_id"].to_numpy(),
        gv_tstrait["individual_id"].to_numpy(),
    )
    np.testing.assert_allclose(
        gv_kernel_sorted["genetic_value"].to_numpy(),
        gv_tstrait["genetic_value"].to_numpy(),
        atol=1e-12,
        rtol=1e-12,
    )


def test_remap_effects_all_present_passes_through(gv_chrom_mod, chroms, effects_mod, catalog):
    """If every causal POS is in the ts, no sites are dropped."""
    effects, _ = effects_mod.assign_effects(
        catalog,
        num_causal=10,
        frac_causal=None,
        maf_threshold=0.0,
        alpha=0.0,
        effect_mean=0.0,
        effect_var=1.0,
        trait_id=0,
        seed=456,
    )
    ts = chroms[2]
    chrom2 = effects[effects["CHR"] == 2].copy()
    out, n = gv_chrom_mod.remap_effects_by_position(chrom2, ts)
    assert n == 0
    assert len(out) == len(chrom2)


def test_rescale_gv_exact_variance_match(augment_mod):
    """Centered + rescaled output has mean 0 and ddof=0 var == target_var exactly."""
    rng = np.random.default_rng(0)
    gv = rng.normal(loc=5.0, scale=2.0, size=1000)  # var ≈ 4
    out, info = augment_mod.rescale_gv(gv, target_var=0.7)
    assert abs(out.mean()) < 1e-12
    assert abs(out.var(ddof=0) - 0.7) < 1e-12
    assert info["scale_factor"] > 0
    assert abs(info["realized_mean"] - 5.0) < 0.5
    assert abs(info["realized_var"] - 4.0) < 0.5


def test_rescale_gv_target_zero(augment_mod):
    """target_var=0 produces an all-zero array."""
    rng = np.random.default_rng(1)
    gv = rng.normal(size=100)
    out, info = augment_mod.rescale_gv(gv, target_var=0.0)
    assert np.allclose(out, 0.0)
    assert info["scale_factor"] == 0.0


def test_rescale_gv_zero_input_variance_raises(augment_mod):
    """Constant GV input has 0 variance → cannot rescale."""
    with pytest.raises(ValueError, match="variance is 0"):
        augment_mod.rescale_gv(np.ones(10), target_var=0.5)


def test_rescale_gv_negative_target_raises(augment_mod):
    with pytest.raises(ValueError, match="target_var must be >= 0"):
        augment_mod.rescale_gv(np.array([0.0, 1.0, 2.0]), target_var=-0.1)


def test_meta_json_roundtrip(effects_mod, catalog, tmp_path):
    """Smoke check: meta dict from assign_effects is JSON-serialisable."""
    _, meta = effects_mod.assign_effects(
        catalog,
        num_causal=5,
        frac_causal=None,
        maf_threshold=0.05,
        alpha=-0.5,
        effect_mean=0.0,
        effect_var=2.0,
        trait_id=0,
        seed=1,
    )
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(meta, default=float))
    loaded = json.loads(p.read_text())
    assert loaded["n_causal"] == 5
    assert loaded["alpha"] == -0.5
