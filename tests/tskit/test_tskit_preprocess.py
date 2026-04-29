"""Synthetic msprime fixture exercising the inlined canonicalize + concat
logic from `workflow/scripts/simace/tskit/`. Skipped if tskit/msprime are
not available (the simACE env doesn't ship them; the workflow uses a
dedicated conda env)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

tskit = pytest.importorskip("tskit")
msprime = pytest.importorskip("msprime")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "workflow" / "scripts" / "simace" / "tskit"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def canonicalize_mod():
    return _load_module("simace_tskit_canonicalize", SCRIPT_DIR / "canonicalize_chrom.py")


@pytest.fixture(scope="module")
def concat_mod():
    return _load_module("simace_tskit_concat", SCRIPT_DIR / "concat_chroms.py")


def _build_chrom(seqlen: float, seed: int, pop_name: str = "p2") -> tskit.TreeSequence:
    """Build a tiny 2-pop tree sequence with mutations. Each population
    contributes diploid samples; we tag the metadata `name` so the
    population filter can find it."""
    demography = msprime.Demography()
    demography.add_population(name="p1", initial_size=500)
    demography.add_population(name=pop_name, initial_size=500)
    demography.add_population(name="ancestral", initial_size=1000)
    demography.add_population_split(time=200, derived=["p1", pop_name], ancestral="ancestral")

    samples = [
        msprime.SampleSet(8, ploidy=2, population="p1"),
        msprime.SampleSet(12, ploidy=2, population=pop_name),
    ]
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=seqlen,
        recombination_rate=1e-8,
        random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=1e-7, random_seed=seed + 1)


@pytest.fixture(scope="module")
def chrom_dir(tmp_path_factory):
    """Three per-chrom .trees with different lengths, identical individual sets."""
    d = tmp_path_factory.mktemp("chroms")
    seqs = {1: 5_000, 2: 7_500, 3: 6_000}
    for n, sl in seqs.items():
        ts = _build_chrom(seqlen=float(sl), seed=100 + n)
        ts.dump(str(d / f"chromosome_{n}.trees"))
    return d, seqs


def test_canonicalize_pop_filter(canonicalize_mod, chrom_dir):
    """Canonicalizing with pop=p2 keeps exactly the p2 individuals (12 ind ⇒ 24 nodes)."""
    d, _ = chrom_dir
    ts = tskit.load(str(d / "chromosome_1.trees"))
    ts_c, stats = canonicalize_mod.canonicalize(ts, pop_name="p2")

    assert stats["n_eligible_individuals"] == 12
    assert ts_c.num_samples == 24
    # Sample IDs are 0..23 after simplify with the canonical sample-node order.
    assert np.array_equal(np.asarray(ts_c.samples()), np.arange(24))
    # filter_individuals=False: full upstream individual table preserved.
    assert ts_c.num_individuals == ts.num_individuals


def test_canonicalize_no_pop_keeps_all_sample_inds(canonicalize_mod, chrom_dir):
    d, _ = chrom_dir
    ts = tskit.load(str(d / "chromosome_1.trees"))
    ts_c, stats = canonicalize_mod.canonicalize(ts, pop_name=None)
    # 8 + 12 diploid individuals across both populations
    assert stats["n_eligible_individuals"] == 20
    assert ts_c.num_samples == 40


def test_canonicalize_cross_chrom_consistency(canonicalize_mod, chrom_dir):
    """Same eligible-individual set on every chromosome — required for concat."""
    d, _ = chrom_dir
    base = None
    for n in (1, 2, 3):
        ts = tskit.load(str(d / f"chromosome_{n}.trees"))
        _, stats = canonicalize_mod.canonicalize(ts, pop_name="p2")
        if base is None:
            base = stats
        assert stats["n_eligible_individuals"] == base["n_eligible_individuals"]
        assert stats["eligible_individual_ids_first10"] == base["eligible_individual_ids_first10"]


def test_canonicalize_one_mutation_per_site(canonicalize_mod, chrom_dir):
    """Every retained site must have exactly one mutation (multi-mut sites dropped)."""
    d, _ = chrom_dir
    ts = tskit.load(str(d / "chromosome_1.trees"))
    ts_c, stats = canonicalize_mod.canonicalize(ts, pop_name="p2")
    counts = np.bincount(ts_c.tables.mutations.site, minlength=ts_c.num_sites)
    assert ts_c.num_sites == ts_c.num_mutations
    if ts_c.num_sites:
        assert int(counts.min()) == 1
        assert int(counts.max()) == 1
    # stats track the drop accounting
    assert stats["num_sites_post"] == stats["num_mutations_post"]
    assert (
        stats["num_sites_post"]
        == stats["num_sites_post_simplify"] - stats["num_sites_dropped_multimut"]
    )


def test_canonicalize_unknown_pop_raises(canonicalize_mod, chrom_dir):
    d, _ = chrom_dir
    ts = tskit.load(str(d / "chromosome_1.trees"))
    with pytest.raises(ValueError, match="population 'p99' not found"):
        canonicalize_mod.canonicalize(ts, pop_name="p99")


def test_stream_concat_seqlen_and_samples(canonicalize_mod, concat_mod, chrom_dir, tmp_path):
    d, seqs = chrom_dir
    canon_paths = []
    for n in (1, 2, 3):
        ts = tskit.load(str(d / f"chromosome_{n}.trees"))
        ts_c, _ = canonicalize_mod.canonicalize(ts, pop_name="p2")
        p = tmp_path / f"chromosome_{n}.canon.trees"
        ts_c.dump(str(p))
        canon_paths.append(p)

    out = tmp_path / "all.trees"
    summary = concat_mod.stream_concat(canon_paths, out)

    assert summary["num_samples"] == 24
    assert summary["total_seqlen"] == pytest.approx(sum(seqs.values()))
    assert out.stat().st_size > 0

    ts_all = tskit.load(str(out))
    assert ts_all.num_samples == 24
    assert ts_all.sequence_length == pytest.approx(sum(seqs.values()))


def test_natural_sort_key_orders_chroms(concat_mod):
    paths = [Path(f"chromosome_{n}.trees") for n in (10, 1, 2, 22, 11)]
    ordered = sorted(paths, key=concat_mod.natural_sort_key)
    assert [p.name for p in ordered] == [
        "chromosome_1.trees",
        "chromosome_2.trees",
        "chromosome_10.trees",
        "chromosome_11.trees",
        "chromosome_22.trees",
    ]


def test_fingerprint_is_deterministic(canonicalize_mod, concat_mod, chrom_dir, tmp_path):
    """Re-canonicalizing + re-concatenating the same inputs gives the same hash."""
    d, _ = chrom_dir

    def build_concat(out_dir: Path) -> str:
        canon_paths = []
        for n in (1, 2, 3):
            ts = tskit.load(str(d / f"chromosome_{n}.trees"))
            ts_c, _ = canonicalize_mod.canonicalize(ts, pop_name="p2")
            p = out_dir / f"chromosome_{n}.canon.trees"
            ts_c.dump(str(p))
            canon_paths.append(p)
        out = out_dir / "all.trees"
        concat_mod.stream_concat(canon_paths, out)
        ts_all = tskit.load(str(out))
        return concat_mod.fingerprint_tables(ts_all)

    a = tmp_path / "run_a"
    a.mkdir()
    b = tmp_path / "run_b"
    b.mkdir()
    fp_a = build_concat(a)
    fp_b = build_concat(b)
    assert fp_a == fp_b
    assert len(fp_a) == 32  # blake2b-128 hex
