"""Shared fixtures and Hypothesis strategies for the simace test suite."""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import settings
from hypothesis import strategies as st

from simace.core.parquet import load_parquet, save_parquet
from simace.core.schema import PEDIGREE
from simace.simulation.simulate import (
    generate_correlated_components,
    mating,
    reproduce,
    run_simulation,
)

# Hypothesis profiles, selected by ``HYPOTHESIS_PROFILE`` (default ``fast``).
# Property files carry no ``@settings`` of their own so that
# ``HYPOTHESIS_PROFILE=thorough`` actually deepens every one of them; the two
# documented carve-outs in tests/simulation/test_simulate_properties.py call
# run_simulation per example and keep an explicit budget.
settings.register_profile("fast", max_examples=100, deadline=None)
settings.register_profile("thorough", max_examples=500, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))

# Pedigree builders are capped small: the ascertainment/closure and sampling
# paths under test are quadratic in places, and Hypothesis reruns them many
# times per test.  cf. pedigree-graph's PEDIGREE_MAX_N = 25.
PEDIGREE_MAX_N = 40


def schema_pad(df: pl.DataFrame, schema: Mapping[str, str]) -> pl.DataFrame:
    """Pad ``df`` with zero/false defaults for any columns required by ``schema``.

    Lets unit-test fixtures stay focused on the columns under test while still
    producing a frame that satisfies the pipeline-stage schema contract.
    """
    n = len(df)
    new_cols = []
    for col, kinds in schema.items():
        if col in df.columns:
            continue
        if "f" in kinds:
            new_cols.append(pl.Series(col, np.zeros(n, dtype=np.float32)))
        elif "b" in kinds:
            new_cols.append(pl.Series(col, np.zeros(n, dtype=bool)))
        else:
            new_cols.append(pl.Series(col, np.zeros(n, dtype=np.int32)))
    return df.with_columns(new_cols) if new_cols else df


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def default_params():
    """Default simulation parameters matching config defaults."""
    return dict(
        seed=42,
        N=1000,
        G_ped=3,
        G_sim=3,
        mating_lambda=0.5,
        p_mztwin=0.02,
        A1=0.5,
        C1=0.2,
        E1=0.3,
        A2=0.5,
        C2=0.2,
        E2=0.3,
        rA=0.3,
        rC=0.5,
        assort1=0.0,
        assort2=0.0,
    )


@pytest.fixture
def tiny_pedigree_parquet(tmp_path: Path) -> Path:
    """Tiny schema-valid pedigree parquet (N=100, two generations)."""
    pedigree = run_simulation(
        seed=42,
        N=100,
        G_ped=2,
        G_sim=2,
        mating_lambda=0.5,
        p_mztwin=0.0,
        A1=0.5,
        C1=0.2,
        E1=0.3,
        A2=0.5,
        C2=0.2,
        E2=0.3,
        rA=0.3,
        rC=0.5,
        assort1=0.0,
        assort2=0.0,
    )
    path = tmp_path / "pedigree.parquet"
    save_parquet(pedigree, path)
    return path


@pytest.fixture
def tiny_phenotype_parquet(tmp_path: Path, tiny_pedigree_parquet: Path) -> Path:
    """Tiny schema-valid phenotype parquet built from the pedigree fixture."""
    from simace.phenotype import run_phenotype

    pedigree = load_parquet(tiny_pedigree_parquet)
    phenotype = run_phenotype(
        pedigree,
        G_pheno=1,
        seed=42,
        standardize="global",
        phenotype_model1="frailty",
        phenotype_params1={"distribution": "weibull", "scale": 316.228, "rho": 2.0},
        beta1=1.0,
        beta_sex1=0.0,
        phenotype_model2="frailty",
        phenotype_params2={"distribution": "weibull", "scale": 316.228, "rho": 2.0},
        beta2=1.0,
        beta_sex2=0.0,
    )
    path = tmp_path / "phenotype.parquet"
    save_parquet(phenotype, path)
    return path


@pytest.fixture
def tiny_censored_parquet(tmp_path: Path, tiny_phenotype_parquet: Path, tiny_pedigree_parquet: Path) -> Path:
    """Tiny schema-valid censored parquet built from the phenotype fixture."""
    from simace.censoring.censor import run_censor

    phenotype = load_parquet(tiny_phenotype_parquet)
    pedigree = load_parquet(tiny_pedigree_parquet)
    censored = run_censor(
        phenotype,
        pedigree,
        censor_age=80.0,
        seed=42,
        gen_censoring={},
        death_scale=79.433,
        death_rho=10.0,
    )
    path = tmp_path / "censored.parquet"
    save_parquet(censored, path)
    return path


@pytest.fixture
def founders_and_offspring(rng):
    """Create a one-generation setup: founders + one generation of offspring.

    Returns (pheno, sex, parents, twins, household_ids, offspring, sex_offspring)
    with known variance components.
    """
    N = 2000
    A1, C1 = 0.5, 0.2
    A2, C2 = 0.5, 0.2
    rA, rC = 0.3, 0.5
    E1 = 1.0 - A1 - C1
    E2 = 1.0 - A2 - C2

    sd_A1, sd_C1, sd_E1 = np.sqrt(A1), np.sqrt(C1), np.sqrt(E1)
    sd_A2, sd_C2, sd_E2 = np.sqrt(A2), np.sqrt(C2), np.sqrt(E2)

    sex = rng.binomial(size=N, n=1, p=0.5)
    a1, a2 = generate_correlated_components(rng, N, sd_A1, sd_A2, rA)
    c1, c2 = generate_correlated_components(rng, N, sd_C1, sd_C2, rC)
    e1 = rng.normal(size=N, scale=sd_E1)
    e2 = rng.normal(size=N, scale=sd_E2)
    pheno = np.stack([a1, c1, e1, a2, c2, e2], axis=-1)

    parents, twins, household_ids = mating(rng, sex, mating_lambda=0.5, p_mztwin=0.02)
    offspring, sex_offspring = reproduce(
        rng,
        pheno,
        parents,
        twins,
        household_ids,
        sd_A1,
        sd_E1,
        sd_C1,
        sd_A2,
        sd_E2,
        sd_C2,
        rA,
        rC,
    )
    return dict(
        pheno=pheno,
        sex=sex,
        parents=parents,
        twins=twins,
        household_ids=household_ids,
        offspring=offspring,
        sex_offspring=sex_offspring,
        sd_A1=sd_A1,
        sd_A2=sd_A2,
        sd_C1=sd_C1,
        sd_C2=sd_C2,
        sd_E1=sd_E1,
        sd_E2=sd_E2,
        rA=rA,
        rC=rC,
    )


def _contiguous(codes: Sequence[object]) -> list[int]:
    """Renumber ``codes`` to dense ``0..k-1`` labels, preserving first-seen order."""
    seen: dict[object, int] = {}
    return [seen.setdefault(code, len(seen)) for code in codes]


@st.composite
def pedigree_frame(draw, *, max_n=PEDIGREE_MAX_N, twins=False, liabilities=False) -> pl.DataFrame:
    """Draw a constructively valid simACE ``PEDIGREE`` frame.

    Domain validity is built in rather than filtered for: ids are unique and
    increase with row order, ``-1`` is the only missing-reference sentinel,
    every non-founder's mother and father come from the immediately preceding
    generation with consistent female/male roles, and household is assigned by
    mother (``simulate.py:795``) — so maternal half-sibs share a household and
    paternal half-sibs do not, matching the simulator.

    With ``twins=True`` some sibships gain an MZ pair: the links are symmetric,
    the pair shares both parents and a household, and the second twin's sex is
    forced to the first's, exactly as ``reproduce()`` does
    (``simulate.py:999``).  Founder sibships are synthetic — a founder's
    unmodelled mother is represented by its household alone.

    With ``liabilities=True`` each row gets drawn ``liability1``/``liability2``
    values and an A/C/E split that sums to them; the split is a fixed
    proportion, not an independent draw, because no property here inspects the
    variance decomposition.  Otherwise ``schema_pad`` supplies zeros.

    Args:
        max_n: soft cap on total rows across all generations.
        twins: whether to link MZ twin pairs.
        liabilities: whether to draw non-zero liabilities and ACE components.

    Returns:
        A frame satisfying ``assert_schema(..., PEDIGREE)``.
    """
    n_gen = draw(st.integers(min_value=1, max_value=3))
    cap = max(2, max_n // n_gen)

    ids: list[int] = []
    generations: list[int] = []
    sexes: list[int] = []
    mothers: list[int] = []
    fathers: list[int] = []
    twin_of: list[int] = []
    households: list[int] = []

    prev_females: list[int] = []
    prev_males: list[int] = []
    next_id = 0
    hh_offset = 0

    for g in range(n_gen):
        more_generations = g < n_gen - 1
        # A generation that must be mated needs at least one row of each sex.
        n = draw(st.integers(min_value=2 if more_generations else 1, max_value=cap))

        if g == 0:
            gen_mother = [-1] * n
            gen_father = [-1] * n
            # Founders are offspring of the unmodelled burn-in generations, so
            # they do have sibships and MZ twins -- represented here by a drawn
            # grouping, since their parents are not in the frame.  Giving each
            # founder its own sibship instead would make ``twins=True`` a no-op
            # on any single-generation frame.
            n_sibships = draw(st.integers(min_value=1, max_value=n))
            sibship: list = [draw(st.integers(min_value=0, max_value=n_sibships - 1)) for _ in range(n)]
        else:
            gen_mother = [draw(st.sampled_from(prev_females)) for _ in range(n)]
            gen_father = [draw(st.sampled_from(prev_males)) for _ in range(n)]
            sibship = list(zip(gen_mother, gen_father, strict=True))

        # Twin pairs are chosen first, because an MZ pair shares one sex.
        # ``sex_leader[k]`` is the row whose sex row ``k`` takes; a twin merges
        # into its partner's, leaving one fewer independent sex to draw.
        gen_twin = [-1] * n
        sex_leader = list(range(n))
        n_sex_groups = n
        if twins:
            buckets: dict[object, list[int]] = {}
            for k, key in enumerate(sibship):
                buckets.setdefault(key, []).append(k)
            for members in buckets.values():
                # Mirrors mating(): the first two offspring of a twin-flagged
                # mating are the MZ pair.
                if len(members) < 2 or not draw(st.booleans()):
                    continue
                # A generation that must be mated needs two sex groups to hold
                # both sexes, so a merge that would collapse it to one is
                # skipped rather than filtered out afterwards.
                if more_generations and n_sex_groups <= 2:
                    continue
                first, second = members[0], members[1]
                gen_twin[first], gen_twin[second] = next_id + second, next_id + first
                sibship[second] = sibship[first]
                sex_leader[second] = first
                n_sex_groups -= 1

        gen_sex = [0] * n
        for index, leader in enumerate(k for k in range(n) if sex_leader[k] == k):
            if more_generations and index < 2:
                gen_sex[leader] = index  # guarantees one female and one male
            else:
                gen_sex[leader] = draw(st.integers(min_value=0, max_value=1))
        gen_sex = [gen_sex[leader] for leader in sex_leader]

        # Household by mother, so maternal half-sibs share one and paternal
        # half-sibs do not.  A founder has no in-frame mother, so its synthetic
        # sibship — twin-merged above — stands in for her.
        gen_hh = _contiguous(sibship if g == 0 else gen_mother)

        ids.extend(range(next_id, next_id + n))
        generations.extend([g] * n)
        sexes.extend(gen_sex)
        mothers.extend(gen_mother)
        fathers.extend(gen_father)
        twin_of.extend(gen_twin)
        households.extend(hh_offset + household for household in gen_hh)

        prev_females = [next_id + k for k in range(n) if gen_sex[k] == 0]
        prev_males = [next_id + k for k in range(n) if gen_sex[k] == 1]
        next_id += n
        hh_offset += max(gen_hh) + 1

    frame = pl.DataFrame(
        {
            "id": np.asarray(ids, dtype=np.int32),
            "generation": np.asarray(generations, dtype=np.int32),
            "sex": np.asarray(sexes, dtype=np.int32),
            "mother": np.asarray(mothers, dtype=np.int32),
            "father": np.asarray(fathers, dtype=np.int32),
            "twin": np.asarray(twin_of, dtype=np.int32),
            "household_id": np.asarray(households, dtype=np.int32),
        }
    )

    if liabilities:
        liability_value = st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False, width=32)
        columns = []
        for trait in (1, 2):
            values = np.asarray([draw(liability_value) for _ in range(len(frame))], dtype=np.float64)
            columns += [
                pl.Series(f"liability{trait}", values),
                pl.Series(f"A{trait}", 0.5 * values),
                pl.Series(f"C{trait}", 0.3 * values),
                pl.Series(f"E{trait}", 0.2 * values),
            ]
        frame = frame.with_columns(columns)

    return schema_pad(frame, PEDIGREE)


def relabel_ids(frame: pl.DataFrame, data) -> pl.DataFrame:
    """Rewrite ``frame``'s ids through a gapped, order-preserving bijection.

    Models the non-contiguous ids left behind when dropout, ascertainment, or
    sampling removes rows without renumbering.  Row topology — and therefore
    every result under test — is unchanged; only the id *values* move.

    Gaps and the starting offset are deliberately small (max id stays under
    roughly ``5n``).  Unlike pedigree-graph's million-spaced stress relabelling,
    simACE's ``create_sample()`` and ``PedigreeArrays`` use direct-address
    tables sized by ``max_id + 1``, and real filtered simACE ids stay inside the
    original dense simulation range.

    Args:
        frame: a frame drawn from :func:`pedigree_frame`.
        data: Hypothesis ``st.data()`` object supplying the gaps and offset.

    Returns:
        The relabelled frame, with mother/father/twin references remapped.
    """
    n = len(frame)
    gaps = np.asarray(
        data.draw(st.lists(st.integers(min_value=1, max_value=3), min_size=n, max_size=n)),
        dtype=np.int64,
    )
    offset = data.draw(st.integers(min_value=0, max_value=n + 2))
    old_ids = frame["id"].to_numpy()
    new_ids = (offset + np.cumsum(gaps)).astype(np.int32)
    mapping = dict(zip(old_ids.tolist(), new_ids.tolist(), strict=True))

    columns = [pl.Series("id", new_ids)]
    for col in ("mother", "father", "twin"):
        values = frame[col].to_numpy()
        remapped = np.asarray([-1 if v < 0 else mapping[int(v)] for v in values], dtype=np.int32)
        columns.append(pl.Series(col, remapped))
    return frame.with_columns(columns)
