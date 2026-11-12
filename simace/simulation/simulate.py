"""ACE pedigree simulation.

Simulates multi-generational pedigrees with:
- A: Additive genetic component
- C: Common/shared environment component
- E: Unique environment component

Supports single-trait and two-trait (bivariate) modes with configurable
cross-trait correlations for genetic (rA) and common environment (rC) components.

:func:`run_simulation` drives the generation loop. One generation's mating is
:mod:`.mating` (its numba kernels are :mod:`.pairing_kernels`), offspring
components are :mod:`.reproduce`, and the pedigree arrays it fills are
:mod:`.pedigree_buffers`. The public names of those modules are re-exported
here.
"""

__all__ = [
    "METROPOLIS_TOL",
    "allocate_offspring",
    "balance_mating_slots",
    "draw_mating_counts",
    "generate_correlated_components",
    "generate_mendelian_noise",
    "pair_partners",
    "place_twins",
    "reproduce",
    "resolve_per_gen_param",
    "run_simulation",
]

import argparse
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from simace.core.cohort import write_pedigree
from simace.core.schema import PEDIGREE
from simace.core.stage import stage
from simace.simulation.assortment import (
    METROPOLIS_TOL,
    AssortmentPlan,
    assortment_schedule,
    write_mating_log,
)
from simace.simulation.inbreeding_plan import DEFAULT_DEPTH, INBREEDING_MODES, resolve
from simace.simulation.mating import (
    _mating_standard,
    _mating_wf,
    allocate_offspring,
    balance_mating_slots,
    draw_mating_counts,
    pair_partners,
    place_twins,
)
from simace.simulation.params import CTransmission, SimulationParams, resolve_per_gen_param
from simace.simulation.pedigree_buffers import _arrays_to_dataframe, _fill_pedigree_slice, _init_pedigree_arrays
from simace.simulation.reproduce import generate_correlated_components, generate_mendelian_noise, reproduce

if TYPE_CHECKING:
    from typing import Any

    import polars as pl

    from simace.simulation.inbreeding import Inbreeding

logger = logging.getLogger(__name__)


@stage(reads=None, writes=PEDIGREE)
def run_simulation(
    *,
    seed: int,
    N: int,
    G_ped: int,
    mating_lambda: float,
    p_mztwin: float,
    A1: float,
    C1: float,
    A2: float,
    C2: float,
    rA: float,
    rC: float,
    rE: float = 0.0,
    E1: float | dict[int, float],
    E2: float | dict[int, float],
    G_sim: int | None = None,
    assort1: float | dict[int, float] = 0.0,
    assort2: float | dict[int, float] = 0.0,
    assort_matrix: list[list[float]] | np.ndarray | None = None,
    mating_model: str = "standard",
    Ctm_maternal1: float = 0.0,
    Ctm_paternal1: float = 0.0,
    Ctm_maternal2: float = 0.0,
    Ctm_paternal2: float = 0.0,
    inbreeding_mode: str = "off",
    inbreeding_depth: int = DEFAULT_DEPTH,
    clip_log: list[dict[str, Any]] | None = None,
    mating_log: list[dict[str, Any]] | None = None,
) -> pl.DataFrame:
    """Run the full ACE simulation for two correlated traits.

    Variance components A (additive genetic), C (shared environment), and
    E (unique environment) are specified as absolute variances.  A and C are
    constant across generations; E may be specified per-generation via a dict
    mapping generation index to value (forward-filled for missing keys).

    Args:
        seed: Random seed
        N: Population size per generation (positive integer)
        G_ped: Number of generations to record in pedigree (integer >= 1)
        mating_lambda: Poisson lambda for zero-truncated mating count distribution (> 0)
        p_mztwin: MZ twin rate per birth, in [0, 1)
        A1: Trait 1 additive genetic variance (>= 0).
        C1: Trait 1 shared-environment variance (>= 0).
        A2: Trait 2 additive genetic variance (>= 0).
        C2: Trait 2 shared-environment variance (>= 0).
        rA: Genetic correlation between traits, in [-1, 1]
        rC: Common environment correlation between traits, in [-1, 1]
        rE: Unique environment correlation between traits, in [-1, 1].
            Default 0 (independent E across traits).
        E1: Trait 1 unique-environment variance.  Scalar (constant) or dict
            mapping generation index → value (forward-filled).
        E2: Trait 2 unique-environment variance.  Same format as E1.
        G_sim: Total generations to simulate (default: G_ped). First G_sim - G_ped
               generations are burn-in and discarded from output.
        assort1: Target mate Pearson correlation on trait 1 liability, in [-1, 1].
            Scalar (constant across generations) or dict mapping raw simulation
            iteration → value (forward-filled, same coordinate system as E/C).
            **Mutually exclusive with ``assort_matrix``** — pass scalar AM if
            you want a fixed 2x2 mate-correlation matrix.
        assort2: Same format as assort1, for trait 2.
        assort_matrix: Optional full 2x2 mate correlation matrix R_mf.
            Overrides assort1/assort2 diagonal with matrix diagonal. Only
            valid when both assort1 and assort2 are scalars.
        mating_model: ``"standard"`` (default) or ``"wright_fisher"``.
            Under ``"wright_fisher"``, ``mating_lambda``, ``p_mztwin``,
            ``assort1``, ``assort2``, and ``assort_matrix`` are silently
            ignored at runtime — the WF path samples each offspring's
            mother and father independently from the prior generation's
            females / males, with multinomial offspring counts and no MZ
            twins.  See ``docs/adr/0002-wright-fisher-mating-model.md``.
        Ctm_maternal1: Trait 1 maternal C transmission coefficient ``b_m``
            in [0, 1] (ADR 0025). Default 0, the fresh household draw.
        Ctm_paternal1: Trait 1 paternal C transmission coefficient ``b_f``.
        Ctm_maternal2: Trait 2 maternal C transmission coefficient.
        Ctm_paternal2: Trait 2 paternal C transmission coefficient.
        inbreeding_mode: ``"off"`` (default), ``"approx"`` or ``"exact"``:
            whether and how each offspring's Mendelian residual is scaled by
            ``sqrt(1 - F_bar)``, ``F_bar`` its parents' mean inbreeding
            coefficient (ADR 0033).
        inbreeding_depth: Generations ``approx`` traces exactly (>= 1).
        clip_log: Under C transmission, receives a record per fresh-draw
            eigenvalue clipped to 0, each with its ``generation`` in
            recorded coordinates (negative during burn-in), ``excess``,
            ``tolerance`` and ``direction``.
        mating_log: Receives one pairing record per mating iteration, with
            its ``iteration``, its offspring's ``generation`` in recorded
            coordinates, and ``mode`` (``random``, ``single``, ``both`` or
            ``wright_fisher``). An assortative record also has ``M``,
            ``proposals``, ``max_proposals``, the final slot-level
            ``errors`` against the targets, ``duplicates_left`` and
            ``converged`` (at least one error, and every error within
            ``METROPOLIS_TOL``; a record that pinned nothing is not
            converged).

    Returns:
        Pedigree DataFrame with columns id, sex, mother, father, twin,
        generation, household_id, A1, C1, E1, liability1, A2, C2, E2,
        liability2.

    Raises:
        ValueError: if any parameter is outside its valid range
    """
    # Validate + normalize inputs.
    # run_simulation is also a public test API, so validation cannot rely on
    # config-load having run. See simace.simulation.params.SimulationParams.
    params = SimulationParams.create(
        seed=seed,
        N=N,
        G_ped=G_ped,
        mating_lambda=mating_lambda,
        p_mztwin=p_mztwin,
        A1=A1,
        C1=C1,
        A2=A2,
        C2=C2,
        rA=rA,
        rC=rC,
        rE=rE,
        E1=E1,
        E2=E2,
        G_sim=G_sim,
        assort1=assort1,
        assort2=assort2,
        assort_matrix=assort_matrix,
        mating_model=mating_model,
        c_transmission=CTransmission(maternal=(Ctm_maternal1, Ctm_maternal2), paternal=(Ctm_paternal1, Ctm_paternal2)),
        inbreeding_mode=inbreeding_mode,
        inbreeding_depth=inbreeding_depth,
    )
    G_sim = params.G_sim

    logger.info("Starting simulation: N=%d, G_ped=%d, seed=%d", N, G_ped, seed)
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    # A and C are scalars (SimulationParams rejects maps); E may be a per-gen
    # dict. These run for both mating models since reproduce() needs them.
    E1_per_gen = resolve_per_gen_param(E1, G_sim, name="E1")
    E2_per_gen = resolve_per_gen_param(E2, G_sim, name="E2")

    sd_C1 = np.sqrt(C1)
    sd_C2 = np.sqrt(C2)
    sd_E1_per_gen = [np.sqrt(v) for v in E1_per_gen]
    sd_E2_per_gen = [np.sqrt(v) for v in E2_per_gen]

    # A is constant across generations
    sd_A1 = np.sqrt(A1)
    sd_A2 = np.sqrt(A2)

    # Standard-model AM plan: the assortment schedule, rho_w, and the
    # |rho_w| < 1 + 4x4 PSD checks are validated up front inside build().
    # WF skips this entirely — no AM, no rho_w-driven cross-AM.
    plan: AssortmentPlan | None = None
    if mating_model == "standard":
        assort1_per_gen, assort2_per_gen = assortment_schedule(assort1, assort2, G_sim=G_sim, assort_matrix=params.R_mf)
        plan = AssortmentPlan.build(
            assort1_per_gen=assort1_per_gen,
            assort2_per_gen=assort2_per_gen,
            R_mf_user=params.R_mf,
            rA=rA,
            rC=rC,
            rE=rE,
            A1=A1,
            A2=A2,
            C1=C1,
            C2=C2,
            E1_per_gen=E1_per_gen,
            E2_per_gen=E2_per_gen,
            G_sim=G_sim,
        )

    # Initialize founders with correlated components (using gen-0 C/E variances)
    sex = rng.binomial(size=N, n=1, p=0.5)

    # A components: correlated via rA (constant across generations)
    a1, a2 = generate_correlated_components(rng, N, sd_A1, sd_A2, rA)

    # C components: correlated via rC (gen-0 variance)
    c1, c2 = generate_correlated_components(rng, N, sd_C1, sd_C2, rC)

    # E components: correlated via rE (gen-0 variance); independent when rE=0
    if rE != 0:
        e1, e2 = generate_correlated_components(rng, N, sd_E1_per_gen[0], sd_E2_per_gen[0], rE)
    else:
        e1 = rng.normal(size=N, loc=0, scale=sd_E1_per_gen[0])
        e2 = rng.normal(size=N, loc=0, scale=sd_E2_per_gen[0])

    pheno = np.stack([a1, c1, e1, a2, c2, e2], axis=-1)  # column order: _A1 .. _E2

    # Simulate generations
    burnin = G_sim - G_ped

    # Pre-allocate pedigree arrays (avoids frame concat per generation)
    total_individuals = N * G_ped
    if total_individuals > np.iinfo(np.int32).max:
        raise ValueError(
            f"Total pedigree size {total_individuals:,} exceeds int32 max "
            f"({np.iinfo(np.int32).max:,}). Reduce N or G_ped."
        )
    ped_arrays = _init_pedigree_arrays(total_individuals, inbreeding=params.inbreeding.mode != "off")
    ped_offset = 0
    id_offset = 0
    household_offset = 0

    # Inbreeding (ADR 0033): iteration i's parents are iteration i - 1's
    # offspring, so the Mendelian scale reads the previous step's F.
    inbreeding: Inbreeding | None = None
    F_offspring: np.ndarray | None = None
    if params.inbreeding.mode != "off":
        from simace.simulation.inbreeding import Inbreeding

        inbreeding = Inbreeding(params.inbreeding, N, G_sim, mother_sorted=mating_model == "standard")
        F_offspring = np.zeros(N)

    for i in range(G_sim):
        t_gen = time.perf_counter()

        if mating_model == "standard":
            assert plan is not None  # built above under the same guard
            a1_i, a2_i, rho_w_i, R_mf_i = plan.for_generation(i)
            pairing: list[dict[str, Any]] = []
            try:
                parents, twins, household_ids = _mating_standard(
                    rng,
                    sex,
                    mating_lambda,
                    p_mztwin,
                    pheno=pheno,
                    assort1=a1_i,
                    assort2=a2_i,
                    rho_w=rho_w_i,
                    assort_matrix=R_mf_i,
                    pairing_log=pairing,
                )
            except ValueError as exc:
                raise ValueError(f"Mating iteration {i}: {exc}") from exc
        else:  # wright_fisher
            parents, twins, household_ids = _mating_wf(rng, sex, N, generation=i)
            pairing = [{"mode": "wright_fisher"}]
        if mating_log is not None:
            mating_log.extend({"iteration": i, "generation": i - burnin, **record} for record in pairing)
        t_mate = time.perf_counter()
        mendelian_scale = None
        if inbreeding is not None:
            assert F_offspring is not None
            mendelian_scale = np.sqrt(1.0 - (F_offspring[parents[:, 0]] + F_offspring[parents[:, 1]]) / 2)
            F_offspring = inbreeding.step(parents, twins)
        t_inbreeding = time.perf_counter()
        generation_clips: list[dict[str, Any]] = []
        try:
            pheno, sex = reproduce(
                rng,
                pheno,
                parents,
                twins,
                household_ids,
                sd_A1,
                sd_E1_per_gen[i],
                sd_C1,
                sd_A2,
                sd_E2_per_gen[i],
                sd_C2,
                rA,
                rC,
                rE,
                params.c_transmission,
                generation_clips,
                mendelian_scale,
            )
        except ValueError as exc:
            raise ValueError(f"Generation {i}: {exc}") from exc
        if clip_log is not None:
            clip_log.extend({"generation": i - burnin, **clip} for clip in generation_clips)
        t_repro = time.perf_counter()
        if i >= burnin:
            n = len(sex)
            is_founder = i == burnin
            parent_offset = id_offset - n if not is_founder else 0
            _fill_pedigree_slice(
                ped_arrays,
                ped_offset,
                pheno,
                sex,
                parents,
                twins,
                household_ids,
                generation=i - burnin,
                is_founder=is_founder,
                id_offset=id_offset,
                parent_offset=parent_offset,
                household_offset=household_offset,
                F=F_offspring,
            )
            household_offset += int(household_ids.max()) + 1
            id_offset += n
            ped_offset += n
        t_fill = time.perf_counter()

        # Per-generation data shape checkpoints
        fam_sizes = np.bincount(household_ids)
        logger.info(
            "Generation %d: %d twins, mean family size %.2f [mating=%.1fs, inbreeding=%.1fs, reproduce=%.1fs, "
            "fill=%.1fs, total=%.1fs]",
            i,
            len(twins) * 2,
            fam_sizes.mean(),
            t_mate - t_gen,
            t_inbreeding - t_mate,
            t_repro - t_inbreeding,
            t_fill - t_repro,
            t_fill - t_gen,
        )

    pedigree = _arrays_to_dataframe(ped_arrays, ped_offset)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Simulation complete in %.1fs: pedigree has %d individuals",
        elapsed,
        len(pedigree),
    )

    return pedigree


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line interface for running ACE simulations."""
    import json

    from simace.core.cli_base import add_logging_args, add_version_arg, float_or_generation_map, init_logging
    from simace.core.publish import publish

    parser = argparse.ArgumentParser(prog=prog, description="Run ACE pedigree simulation")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--N", type=int, default=1000, help="Founder population size")
    parser.add_argument("--G-ped", type=int, default=3, help="Number of pedigree generations")
    parser.add_argument("--G-sim", type=int, default=None, help="Number of burn-in generations (default: G_ped)")
    parser.add_argument(
        "--mating-model",
        type=str,
        default="standard",
        choices=("standard", "wright_fisher"),
        help="Mating model: standard (ZTP fecundity, paired) or wright_fisher (sex-structured idealized WF).",
    )
    parser.add_argument(
        "--mating-lambda", type=float, default=0.5, help="ZTP mating count lambda (standard model only)"
    )
    parser.add_argument("--p-mztwin", type=float, default=0.004, help="MZ twin rate per birth (standard model only)")
    parser.add_argument("--A1", type=float, default=0.5, help="Additive genetic variance for trait 1")
    parser.add_argument("--C1", type=float, default=0.2, help="Shared environment variance for trait 1")
    parser.add_argument(
        "--E1",
        type=float_or_generation_map,
        required=True,
        help="Unique environment variance for trait 1: a scalar or a JSON per-generation map",
    )
    parser.add_argument("--A2", type=float, default=0.5, help="Additive genetic variance for trait 2")
    parser.add_argument("--C2", type=float, default=0.2, help="Shared environment variance for trait 2")
    parser.add_argument(
        "--E2",
        type=float_or_generation_map,
        required=True,
        help="Unique environment variance for trait 2: a scalar or a JSON per-generation map",
    )
    parser.add_argument("--rA", type=float, default=0.5, help="Cross-trait genetic correlation")
    parser.add_argument("--rC", type=float, default=0.3, help="Cross-trait shared environment correlation")
    parser.add_argument("--rE", type=float, default=0.0, help="Cross-trait unique environment correlation")
    parser.add_argument(
        "--assort1",
        type=float_or_generation_map,
        default=0.0,
        help="Mate correlation on trait 1 liability: a scalar or a JSON per-generation map",
    )
    parser.add_argument(
        "--assort2",
        type=float_or_generation_map,
        default=0.0,
        help="Mate correlation on trait 2 liability: a scalar or a JSON per-generation map",
    )
    parser.add_argument(
        "--assort-matrix",
        type=json.loads,
        default=None,
        help="Optional 2x2 mate correlation matrix as JSON, e.g. '[[0.3, 0.05], [0.05, 0.15]]'",
    )
    for side, parent in (("maternal", "mother"), ("paternal", "father")):
        for t in (1, 2):
            parser.add_argument(
                f"--Ctm-{side}{t}",
                type=float,
                default=0.0,
                help=f"Share of the {parent}'s C{t} an offspring inherits, in [0, 1] (ADR 0025)",
            )
    parser.add_argument(
        "--inbreeding-mode",
        default="off",
        choices=INBREEDING_MODES,
        help="Scale the Mendelian residual by sqrt(1 - parental F): off, approx or exact (ADR 0033)",
    )
    parser.add_argument("--inbreeding-depth", type=int, default=DEFAULT_DEPTH, help="Generations approx traces exactly")
    parser.add_argument("--output-pedigree", required=True, help="Output pedigree parquet path")
    parser.add_argument("--output-mating", default=None, help="Output path of the pairing records (mating.yaml)")
    args = parser.parse_args(argv)

    init_logging(args)

    c_transmission = CTransmission(
        maternal=(args.Ctm_maternal1, args.Ctm_maternal2), paternal=(args.Ctm_paternal1, args.Ctm_paternal2)
    )
    clip_log: list[dict[str, Any]] | None = [] if c_transmission.active else None
    mating_log: list[dict[str, Any]] = []
    pedigree = run_simulation(
        seed=args.seed,
        N=args.N,
        G_ped=args.G_ped,
        mating_lambda=args.mating_lambda,
        p_mztwin=args.p_mztwin,
        A1=args.A1,
        C1=args.C1,
        A2=args.A2,
        C2=args.C2,
        rA=args.rA,
        rC=args.rC,
        rE=args.rE,
        E1=args.E1,
        E2=args.E2,
        G_sim=args.G_sim,
        assort1=args.assort1,
        assort2=args.assort2,
        assort_matrix=args.assort_matrix,
        mating_model=args.mating_model,
        Ctm_maternal1=args.Ctm_maternal1,
        Ctm_paternal1=args.Ctm_paternal1,
        Ctm_maternal2=args.Ctm_maternal2,
        Ctm_paternal2=args.Ctm_paternal2,
        inbreeding_mode=args.inbreeding_mode,
        inbreeding_depth=args.inbreeding_depth,
        clip_log=clip_log,
        mating_log=mating_log,
    )

    inbreeding = None
    if args.inbreeding_mode != "off":
        inbreeding = resolve(args.inbreeding_mode, args.inbreeding_depth, args.N, args.G_sim or args.G_ped)
    with publish(args.output_pedigree) as (tmp,):
        write_pedigree(pedigree, tmp, c_transmission_clips=clip_log, inbreeding=inbreeding)
    if args.output_mating is not None:
        write_mating_log(mating_log, args.output_mating)
