"""Echo simulation scenario parameters to a YAML sidecar.

Snakemake's ``simulate`` rule writes ``pedigree.full.parquet``;
downstream rules (``validate``, ``stats``, ``assemble_atlas``) consume
``params.yaml`` for scenario provenance.  ``params.yaml`` is purely an
echo of the scenario config — no computation; ``run_simulation`` does
not need to produce it — so it lives in its own rule and uses the same
``run_wrapper`` seam as every other stage.

PR2 mirrors the prior hand-rolled wrapper output exactly: ``E1``/``E2``
are echoed as-is (``null`` allowed), ``assort_matrix`` is included only
when not ``None``.  PR3 will reconcile this with the CLI's stricter
behavior.
"""

from __future__ import annotations

__all__ = ["cli", "emit_params"]

import argparse
from typing import Any


def emit_params(
    *,
    seed: int,
    rep: int,
    A1: float,
    C1: float,
    E1: float | None,
    A2: float,
    C2: float,
    E2: float | None,
    rA: float,
    rC: float,
    rE: float,
    N: int,
    G_ped: int,
    G_sim: int | None,
    mating_lambda: float,
    p_mztwin: float,
    assort1: float,
    assort2: float,
    assort_matrix: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Build the params.yaml dict for a single replicate.

    All keyword-only so :func:`simace.core.snakemake_adapter.run_wrapper`
    can introspect the signature and pull each value from
    ``snakemake.params`` by name.

    Args:
        seed: per-replicate seed (already offset by rep upstream).
        rep: replicate number (1-based).
        A1: trait-1 additive-genetic variance.
        C1: trait-1 shared-environment variance.
        E1: trait-1 unique-environment variance.  Echoed as-is —
            ``None`` round-trips as ``null`` in YAML, matching the prior
            hand-rolled wrapper.
        A2: trait-2 additive-genetic variance.
        C2: trait-2 shared-environment variance.
        E2: trait-2 unique-environment variance.  Echoed as-is.
        rA: cross-trait genetic correlation.
        rC: cross-trait shared-environment correlation.
        rE: cross-trait unique-environment correlation.
        N: founder population size.
        G_ped: pedigree generations.
        G_sim: simulation generations including burn-in.
        mating_lambda: ZTP mating count parameter.
        p_mztwin: MZ twin probability.
        assort1: trait-1 assortative-mating correlation.
        assort2: trait-2 assortative-mating correlation.
        assort_matrix: optional 2x2 correlation matrix; included in the
            dict only when not ``None``.

    Returns:
        Dict to be serialized to ``params.yaml`` via :func:`dump_yaml`.
    """
    out: dict[str, Any] = {
        "seed": seed,
        "rep": rep,
        "A1": A1,
        "C1": C1,
        "E1": E1,
        "A2": A2,
        "C2": C2,
        "E2": E2,
        "rA": rA,
        "rC": rC,
        "rE": rE,
        "N": N,
        "G_ped": G_ped,
        "G_sim": G_sim,
        "mating_lambda": mating_lambda,
        "p_mztwin": p_mztwin,
        "assort1": assort1,
        "assort2": assort2,
    }
    if assort_matrix is not None:
        out["assort_matrix"] = assort_matrix
    return out


def cli() -> None:
    """Command-line interface for emitting params.yaml from a scenario config."""
    from simace.core.cli_base import add_logging_args, init_logging
    from simace.core.yaml_io import dump_yaml, load_yaml

    parser = argparse.ArgumentParser(description="Emit params.yaml from a scenario config")
    add_logging_args(parser)
    parser.add_argument("--config", required=True, help="Scenario config YAML (single scenario, flat keys)")
    parser.add_argument("--rep", type=int, required=True, help="Replicate number (1-based)")
    parser.add_argument("--output", required=True, help="Output params.yaml path")
    args = parser.parse_args()
    init_logging(args)

    cfg = load_yaml(args.config)
    seed = int(cfg["seed"]) + args.rep - 1
    params = emit_params(
        seed=seed,
        rep=args.rep,
        A1=cfg["A1"],
        C1=cfg["C1"],
        E1=cfg.get("E1"),
        A2=cfg["A2"],
        C2=cfg["C2"],
        E2=cfg.get("E2"),
        rA=cfg["rA"],
        rC=cfg["rC"],
        rE=cfg.get("rE", 0.0),
        N=cfg["N"],
        G_ped=cfg["G_ped"],
        G_sim=cfg.get("G_sim"),
        mating_lambda=cfg["mating_lambda"],
        p_mztwin=cfg["p_mztwin"],
        assort1=cfg.get("assort1", 0.0),
        assort2=cfg.get("assort2", 0.0),
        assort_matrix=cfg.get("assort_matrix"),
    )
    dump_yaml(params, args.output, sort_keys=True)
