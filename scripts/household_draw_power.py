r"""Power of the ``household_draw`` test against a wrong paternal C transmission (simACE #81).

Simulates pedigrees with b_f = 0 at each N and seed, then scores each with
:func:`simace.analysis.validate.inheritance.run_household_draw` as if
``Ctm_paternal1`` were −δ. The draw then carries δ·C_father, which PHS pairs
share, so its PHS correlation is about δ²/(1 + δ²) when b_m = b_f = 0.

Prints, per N and δ: the median PHS pair count, the mean observed and
predicted correlation, the per-rep z (mean, SD, share above 4), and the
power of the verdict's mean-z test over 20 reps (the test's ``min_reps``).
Power is the share of 5,000 resamples of 20 reps whose |mean z| exceeds
Φ⁻¹(1 − α/2)/√20, with α the per-test level of ``inheritance_calibration``'s
folder verdict (0.05/108, ADR 0029). Parameters are the fixture's in
``tests/analysis/test_validate_inheritance.py``.

Example::

    pixi run python scripts/household_draw_power.py --n 1000 2000 20000 100000 --seeds 50
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

from simace.analysis.validate.calibrated import RepInputs
from simace.analysis.validate.inheritance import run_household_draw
from simace.core.cohort import write_pedigree
from simace.simulation.simulate import run_simulation

PARAMS = dict(
    G_ped=3,
    G_sim=3,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.2,
    E1=0.3,
    A2=0.4,
    C2=0.3,
    E2=0.3,
    rA=0.3,
    rC=0.0,
)
DELTAS = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
REPS = 20
ALPHA = 0.05 / 108


def score(n: int, seed: int, deltas: tuple[float, ...], work: Path) -> list[dict]:
    """One pedigree at ``n`` and ``seed`` scored at each δ: the trait-1 PHS cell's pairs, r, SE and z."""
    pedigree_path = work / "pedigree.parquet"
    write_pedigree(run_simulation(seed=seed, N=n, **PARAMS).pedigree, pedigree_path)
    rows = []
    for delta in deltas:
        report_path = work / "report.yaml"
        parameters = {**PARAMS, "seed": seed, "N": n, "Ctm_paternal1": -delta}
        report_path.write_text(yaml.safe_dump({"inputs": {"parameters": parameters}}))
        cell = run_household_draw(RepInputs(pedigree_path, report_path, seed=seed, phenotype={}))["cells"]["trait1_PHS"]
        rows.append(
            dict(N=n, seed=seed, delta=delta, n_pairs=cell["n_pairs"], r=cell["observed"], se=cell["se"], z=cell["z"])
        )
    return rows


def summarize(table: pl.DataFrame) -> pl.DataFrame:
    """Per N and δ: pairs, observed and predicted r, per-rep z, and the 20-rep mean-z power."""
    bound = float(stats.norm.isf(ALPHA / 2)) / np.sqrt(REPS)
    rng = np.random.default_rng(0)
    out = []
    for (n, delta), group in table.group_by(["N", "delta"], maintain_order=True):
        z = group["z"].to_numpy()
        means = z[rng.integers(0, len(z), (5000, REPS))].mean(axis=1)
        out.append(
            dict(
                N=n,
                delta=delta,
                reps=len(z),
                pairs=int(group["n_pairs"].median()),
                r=group["r"].mean(),
                r_predicted=delta**2 / (1 + delta**2),
                mean_z=z.mean(),
                sd_z=z.std(ddof=1),
                share_z_over_4=(z > 4).mean(),
                power_20_reps=(np.abs(means) > bound).mean(),
            )
        )
    return pl.DataFrame(out).sort("N", "delta")


def main() -> None:
    """Score every N and seed, then print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, nargs="+", default=[1000, 2000, 20000])
    parser.add_argument("--seeds", type=int, default=50, help="seeds 1..SEEDS per N")
    parser.add_argument("--out", type=Path, help="write the per-rep rows to this TSV")
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as work:
        for n in args.n:
            for seed in range(1, args.seeds + 1):
                rows.extend(score(n, seed, DELTAS, Path(work)))
            print(f"N = {n} done", file=sys.stderr, flush=True)
    table = pl.DataFrame(rows)
    if args.out:
        table.write_csv(args.out, separator="\t")
    print(f"mean-z bound over {REPS} reps at alpha {ALPHA:.3g}: {stats.norm.isf(ALPHA / 2) / np.sqrt(REPS):.3f}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        print(summarize(table))


if __name__ == "__main__":
    main()
