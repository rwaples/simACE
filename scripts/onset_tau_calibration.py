"""Acceptance of the onset-τ check over a folder's validated reps (simACE #29, ADR 0027).

Run after:

    pixi run simace run onset_tau --jobs 8 --no-plots
    pixi run simace validate onset_tau --jobs 8
    pixi run simace gather onset_tau

then ``pixi run python scripts/onset_tau_calibration.py onset_tau``. It reads
``results/{folder}/report_summary.tsv`` and tests, as one Bonferroni family at
a familywise 5%:

- per scenario × cell, and per cell pooled over scenarios: the mean of the
  scored z = (observed − expected) / SE is within ``Φ⁻¹(1 − α/2)/√n``, and
  ``SD² · (n − 1)`` lies within the ``χ²_{n−1}`` quantiles at ``α/2`` and
  ``1 − α/2``;
- per scenario: the number of reps with a failing cell is at most the
  binomial(reps, 0.05) quantile at ``1 − α``;
- coverage: every scenario × cell is scored in at least 95% of its reps.

Prints the table and exits 1 on any miss.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

MIN_COVERAGE = 0.95
REP_FAILURE_RATE = 0.05


def _z_columns(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("onset_tau_z_")]


def _mean_sd_tests(z: np.ndarray, alpha: float) -> tuple[bool, bool, float, float]:
    n = len(z)
    mean, sd = float(z.mean()), float(z.std(ddof=1))
    mean_ok = abs(mean) <= stats.norm.isf(alpha / 2) / math.sqrt(n)
    lo, hi = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], n - 1)
    sd_ok = lo <= sd**2 * (n - 1) <= hi
    return mean_ok, sd_ok, mean, sd


def main() -> None:
    """Print the acceptance table for one folder and exit 1 on any miss."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    frame = pl.read_csv(args.results / args.folder / "report_summary.tsv", separator="\t", infer_schema_length=None)
    cells = _z_columns(frame)
    scenarios = sorted(frame["scenario"].unique().to_list())

    groups: list[tuple[str, str, np.ndarray, int]] = []
    for scenario in scenarios:
        rows = frame.filter(pl.col("scenario") == scenario)
        for cell in cells:
            z = rows[cell].drop_nulls().to_numpy()
            if len(z):
                groups.append((scenario, cell.removeprefix("onset_tau_z_"), z, len(rows)))
    pooled = {}
    for _, cell, z, _ in groups:
        pooled.setdefault(cell, []).append(z)

    n_tests = 2 * len(groups) + 2 * len(pooled) + len(scenarios)
    alpha = 0.05 / n_tests
    misses: list[str] = []
    print(f"{n_tests} tests at alpha = {alpha:.2e} each (familywise 5%)\n")
    print(f"{'scenario':26s} {'cell':16s} {'n':>4s} {'cover':>6s} {'mean z':>7s} {'SD z':>6s}")
    for scenario, cell, z, n_reps in groups:
        mean_ok, sd_ok, mean, sd = _mean_sd_tests(z, alpha)
        cover = len(z) / n_reps
        flags = "" if mean_ok and sd_ok and cover >= MIN_COVERAGE else "  <-- MISS"
        print(f"{scenario:26s} {cell:16s} {len(z):4d} {cover:6.2f} {mean:+7.3f} {sd:6.3f}{flags}")
        if not mean_ok:
            misses.append(f"{scenario} {cell}: mean z {mean:+.3f}")
        if not sd_ok:
            misses.append(f"{scenario} {cell}: SD z {sd:.3f}")
        if cover < MIN_COVERAGE:
            misses.append(f"{scenario} {cell}: scored in {cover:.0%} of reps")

    print(f"\n{'pooled over scenarios':43s} {'n':>4s} {'mean z':>7s} {'SD z':>6s}")
    for cell, parts in pooled.items():
        z = np.concatenate(parts)
        mean_ok, sd_ok, mean, sd = _mean_sd_tests(z, alpha)
        flags = "" if mean_ok and sd_ok else "  <-- MISS"
        print(f"{cell:43s} {len(z):4d} {mean:+7.3f} {sd:6.3f}{flags}")
        if not mean_ok:
            misses.append(f"pooled {cell}: mean z {mean:+.3f}")
        if not sd_ok:
            misses.append(f"pooled {cell}: SD z {sd:.3f}")

    print(f"\n{'scenario':26s} {'reps':>5s} {'failing':>8s} {'allowed':>8s}")
    for scenario in scenarios:
        rows = frame.filter((pl.col("scenario") == scenario) & pl.col("onset_tau_n_scored").is_not_null())
        failing = int((rows["onset_tau_n_failed"] > 0).sum())
        allowed = int(stats.binom.isf(alpha, len(rows), REP_FAILURE_RATE))
        flags = "" if failing <= allowed else "  <-- MISS"
        print(f"{scenario:26s} {len(rows):5d} {failing:8d} {allowed:8d}{flags}")
        if failing > allowed:
            misses.append(f"{scenario}: {failing} failing reps > {allowed}")

    print("\nACCEPTED" if not misses else "\nNOT ACCEPTED:\n  " + "\n  ".join(misses))
    sys.exit(1 if misses else 0)


if __name__ == "__main__":
    main()
