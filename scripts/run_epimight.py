#!/usr/bin/env python
r"""Standalone EPIMIGHT runner for a single simACE phenotype.parquet.

Drives the full pipeline without Snakemake or the private fitACE repo:

    phenotype.parquet
        -> trait{1,2}.epimight_in.parquet  +  true_parameters.json   (Python)
        -> tsv/{h2_d1,h2_d2,gc}_meta_<kind>.tsv per relationship kind (R)
        -> summary.tsv aggregated over kinds                          (Python)

Downsampling of relatives and Rubin's-rules pooling happen inside the
EPIMIGHT R package (MultipleImputationAnalysis) — this script just
orchestrates the pieces.

Requirements
------------
- Python: pandas, numpy, pyarrow, simace (public sister repo, importable).
- R: the EPIMIGHT package plus arrow, data.table, dplyr, dtplyr, readr,
  jsonlite. The package source ships with simACE at
  ``simACE/epimight/EPIMIGHT/epimight``; this script will install it on
  first run if not already installed.
- The user's R env can be reached either as a plain ``Rscript`` on PATH
  or via ``--conda-env <name>`` (defaults to ``epimight``).

Example:
-------
    python scripts/run_epimight.py \\
        --phenotype results/dev/test/rep1/phenotype.parquet \\
        --output-dir /tmp/epimight_run \\
        --kinds PO,FS,HS \\
        --K 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Vendored from fitace.epimight.create_parquet (kept in sync as of 2026-04)
# ---------------------------------------------------------------------------

BASE_YEAR = 1960  # calendar year offset: born_at_year = BASE_YEAR + generation

KIND_TO_PAIRS = {
    "PO": ["MO", "FO"],
    "FS": ["FS", "MZ"],
    "HS": ["MHS", "PHS"],
    "mHS": ["MHS"],
    "pHS": ["PHS"],
    "1C": ["1C"],
    "Av": ["Av"],
    "1G": ["GP"],
}

# Asymmetric kinds: count older-generation relatives only for the younger
# individual. PO/1G already use (younger, older) ordering; Av loses
# direction in canonical lo/hi form so re-orient by generation.
_UNIDIRECTIONAL_KINDS = {"PO", "1G", "Av"}


def _orient_pairs_by_generation(pair_list, generations):
    oriented = []
    for idx1, idx2 in pair_list:
        if len(idx1) == 0:
            oriented.append((idx1, idx2))
            continue
        swap = generations[idx1] < generations[idx2]
        new_idx1 = np.where(swap, idx2, idx1)
        new_idx2 = np.where(swap, idx1, idx2)
        oriented.append((new_idx1, new_idx2))
    return oriented


def _count_affected_relatives(pair_list, affected, n, unidirectional=False):
    counts = np.zeros(n, dtype=int)
    for idx1, idx2 in pair_list:
        if len(idx1) == 0:
            continue
        np.add.at(counts, idx1, affected[idx2].astype(int))
        if not unidirectional:
            np.add.at(counts, idx2, affected[idx1].astype(int))
    return counts


def _count_total_relatives(pair_list, n, unidirectional=False):
    counts = np.zeros(n, dtype=int)
    for idx1, idx2 in pair_list:
        if len(idx1) == 0:
            continue
        np.add.at(counts, idx1, 1)
        if not unidirectional:
            np.add.at(counts, idx2, 1)
    return counts


def build_epimight_inputs(phenotype_path: Path, output_dir: Path) -> None:
    """phenotype.parquet -> trait{1,2}.epimight_in.parquet + true_parameters.json."""
    from simace.core.pedigree_graph import PedigreeGraph

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(phenotype_path)
    n = len(df)

    print(f"Extracting relationship pairs from {n} rows...")
    all_pairs = PedigreeGraph(df).extract_pairs()

    affected1 = df["affected1"].values.astype(bool)
    affected2 = df["affected2"].values.astype(bool)
    generations = df["generation"].values

    diag1, diag2, nrel = {}, {}, {}
    for kind, pair_types in KIND_TO_PAIRS.items():
        pairs = [(all_pairs[pt][0], all_pairs[pt][1]) for pt in pair_types if pt in all_pairs]
        uni = kind in _UNIDIRECTIONAL_KINDS
        if uni and kind == "Av":
            pairs = _orient_pairs_by_generation(pairs, generations)
        diag1[kind] = _count_affected_relatives(pairs, affected1, n, unidirectional=uni)
        diag2[kind] = _count_affected_relatives(pairs, affected2, n, unidirectional=uni)
        nrel[kind] = _count_total_relatives(pairs, n, unidirectional=uni)

    def build(affected_col: str, time_col: str, diag: dict) -> pd.DataFrame:
        gen = df["generation"].astype("int32")
        out = pd.DataFrame(
            {
                "person_id": df["id"],
                "born_at": gen,
                "born_at_year": (BASE_YEAR + gen).astype(int),
                "dead_at_year": (BASE_YEAR + gen + df["death_age"]).astype(int),
                "failure_status": df[affected_col].astype(int),
                "failure_time": df[time_col].astype(int),
            }
        )
        for kind in KIND_TO_PAIRS:
            out[f"diagnosed_relatives_{kind}"] = diag[kind].astype(int)
        for kind in KIND_TO_PAIRS:
            out[f"n_relatives_{kind}"] = nrel[kind].astype(int)
        return out

    t1 = build("affected1", "t_observed1", diag1)
    t2 = build("affected2", "t_observed2", diag2)

    t1_path = output_dir / "trait1.epimight_in.parquet"
    t2_path = output_dir / "trait2.epimight_in.parquet"
    t1.to_parquet(t1_path, index=False)
    t2.to_parquet(t2_path, index=False)
    print(f"  wrote {t1_path}")
    print(f"  wrote {t2_path}")

    # True genetic parameters from latent components
    needed = {"A1", "C1", "E1", "A2", "C2", "E2"}
    if needed.issubset(df.columns):
        L1 = df["A1"] + df["C1"] + df["E1"]
        L2 = df["A2"] + df["C2"] + df["E2"]
        truth = {
            "h2_trait1_true": float(np.var(df["A1"]) / np.var(L1)),
            "h2_trait2_true": float(np.var(df["A2"]) / np.var(L2)),
            "genetic_correlation_true": float(np.corrcoef(df["A1"], df["A2"])[0, 1]),
            "phenotypic_correlation_true": float(np.corrcoef(L1, L2)[0, 1]),
        }
        truth_path = output_dir / "true_parameters.json"
        truth_path.write_text(json.dumps(truth, indent=2))
        print(f"  wrote {truth_path}")
    else:
        print("  [WARN] A/C/E columns missing — skipping true_parameters.json")


# ---------------------------------------------------------------------------
# Embedded EPIMIGHT R driver (verbatim copy of fitACE/fitace/epimight/guide-yob.R)
# ---------------------------------------------------------------------------

GUIDE_YOB_R = r"""#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(dtplyr)
  library(readr)
  library(epimight)
  library(arrow)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
base_dir          <- if (length(args) >= 1) args[1] else "."
relationship_kind <- if (length(args) >= 2) args[2] else "FS"
seed              <- if (length(args) >= 3) as.integer(args[3]) else 42L
K                 <- if (length(args) >= 4) as.integer(args[4]) else 20L
rubin_level       <- if (length(args) >= 5) args[5] else "meta"

d1_earliest_onset_age <- 1L
d2_earliest_onset_age <- 1L

diag_col <- paste0("diagnosed_relatives_", relationship_kind)
nrel_col <- paste0("n_relatives_", relationship_kind)

d1_raw <- read_parquet(file.path(base_dir, "trait1.epimight_in.parquet")) |> as.data.frame()
d2_raw <- read_parquet(file.path(base_dir, "trait2.epimight_in.parquet")) |> as.data.frame()

if (!(diag_col %in% names(d1_raw))) {
  stop("Column '", diag_col, "' not found in trait1.epimight_in.parquet. ",
       "Available diagnosed_relatives columns: ",
       paste(grep("^diagnosed_relatives_", names(d1_raw), value = TRUE), collapse = ", "))
}

d1_tte <- d1_raw |>
  select(person_id, born_at_year, failure_status, failure_time,
         diagnosed_relatives = !!sym(diag_col),
         n_relatives = !!sym(nrel_col))
d2_tte <- d2_raw |>
  select(person_id, born_at_year, failure_status, failure_time,
         diagnosed_relatives = !!sym(diag_col),
         n_relatives = !!sym(nrel_col))

message("Disorder 1 survival data: ", nrow(d1_tte), " rows")
message("Disorder 2 survival data: ", nrow(d2_tte), " rows")
message("Relationship kind: ", relationship_kind, " (column: ", diag_col, ")")

c1_tte <- d1_tte |>
  inner_join(d2_tte, by = join_by(person_id)) |>
  rename(
    born_at_year           = born_at_year.x,
    d1_failure_status      = failure_status.x,
    d1_failure_time        = failure_time.x,
    d1_diagnosed_relatives = diagnosed_relatives.x,
    d1_n_relatives         = n_relatives.x,
    d2_failure_status      = failure_status.y,
    d2_failure_time        = failure_time.y,
    d2_diagnosed_relatives = diagnosed_relatives.y,
    d2_n_relatives         = n_relatives.y
  ) |>
  select(person_id, born_at_year, starts_with("d1_"), starts_with("d2_")) |>
  as.data.table()

message("c1_tte: ", nrow(c1_tte), " rows")
message("Running MI analysis: K=", K, ", seed=", seed, ", rubin_level=", rubin_level)

mi <- MultipleImputationAnalysis$new(
  c1_tte            = c1_tte,
  relationship_kind = relationship_kind,
  K                 = K,
  seed              = seed,
  d1_earliest_onset = d1_earliest_onset_age,
  d2_earliest_onset = d2_earliest_onset_age
)

results <- mi$run(rubin_level = rubin_level)

rk <- relationship_kind
tsv_dir <- file.path(base_dir, "tsv")
dir.create(tsv_dir, showWarnings = FALSE, recursive = TRUE)
tsv <- function(name, df) {
  path <- file.path(tsv_dir, paste0(name, "_", rk, ".tsv"))
  write.table(df, path, sep = "\t", row.names = FALSE, quote = FALSE)
  message("  ", path)
  path
}

message("Writing TSV files:")
tsv("h2_d1_meta", results$h2_d1$rubin_meta)
tsv("h2_d2_meta", results$h2_d2$rubin_meta)
tsv("gc_meta",    results$gc$rubin_meta)
tsv("h2_d1_resamples", results$h2_d1$resample_meta)
tsv("h2_d2_resamples", results$h2_d2$resample_meta)
tsv("gc_resamples",    results$gc$resample_meta)

if (!is.null(results$h2_d1$per_year)) {
  tsv("h2_d1_per_year", results$h2_d1$per_year)
  tsv("h2_d2_per_year", results$h2_d2$per_year)
  tsv("gc_per_year",    results$gc$per_year)
}

message("Done.")
"""


# ---------------------------------------------------------------------------
# R invocation
# ---------------------------------------------------------------------------


def _r_command(conda_env: str | None) -> list[str]:
    """Build the prefix used to invoke Rscript (with or without conda)."""
    if conda_env:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "Rscript"]
    rscript = shutil.which("Rscript")
    if rscript is None:
        sys.exit("Rscript not found on PATH. Pass --conda-env <name> or install R.")
    return [rscript]


def ensure_epimight_installed(r_cmd: list[str], pkg_source: Path) -> None:
    """Install the EPIMIGHT R package from source if not already present."""
    if not pkg_source.exists():
        sys.exit(
            f"EPIMIGHT R package source not found at {pkg_source}.\n"
            "Pass --epimight-pkg <path> pointing to the EPIMIGHT/epimight directory."
        )
    check = (
        f"if (!requireNamespace('epimight', quietly=TRUE)) install.packages('{pkg_source}', repos=NULL, type='source')"
    )
    print(f"Ensuring epimight R package is installed (source: {pkg_source})...")
    subprocess.run([*r_cmd, "-e", check], check=True)


def run_kind(
    r_cmd: list[str], r_script: Path, output_dir: Path, kind: str, seed: int, K: int, rubin_level: str
) -> None:
    """Run the EPIMIGHT R driver for a single relationship kind."""
    print(f"\n=== EPIMIGHT: kind={kind} K={K} seed={seed} rubin_level={rubin_level} ===")
    subprocess.run(
        [*r_cmd, str(r_script), str(output_dir), kind, str(seed), str(K), rubin_level],
        check=True,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_summary(output_dir: Path, kinds: list[str]) -> Path:
    """Concatenate per-kind meta TSVs into one tidy summary.tsv."""
    tsv_dir = output_dir / "tsv"
    rows = []
    for kind in kinds:
        for param, fname in [
            ("h2_d1", f"h2_d1_meta_{kind}.tsv"),
            ("h2_d2", f"h2_d2_meta_{kind}.tsv"),
            ("gc", f"gc_meta_{kind}.tsv"),
        ]:
            path = tsv_dir / fname
            if not path.exists():
                print(f"  [WARN] missing {path} — skipping")
                continue
            df = pd.read_csv(path, sep="\t")
            row = {"kind": kind, "parameter": param}
            row.update(df.iloc[0].to_dict())
            rows.append(row)

    summary = pd.DataFrame(rows)

    # Attach truth if available
    truth_path = output_dir / "true_parameters.json"
    if truth_path.exists():
        truth = json.loads(truth_path.read_text())
        truth_map = {
            "h2_d1": truth.get("h2_trait1_true"),
            "h2_d2": truth.get("h2_trait2_true"),
            "gc": truth.get("genetic_correlation_true"),
        }
        summary["true"] = summary["parameter"].map(truth_map)

    out_path = output_dir / "summary.tsv"
    summary.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote aggregated summary -> {out_path}")
    print(summary.to_string(index=False))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phenotype", required=True, type=Path, help="Path to phenotype.parquet from a simACE rep.")
    p.add_argument("--output-dir", required=True, type=Path, help="Output directory (created if missing).")
    p.add_argument(
        "--kinds",
        default="PO,FS,HS",
        help=f"Comma-separated relationship kinds. Choices: {','.join(KIND_TO_PAIRS)}. Default: PO,FS,HS",
    )
    p.add_argument("--K", type=int, default=20, help="Number of resamples for multiple imputation (default: 20).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rubin-level", choices=["meta", "per_year"], default="meta", help="Rubin pooling granularity (default: meta)."
    )
    p.add_argument(
        "--conda-env", default="epimight", help="Conda env containing R+epimight. Set to '' to use Rscript on PATH."
    )
    p.add_argument(
        "--epimight-pkg",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "epimight" / "EPIMIGHT" / "epimight",
        help="Path to the EPIMIGHT R package source (for first-run install).",
    )
    return p.parse_args()


def main() -> None:
    """Run create_parquet -> EPIMIGHT R driver per kind -> aggregate summary."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in KIND_TO_PAIRS]
    if unknown:
        sys.exit(f"Unknown kinds: {unknown}. Choices: {list(KIND_TO_PAIRS)}")

    # 1. Build EPIMIGHT input parquets + truth from phenotype.parquet
    build_epimight_inputs(args.phenotype.resolve(), output_dir)

    # 2. Run EPIMIGHT R driver per kind
    r_cmd = _r_command(args.conda_env or None)
    ensure_epimight_installed(r_cmd, args.epimight_pkg.resolve())

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as f:
        f.write(GUIDE_YOB_R)
        r_script = Path(f.name)

    try:
        for kind in kinds:
            run_kind(r_cmd, r_script, output_dir, kind, args.seed, args.K, args.rubin_level)
    finally:
        os.unlink(r_script)

    # 3. Aggregate per-kind meta TSVs into one summary
    aggregate_summary(output_dir, kinds)


if __name__ == "__main__":
    main()
