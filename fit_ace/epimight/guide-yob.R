#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(dtplyr)
  library(readr)
  library(epimight)
  library(arrow)
  library(jsonlite)
})

## ──────────────────────────────────────────────────────────────────────────────
## Input directory and relationship kind from command line
## ──────────────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
base_dir          <- if (length(args) >= 1) args[1] else "."
relationship_kind <- if (length(args) >= 2) args[2] else "FS"
seed              <- if (length(args) >= 3) as.integer(args[3]) else 42L
K                 <- if (length(args) >= 4) as.integer(args[4]) else 20L
rubin_level       <- if (length(args) >= 5) args[5] else "meta"

## ──────────────────────────────────────────────────────────────────────────────
## Explicit analysis parameters
## ──────────────────────────────────────────────────────────────────────────────
d1_earliest_onset_age <- 1L
d2_earliest_onset_age <- 1L

## ──────────────────────────────────────────────────────────────────────────────
## Input — select the diagnosed_relatives column for the chosen relationship kind
## ──────────────────────────────────────────────────────────────────────────────
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

## Primary join to construct c1_tte (pre-joined input for MI class)
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

## ──────────────────────────────────────────────────────────────────────────────
## Run Multiple Imputation Analysis
## ──────────────────────────────────────────────────────────────────────────────
mi <- MultipleImputationAnalysis$new(
  c1_tte            = c1_tte,
  relationship_kind = relationship_kind,
  K                 = K,
  seed              = seed,
  d1_earliest_onset = d1_earliest_onset_age,
  d2_earliest_onset = d2_earliest_onset_age
)

results <- mi$run(rubin_level = rubin_level)

## ──────────────────────────────────────────────────────────────────────────────
## Write TSV files
## ──────────────────────────────────────────────────────────────────────────────

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

## Rubin-combined meta (1 row each)
tsv("h2_d1_meta", results$h2_d1$rubin_meta)
tsv("h2_d2_meta", results$h2_d2$rubin_meta)
tsv("gc_meta",    results$gc$rubin_meta)

## Per-resample meta (K rows each, for diagnostics)
tsv("h2_d1_resamples", results$h2_d1$resample_meta)
tsv("h2_d2_resamples", results$h2_d2$resample_meta)
tsv("gc_resamples",    results$gc$resample_meta)

## ──────────────────────────────────────────────────────────────────────────────
## Write summary markdown report
## ──────────────────────────────────────────────────────────────────────────────

report_path <- file.path(base_dir, paste0("results_", rk, ".md"))
message("Writing report to: ", report_path)

md <- character()
w <- function(...) md <<- c(md, paste0(...))

h2_d1_meta <- results$h2_d1$rubin_meta
h2_d2_meta <- results$h2_d2$rubin_meta
gc_meta    <- results$gc$rubin_meta

w("# EPIMIGHT Analysis Results (Multiple Imputation)")
w("")
w("**Relationship kind:** ", rk)
w("**Resamples (K):** ", K)
w("**Seed:** ", seed)
w("")

## ── Heritability ──
w("## Heritability (Rubin-combined)")
w("")
w("| Trait | estimate | se | 95% CI | B/T | K |")
w("|-------|----------|----|--------|-----|---|")
w(sprintf("| d1 | %.4f | %.4f | [%.4f, %.4f] | %.3f | %d |",
          h2_d1_meta$fixed_meta, h2_d1_meta$fixed_se,
          h2_d1_meta$fixed_l95, h2_d1_meta$fixed_u95,
          ifelse(is.na(h2_d1_meta$b_over_t), 0, h2_d1_meta$b_over_t),
          h2_d1_meta$k_resamples))
w(sprintf("| d2 | %.4f | %.4f | [%.4f, %.4f] | %.3f | %d |",
          h2_d2_meta$fixed_meta, h2_d2_meta$fixed_se,
          h2_d2_meta$fixed_l95, h2_d2_meta$fixed_u95,
          ifelse(is.na(h2_d2_meta$b_over_t), 0, h2_d2_meta$b_over_t),
          h2_d2_meta$k_resamples))
w("")

## ── Genetic Correlation ──
w("## Genetic Correlation (Rubin-combined)")
w("")
w("| estimate | se | 95% CI | B/T | K |")
w("|----------|----|--------|-----|---|")
w(sprintf("| %.4f | %.4f | [%.4f, %.4f] | %.3f | %d |",
          gc_meta$fixed_meta, gc_meta$fixed_se,
          gc_meta$fixed_l95, gc_meta$fixed_u95,
          ifelse(is.na(gc_meta$b_over_t), 0, gc_meta$b_over_t),
          gc_meta$k_resamples))
w("")

## ── Rubin Diagnostics ──
if (K > 1L) {
  w("## Rubin Diagnostics")
  w("")
  w("B/T = fraction of total variance due to relative selection randomness.")
  w("Large B/T → single-pair approach loses substantial information for this kind.")
  w("")
  w("| Parameter | Within (W) | Between (B) | Total (T) | B/T |")
  w("|-----------|-----------|------------|----------|-----|")
  w(sprintf("| h2 d1 | %.6f | %.6f | %.6f | %.3f |",
            h2_d1_meta$within_var, h2_d1_meta$between_var,
            h2_d1_meta$total_var, h2_d1_meta$b_over_t))
  w(sprintf("| h2 d2 | %.6f | %.6f | %.6f | %.3f |",
            h2_d2_meta$within_var, h2_d2_meta$between_var,
            h2_d2_meta$total_var, h2_d2_meta$b_over_t))
  w(sprintf("| GC    | %.6f | %.6f | %.6f | %.3f |",
            gc_meta$within_var, gc_meta$between_var,
            gc_meta$total_var, gc_meta$b_over_t))
  w("")
}

## ── True vs Observed comparison ──
truth_path <- file.path(base_dir, "true_parameters.json")
if (file.exists(truth_path)) {
  truth <- fromJSON(truth_path)

  obs_h2_d1 <- h2_d1_meta$fixed_meta
  obs_h2_d2 <- h2_d2_meta$fixed_meta
  obs_gc    <- gc_meta$fixed_meta

  rel_err <- function(obs, true) {
    if (abs(true) < 1e-9) return(NA_real_)
    100 * (obs - true) / true
  }

  w("## True vs Observed Comparison")
  w("")
  w("| Parameter | True | Observed | Relative Error (%) |")
  w("|-----------|------|----------|-------------------|")
  w(sprintf("| h2 trait 1 | %.4f | %.4f | %+.1f%% |",
            truth$h2_trait1_true, obs_h2_d1, rel_err(obs_h2_d1, truth$h2_trait1_true)))
  w(sprintf("| h2 trait 2 | %.4f | %.4f | %+.1f%% |",
            truth$h2_trait2_true, obs_h2_d2, rel_err(obs_h2_d2, truth$h2_trait2_true)))
  w(sprintf("| Genetic correlation | %.4f | %.4f | %+.1f%% |",
            truth$genetic_correlation_true, obs_gc, rel_err(obs_gc, truth$genetic_correlation_true)))
  w(sprintf("| Phenotypic correlation | %.4f | - | - |",
            truth$phenotypic_correlation_true))
  w("")
} else {
  w("## True Parameters")
  w("")
  w("*true_parameters.json not found — skipping comparison.*")
  w("")
}

## ── Data files index ──
w("## Data Files")
w("")
w("| File | Description |")
w("|------|-------------|")
w(sprintf("| `tsv/h2_d1_meta_%s.tsv` | Rubin-combined h2 trait 1 |", rk))
w(sprintf("| `tsv/h2_d2_meta_%s.tsv` | Rubin-combined h2 trait 2 |", rk))
w(sprintf("| `tsv/gc_meta_%s.tsv` | Rubin-combined genetic correlation |", rk))
w(sprintf("| `tsv/h2_d1_resamples_%s.tsv` | Per-resample h2 trait 1 (K rows) |", rk))
w(sprintf("| `tsv/h2_d2_resamples_%s.tsv` | Per-resample h2 trait 2 (K rows) |", rk))
w(sprintf("| `tsv/gc_resamples_%s.tsv` | Per-resample GC (K rows) |", rk))
w("")

writeLines(md, report_path)
message("Done. Report written to: ", report_path)
