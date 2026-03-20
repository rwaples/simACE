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
base_dir <- if (length(args) >= 1) args[1] else "."
relationship_kind <- if (length(args) >= 2) args[2] else "FS"

## ──────────────────────────────────────────────────────────────────────────────
## Explicit analysis parameters
## ──────────────────────────────────────────────────────────────────────────────
d1_earliest_onset_age <- 1L
d2_earliest_onset_age <- 1L

## ──────────────────────────────────────────────────────────────────────────────
## Input — select the diagnosed_relatives column for the chosen relationship kind
## ──────────────────────────────────────────────────────────────────────────────
diag_col <- paste0("diagnosed_relatives_", relationship_kind)

d1_raw <- read_parquet(file.path(base_dir, "NDD.parquet")) |> as.data.frame()
d2_raw <- read_parquet(file.path(base_dir, "NDG.parquet")) |> as.data.frame()

if (!(diag_col %in% names(d1_raw))) {
  stop("Column '", diag_col, "' not found in NDD.parquet. ",
       "Available diagnosed_relatives columns: ",
       paste(grep("^diagnosed_relatives_", names(d1_raw), value = TRUE), collapse = ", "))
}

d1_tte <- d1_raw |>
  select(person_id, born_at_year, failure_status, failure_time,
         diagnosed_relatives = !!sym(diag_col))
d2_tte <- d2_raw |>
  select(person_id, born_at_year, failure_status, failure_time,
         diagnosed_relatives = !!sym(diag_col))

message("Disorder 1 survival data: ", nrow(d1_tte), " rows")
message("Disorder 2 survival data: ", nrow(d2_tte), " rows")
message("Relationship kind: ", relationship_kind, " (column: ", diag_col, ")")

## Primary join to construct the cohorts
##   c1 = base cohort
##   c2 = relatives diagnosed with disorder 1
##   c3 = relatives diagnosed with disorder 2
c1_tte <- d1_tte |>
  inner_join(d2_tte, by = join_by(person_id)) |>
  rename(
    born_at_year           = born_at_year.x,
    d1_failure_status      = failure_status.x,
    d1_failure_time        = failure_time.x,
    d1_diagnosed_relatives = diagnosed_relatives.x,
    d2_failure_status      = failure_status.y,
    d2_failure_time        = failure_time.y,
    d2_diagnosed_relatives = diagnosed_relatives.y
  ) |>
  select(person_id, born_at_year, starts_with("d1_"), starts_with("d2_"))

c2_tte <- c1_tte |> filter(d1_diagnosed_relatives > 0)
c3_tte <- c1_tte |> filter(d2_diagnosed_relatives > 0)

if (nrow(c2_tte) == 0) stop("c2_tte is empty: no individuals have diagnosed relatives for disorder 1")
if (nrow(c3_tte) == 0) stop("c3_tte is empty: no individuals have diagnosed relatives for disorder 2")

message("c1_tte: ", nrow(c1_tte), " | c2_tte: ", nrow(c2_tte), " | c3_tte: ", nrow(c3_tte))

## ──────────────────────────────────────────────────────────────────────────────
## Helper: stratified CIF by birth year
## ──────────────────────────────────────────────────────────────────────────────
cif_analysis <- CumulativeIncidenceAnalysis$new()

run_cif_stratified <- function(disorder, tte_df, earliest_onset) {
  status_col <- paste0(disorder, "_failure_status")
  time_col   <- paste0(disorder, "_failure_time")

  tte <- tte_df |>
    rename(
      failure_status = !!as.name(status_col),
      failure_time   = !!as.name(time_col)
    ) |>
    as.data.table()

  cif_analysis$run(
    tte = tte,
    earliest_onset = earliest_onset,
    group_columns = list("born_at_year")
  )
}

## Run CIF for the 5 required sets
re_d1_c1 <- run_cif_stratified("d1", c1_tte, d1_earliest_onset_age)
re_d1_c2 <- run_cif_stratified("d1", c2_tte, d1_earliest_onset_age)
re_d1_c3 <- run_cif_stratified("d1", c3_tte, d1_earliest_onset_age)
re_d2_c1 <- run_cif_stratified("d2", c1_tte, d2_earliest_onset_age)
re_d2_c3 <- run_cif_stratified("d2", c3_tte, d2_earliest_onset_age)

## ──────────────────────────────────────────────────────────────────────────────
## Heritability by cohort (d1: c1 vs c2; d2: c1 vs c3)
## ──────────────────────────────────────────────────────────────────────────────
h2_analysis <- HeritabilityAnalysis$new()

run_h2_stratified <- function(c1_estimates, c2_estimates) {
  combined_estimates <- c1_estimates |>
    inner_join(c2_estimates, by = join_by(time, born_at_year)) |>
    rename(
      cohort1_estimates = estimate.x,
      cohort1_cases     = cases.x,
      cohort2_estimates = estimate.y,
      cohort2_cases     = cases.y
    ) |>
    select(time, born_at_year, starts_with("cohort"))

  h2_analysis$run(
    relationship_kind = relationship_kind,
    estimates         = combined_estimates
  )
}

h2_d1 <- run_h2_stratified(re_d1_c1, re_d1_c2)
h2_d2 <- run_h2_stratified(re_d2_c1, re_d2_c3)

## h2 tmax per year (filter to max follow-up before meta-analysis)
h2_d1_tmax <- h2_d1 |>
  group_by(born_at_year) |>
  filter(time == max(time)) |>
  ungroup() |>
  select(born_at_year, time, h2, se, l95, u95) |>
  arrange(born_at_year) |>
  as.data.table()

h2_d2_tmax <- h2_d2 |>
  group_by(born_at_year) |>
  filter(time == max(time)) |>
  ungroup() |>
  select(born_at_year, time, h2, se, l95, u95) |>
  arrange(born_at_year) |>
  as.data.table()

## h2 meta-analysis (tmax only — consistent with GC meta)
h2_d1_meta <- h2_analysis$run_meta(h2_d1_tmax)
h2_d2_meta <- h2_analysis$run_meta(h2_d2_tmax)

## ──────────────────────────────────────────────────────────────────────────────
## Assemble consistent data for GC
##  - Option A: full grid (all common times across all series)
##  - Option B: for each birth year, use only the largest common follow-up time
## ──────────────────────────────────────────────────────────────────────────────

## RE (3 estimate columns + cases)
re_combined <- re_d1_c1 |>
  select(time, born_at_year,
         re_d1_c1_estimates = estimate,
         re_d1_c1_cases     = cases) |>
  inner_join(
    re_d1_c3 |>
      select(time, born_at_year,
             re_d1_c3_estimates = estimate,
             re_d1_c3_cases     = cases),
    by = join_by(time, born_at_year)
  ) |>
  inner_join(
    re_d2_c1 |>
      select(time, born_at_year,
             re_d2_c1_estimates = estimate,
             re_d2_c1_cases     = cases),
    by = join_by(time, born_at_year)
  )

## H2 (two columns: h2_d1, h2_d2)
h2_combined <- h2_d1 |>
  select(time, born_at_year, h2_d1 = h2) |>
  inner_join(h2_d2 |>
               select(time, born_at_year, h2_d2 = h2),
             by = join_by(time, born_at_year))

## Option A: full grid (inner_join ensures common times)
combined_full <- re_combined |>
  inner_join(h2_combined, by = join_by(time, born_at_year)) |>
  arrange(born_at_year, time) |>
  as.data.table()

## Option B: select the maximum common time per birth year
combined_tmax_by_year <- combined_full |>
  group_by(born_at_year) |>
  filter(time == max(time)) |>
  ungroup() |>
  arrange(born_at_year, time) |>
  as.data.table()

## ──────────────────────────────────────────────────────────────────────────────
## GC: A) full grid and B) tmax per year
## ──────────────────────────────────────────────────────────────────────────────
gc_analysis <- GeneticCorrelationAnalysis$new()

## A) full grid
gc_d1_d2_all <- gc_analysis$run(
  relationship_kind = relationship_kind,
  estimates         = combined_full
)

## B) tmax per year
gc_d1_d2_tmax <- gc_analysis$run(
  relationship_kind = relationship_kind,
  estimates         = combined_tmax_by_year
)

## Meta-analysis (tmax version)
meta_tmax <- gc_analysis$run_meta(gc_d1_d2_tmax)

## ──────────────────────────────────────────────────────────────────────────────
## Extract observed values (tmax per year)
## ──────────────────────────────────────────────────────────────────────────────

gc_tmax <- gc_d1_d2_tmax |>
  select(born_at_year, time, rhh, rhog, se, l95, u95, h2_comb, h2_l95, h2_u95) |>
  arrange(born_at_year)

## ──────────────────────────────────────────────────────────────────────────────
## Write TSV files for large data tables
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
tsv("cif_d1_c1", re_d1_c1 |> arrange(born_at_year, time))
tsv("cif_d1_c2", re_d1_c2 |> arrange(born_at_year, time))
tsv("cif_d1_c3", re_d1_c3 |> arrange(born_at_year, time))
tsv("cif_d2_c1", re_d2_c1 |> arrange(born_at_year, time))
tsv("cif_d2_c3", re_d2_c3 |> arrange(born_at_year, time))
tsv("h2_d1", h2_d1 |> arrange(born_at_year, time))
tsv("h2_d2", h2_d2 |> arrange(born_at_year, time))
tsv("gc_full", gc_d1_d2_all |> arrange(born_at_year, time))

meta_to_df <- function(m) {
  data.frame(
    fixed_meta = m$fixed_meta, fixed_se = m$fixed_se,
    fixed_l95 = m$fixed_l95, fixed_u95 = m$fixed_u95,
    rand_meta = m$rand_meta, rand_se = m$rand_se,
    rand_l95 = m$rand_l95, rand_u95 = m$rand_u95
  )
}
tsv("h2_d1_meta", meta_to_df(h2_d1_meta))
tsv("h2_d2_meta", meta_to_df(h2_d2_meta))
tsv("gc_meta", meta_to_df(meta_tmax))

## ──────────────────────────────────────────────────────────────────────────────
## Write summary markdown report
## ──────────────────────────────────────────────────────────────────────────────

report_path <- file.path(base_dir, paste0("results_", rk, ".md"))
message("Writing report to: ", report_path)

md <- character()
w <- function(...) md <<- c(md, paste0(...))

w("# EPIMIGHT Analysis Results")
w("")
w("**Relationship kind:** ", rk)
w("")

## ── Cohort sizes ──
w("## Cohort Sizes")
w("")
w("| Cohort | Description | N |")
w("|--------|-------------|---|")
w(sprintf("| c1 | Base (all individuals) | %d |", nrow(c1_tte)))
w(sprintf("| c2 | Relatives diagnosed with d1 | %d |", nrow(c2_tte)))
w(sprintf("| c3 | Relatives diagnosed with d2 | %d |", nrow(c3_tte)))
w("")

## ── h2 tmax per year ──
w("## Heritability (tmax per year)")
w("")
w("### Trait 1 (d1)")
w("")
w("| born_at_year | time | h2 | se | l95 | u95 |")
w("|-------------|------|-----|-----|------|------|")
for (i in seq_len(nrow(h2_d1_tmax))) {
  r <- h2_d1_tmax[i, ]
  w(sprintf("| %d | %d | %.4f | %.4f | %.4f | %.4f |",
            r$born_at_year, r$time, r$h2, r$se, r$l95, r$u95))
}
w("")
w("### Trait 2 (d2)")
w("")
w("| born_at_year | time | h2 | se | l95 | u95 |")
w("|-------------|------|-----|-----|------|------|")
for (i in seq_len(nrow(h2_d2_tmax))) {
  r <- h2_d2_tmax[i, ]
  w(sprintf("| %d | %d | %.4f | %.4f | %.4f | %.4f |",
            r$born_at_year, r$time, r$h2, r$se, r$l95, r$u95))
}
w("")

## ── h2 meta-analysis ──
w("## Heritability Meta-analysis")
w("")
w("### Trait 1 (d1)")
w("")
w(sprintf("| | estimate | se | l95 | u95 |"))
w("|--|----------|-----|------|------|")
w(sprintf("| Fixed  | %.4f | %.4f | %.4f | %.4f |",
          h2_d1_meta$fixed_meta, h2_d1_meta$fixed_se, h2_d1_meta$fixed_l95, h2_d1_meta$fixed_u95))
w(sprintf("| Random | %.4f | %.4f | %.4f | %.4f |",
          h2_d1_meta$rand_meta, h2_d1_meta$rand_se, h2_d1_meta$rand_l95, h2_d1_meta$rand_u95))
w("")
w("### Trait 2 (d2)")
w("")
w(sprintf("| | estimate | se | l95 | u95 |"))
w("|--|----------|-----|------|------|")
w(sprintf("| Fixed  | %.4f | %.4f | %.4f | %.4f |",
          h2_d2_meta$fixed_meta, h2_d2_meta$fixed_se, h2_d2_meta$fixed_l95, h2_d2_meta$fixed_u95))
w(sprintf("| Random | %.4f | %.4f | %.4f | %.4f |",
          h2_d2_meta$rand_meta, h2_d2_meta$rand_se, h2_d2_meta$rand_l95, h2_d2_meta$rand_u95))
w("")

## ── GC tmax per year ──
w("## Genetic Correlation (tmax per year)")
w("")
w("| born_at_year | time | rhh | rhog | se | l95 | u95 | h2_comb |")
w("|-------------|------|------|------|-----|------|------|---------|")
for (i in seq_len(nrow(gc_tmax))) {
  r <- gc_tmax[i, ]
  w(sprintf("| %d | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |",
            r$born_at_year, r$time, r$rhh, r$rhog, r$se, r$l95, r$u95, r$h2_comb))
}
w("")

## ── GC meta-analysis ──
w("## Genetic Correlation Meta-analysis (tmax per year)")
w("")
w("| | estimate | se | l95 | u95 |")
w("|--|----------|-----|------|------|")
w(sprintf("| Fixed  | %.4f | %.4f | %.4f | %.4f |",
          meta_tmax$fixed_meta, meta_tmax$fixed_se, meta_tmax$fixed_l95, meta_tmax$fixed_u95))
w(sprintf("| Random | %.4f | %.4f | %.4f | %.4f |",
          meta_tmax$rand_meta, meta_tmax$rand_se, meta_tmax$rand_l95, meta_tmax$rand_u95))
w("")

## ── True vs Observed comparison ──
truth_path <- file.path(base_dir, "true_parameters.json")
if (file.exists(truth_path)) {
  truth <- fromJSON(truth_path)

  obs_h2_d1 <- median(h2_d1_tmax$h2, na.rm = TRUE)
  obs_h2_d2 <- median(h2_d2_tmax$h2, na.rm = TRUE)
  obs_gc    <- median(gc_tmax$rhog, na.rm = TRUE)

  rel_err <- function(obs, true) {
    if (abs(true) < 1e-9) return(NA_real_)
    100 * (obs - true) / true
  }

  w("## True vs Observed Comparison")
  w("")
  w("| Parameter | True | Observed (median) | Relative Error (%) |")
  w("|-----------|------|-------------------|-------------------|")
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
w(sprintf("| `tsv/cif_d1_c1_%s.tsv` | CIF: disorder 1 in base cohort |", rk))
w(sprintf("| `tsv/cif_d1_c2_%s.tsv` | CIF: disorder 1 in relatives of d1-affected |", rk))
w(sprintf("| `tsv/cif_d1_c3_%s.tsv` | CIF: disorder 1 in relatives of d2-affected |", rk))
w(sprintf("| `tsv/cif_d2_c1_%s.tsv` | CIF: disorder 2 in base cohort |", rk))
w(sprintf("| `tsv/cif_d2_c3_%s.tsv` | CIF: disorder 2 in relatives of d2-affected |", rk))
w(sprintf("| `tsv/h2_d1_%s.tsv` | Heritability trait 1 (all time points) |", rk))
w(sprintf("| `tsv/h2_d2_%s.tsv` | Heritability trait 2 (all time points) |", rk))
w(sprintf("| `tsv/gc_full_%s.tsv` | Genetic correlation full grid (all time x year) |", rk))
w("")

writeLines(md, report_path)
message("Done. Report written to: ", report_path)
