# 9a gate notes (2026-09-09)

Run: `EPIMIGHT_CONDA_ENV=epimight-master pixi run --frozen python tools/pg08_release_gate.py run --stage 9a --routing source --slow`.
Every consumer unit's `routing` step resolved `pedigree_graph` under `external/pedigree-graph/`.

## fitACE_epimight `pytest-slow`: 5 failures, pre-existing, not 0.8

`test_atlas_e2e_emitter_to_html`, `test_cli_end_to_end`,
`test_r_driver_schema_renames_fixed_cols`, `test_emitter_to_driver_roundtrip`,
`test_onset_bounds_constrain_cif_time_axis` fail inside R
(`pipe$run failed (After joining h2 results for both disorders no data was left)`)
on the 60-person `medium_pedigree` fixture with `draws=3` against epimight R
1.0.1 in the `epimight-master` conda env.

Evidence that this is independent of pedigree-graph 0.8: the emitter's only
pedigree-graph call is `relationship_pairs(max_degree=3)`
(`fitace_epimight/create_input.py:84`), and on the same fixture every one of the
23 category pair sets is identical between the locked 0.7.1 wheel and the 0.8
source checkout (`MO` 40, `FO` 40, `FS` 20, `GP` 80, `Av` 40, `1C` 20, all
others 0; script `pairs_cmp.py` in the session scratchpad). The R driver
therefore sees the same input it saw before the migration. Slice 8 ran the
epimight suite with `slow` deselected, so this suite had not been exercised on
this box since the R package moved to 1.0.1. The two live-R tests in
`test_fractional.py` pass.

The slow suite could not run at all before this gate because the conda env
name `epimight` was hard-coded while the box has `epimight-master`; the
`EPIMIGHT_CONDA_ENV` override (uncommitted in fitACE_epimight) is what let it run.

## ace_iter_reml

`ctest` is not installed anywhere on the box; the unit runs the six test
executables directly. `pedsum` `cli-smoke` reads a TSV written from the smoke
pedigree parquet.

## 30k integration

Not rerun in 9a. The slice 8f `random_30k` library snapshot
(`../../migrated/library.json`, written 2026-09-09 03:36 with
`pg08_migration_diff.py snapshot --source`) was taken against pedigree-graph
`c8c963a`, which is still the `v0.8` head at gate time, so its comparison with
the 0.7.1 baseline (`../../report.md`, `library.random_30k.*` rows) is the
9a 30k evidence: pair sets, kinship values, and candidate-matrix support at 30k,
with the approximate matrix's DP-pass cost (75.7 s, 6938 MiB) recorded there.

## 300k release cases (recorded, not gated)

`random_300k` from `tests/parity/pedigrees.RELEASE_FIXTURES` (seed 300000,
15000 founders, 8 generations, 36000 per generation), pedigree-graph
`c8c963a`, one thread, one cell per child process. Reports:
`external/pedigree-graph/benchmarks/reports/release_300k{,_inbreeding,_counts}.json`.

| operation | wall | peak RSS | note |
|---|---:|---:|---|
| `relationship_pairs(max_degree=3)` | 2.57 s | 846 MiB | 5.8M pairs (MO 277854, FO 278044, GP 925918, GGP 1502119, HAv 2086186, MHS 302281, PHS 300508, MZ 5506, 1C 16480) |
| `connected_component_ids()` | 0.07 s | 212 MiB | 142 components |
| `inbreeding()` | 1.71 s | 211 MiB | mean F 3.7e-5, max F 0.127 |
| `estimate_relationship_counts(max_degree=3)` | 0.29 s | 327 MiB | |
| `pair_kinship` over the degree-3 pairs | **timeout at 3600 s** | 6.9 GiB | unverified at 300k |
| complete / approximate kinship matrices | not attempted | | unverified at 300k; the DP pass costs 6.9 GiB at 30k |

fitACE `write_pedigree(min_kinship=0.001)` on a generated 101k-row pedigree
(seed 100000, `gate/9a/write_pedigree_100k.json`): 293.9 s, peak RSS
21.97 GiB, 2.7 GB sparse GRM. Runnable on the 30 GiB box; the RSS is the
approximate matrix's complete DP pass and is the practical ceiling here.
