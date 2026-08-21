# simACE pixi adoption (replacing conda for simACE work; retiring uv)

Status: **implemented and released.** Landed as `78c53e0` (ADR renumbered to
**0016** after polars took 0015), merged dev→master, released in `v2026.08`.
Remaining item: the perf gate's same-code conda-vs-pixi A/B never ran before
the release (overtaken by the polars migration + release); run 2026-08-21 with
Snakemake-native guardrails (`--cores 4 --resources mem_mb=20000`, 21-job DAG
per run: `stats.done` + `plots/atlas.html`, not `scenario.done` which drags in
the folder-wide report gather). Results below.

## Perf gate results (2026-08-21, baseline100K, 3× per env alternating)

**Scientific/compute stages: PASS.** Per-(stage, rep) medians — analyze wall
within ±1.4%, simulate/phenotype/ascertainment (all ≤1.3s) scatter both
directions within noise; cpu_time parity throughout. `simulate` max_rss is
**40% lower** under pixi (162–184 vs 290–294 MB). `assemble_atlas` is faster
and leaner under pixi. Total wall medians: conda 45s, pixi 48s.

**plot_phenotype: breaches the 5% wall rule — flagged, cause characterized.**
+14.6% concurrent (22.35 vs 19.51s median), +16% serial (23.36 vs 20.10s),
reproducible in every run. But **cpu_time is identical** (15.6–15.7s both
envs), io_in/io_out equal, and max_rss is 40% *lower* under pixi (507 vs
819 MB). The stage does identical work and waits ~3.3s longer — reduced
effective parallelism in some phase (mean_load 0.67 vs 0.78). Ruled out by
direct measurement: package builds (identical conda-forge builds), BLAS (same
openblas), font caches (text-render microbench equal), thread-pool defaults
(numba/polars/pyarrow all 12 in both), interpreter startup (0.02 vs 0.08s),
allocator packages (none in either env). Root cause found by the py-spy
follow-up — see the resolved disposition below.

Scope note: the breach is confined to the presentation layer (atlas plotting,
+3.3s absolute per scenario); no scientific stage regressed and memory
improved. Raw data: scratchpad bench/ TSV archives (session-local).

**Disposition (2026-08-21): gate CLOSED — resolved as no regression (#11).**
The py-spy follow-up found the breach was a measurement artifact: `pixi run`
exports `MPLBACKEND=Agg` (pixi.toml `[activation.env]`), while the conda side
inherited the desktop session (`DISPLAY` set, `MPLBACKEND` unset) and rendered
under **QtAgg**. The backend — not the environment — carries the entire
signature: with Agg forced in both envs the stage is at parity (~24.3s wall,
~530 MB RSS in both); the cross test (conda+Agg vs pixi+QtAgg) flips wall and
RSS wholesale (conda+Agg 23.0s/521 MB, pixi+QtAgg 19.6s/824 MB). py-spy
`--idle` profiles show the ~3s as off-CPU gaps spread uniformly across the
Agg render stack at identical cpu_time (mean_load 66% vs 77%) — no blocking
call, no hotspot, nothing actionable in simACE. QtAgg's extra ~300 MB is Qt6
plus a QApplication, so the *pixi* configuration (headless Agg, deterministic,
lighter) is the correct pipeline behavior; conda's faster wall was a
desktop-session accident. ADR 0016 stands; issue #11 closed with the full
evidence chain.

Evidence base: plans/pixi-spike-findings.md (pixi 0.76.2 spike, all green).
Decision record: docs/adr/0016-pixi-canonical-simace-environment.md.

## Locked decisions

1. **Boundary** — pixi canonical for all simACE-scoped commands (checks AND
   Snakemake pipeline). Conda env stays maintained purely as the family
   environment; `envs/environment.yml` unchanged in role for fitACE layering.
2. **Command surface** — scope-based split. simACE commands, /commit gates,
   and the agent allowlist move to `pixi run` form; `tools/typecheck_family.py`
   and anything importing fitace stay conda (sisters' editables live there).
3. **Perf gate** — repeated-run comparison (N≥3) of a representative scenario,
   conda vs pixi, per-stage median wall + peak RSS; >5% regression blocks.
   Scenario chosen at implementation: main stages ≥30s each (small_test is too
   short to measure).
4. **uv retirement** — complete: `.python-version`, `uv.lock`, `[tool.uv]`,
   `.venv/` ignore, docs sections, CLAUDE.md rule, settings.json allowlist
   entries; `pip uninstall uv`; delete `.venv/` dirs.
5. **Manifest** — single default env mirroring the conda env; no `[tasks]`;
   `requires-pixi` pinned to the tested 0.76 series; `platforms = ["linux-64"]`;
   pixi.toml authoritative for simACE pins with a same-commit sync rule for
   `envs/environment.yml`.
6. **pedigree-graph editable** — stays a conda-env workflow (compat-mode, per
   RELEASE.md). pixi always consumes the released wheel; a short comment in
   pixi.toml documents the local uncommitted editable entry for ad-hoc use.

## Implementation sequence

1. Commit `pixi.toml` (spike manifest minus `[tasks]`, plus `requires-pixi`,
   plus the editable-recipe comment) and a **regenerated** `pixi.lock` — the
   spike lock predates the pedigree-graph range switch; the fresh lock must
   resolve pedigree-graph 0.6.0 from PyPI, not git.
2. `.gitignore`: add `.pixi/`; drop `.venv/` (uv retirement).
3. Remove uv artifacts (decision 4) and uninstall uv; delete `.venv/`.
4. Docs: README + docs/getting-started/installation.md (pixi setup, health
   commands, macOS-stays-conda note), CLAUDE.md (Key Rules routing: simACE =
   `pixi run`, family = conda; Snakemake section commands; pin-sync rule).
5. `/commit` skill SKILL.md: gate commands → `pixi run pytest|ruff|snakefmt`,
   ty drift gate → `pixi run ty check --ignore all --error unresolved-import`;
   note typecheck_family stays conda.
6. `.claude/settings.json`: replace the four uv allow rules with pixi
   equivalents (`pixi run pytest*`, `pixi run ruff*`, `pixi run ty*`,
   `pixi install*`, `pixi lock*` — scoped, no bare `pixi run *`).
7. Verify (functional): `pixi install --locked` + pytest/ruff/ty under pixi;
   Snakemake smoke test; bare-clone check in a fresh worktree (no external/,
   no fitACE/); `tools/typecheck_family.py` still green from conda; lock
   byte-stable after all commands.
8. Verify (perf gate, decision 3): timed repeated runs; record medians in the
   findings doc. >5% per-stage regression = stop, diagnose, or revert.
9. Memory + plans updates; promote this plan to docs/plans/ if desired.

## Rollback

Remove `pixi.toml`, `pixi.lock`, `.pixi/` ignore; restore the uv pilot per
plans/uv-dev-loop-adoption-v2.md rollback (or simply re-follow its
implementation sequence); conda path was never altered. ADR 0016 would be
superseded, not deleted.
