# fitACE-layer pixi spike (2026-08-21)

Stage 2 of the ADR 0016 staging proposal: can pixi absorb what the conda
family env does for fitACE + the sisters? Spike manifest: `fitACE/pixi.toml`
(uncommitted), env at `fitACE/.pixi/`.

## The layer is smaller than assumed

`envs/environment-fitace.yml` is just `pip install -e ../fitACE[plot]`; the
sisters are editable-installed by hand (RELEASE.md step 2). EPIMIGHT R lives in
its own dedicated conda env (`epimight-master`, invoked by name from the
rules); the `ace_iter_reml` / `ace_pcgc` C++ binaries and LDAK build in their
own envs. **The family layer is purely nine cross-dependent private Python
editables** — no R, no native compilation.

## Results

| Check | Result |
|---|---|
| Solve + install: simace (`..`) + fitace (`.`) + 7 sisters as editable path deps | 10s lock / 3.5s install (warm); all nine import, incl. `fitace_sreml` |
| Version floors (`fitace>=2026.08` etc. vs setuptools-scm dev versions) | Resolve correctly |
| Test suites under the pixi env | fitace 280, pcgc 214+1s, iter_reml 110+4s, tetraher 31+1s, pafgrs 119, frailty 7, epimight 176 (19 deselected = R set) — **all pass** |
| fitACE Snakemake orchestrator (cross-repo `include:` chain) | Loads; documented target dry-runs green (schedules `epimight_create_input, epimight_run`); a data-missing target fails **identically** in conda and pixi |
| pedigree-graph | 0.7.0 wheel in the pixi env (the editable role stays conda for now; same local-entry recipe as simACE if conda retires) |

## The one parity gap: ty — RESOLVED 2026-08-21: there is no gap

The apparent divergence ("pixi ty reports ~78 advisories the conda run does
not, same source") was a **confounded comparison, not a resolution
difference**. The "conda clean" data points were the 2026-08-13/17 sweeps —
run on pre-polars-migration source (`70237c9`). The polars-migration merge
(`55f767b`) landed 2026-08-20 13:42 in another session; the first
post-merge typecheck of any kind was this spike's pixi sweep. No
conda-resolved check ran on post-merge source until later that morning —
and when it did, it reported the same findings.

Controlled A/B on frozen identical source (`b29aac2`, pre-triage): conda
`--python` → 39 diagnostics, pixi `--python` → 39, bare conda discovery →
39. Full family sweep on current source: conda and pixi agree repo-for-repo,
count-for-count (simACE clean; 34 findings: fitACE 6, epimight 5, pcgc 13,
iter_reml 5, pafgrs 1, stan 2, pedsum 2). Issue #12 records the resolution.
Nothing to report upstream to astral-sh/ty; the sweep can run under either
env.

At least one surfaced finding was a **real post-polars annotation bug**:
`compute_prevalence(df: pd.DataFrame)` (`simace/analysis/stats/incidence.py:441`)
while `analyze.py:122` passes a polars frame. The findings were triage
(simACE side fixed in `fc51095`/`a3e142b`/`5770297`), not suppression — the
polars migration made them, both envs see them.

## Read

The fitACE layer migrates cleanly at the packaging/test/orchestration level.
Open items before conda could actually retire: (1) ~~the ty parity
investigation~~ (resolved — no gap) + the family-repo finding triage (34
findings, both envs agree); (2) where the family manifest lives and what
happens to `environment-fitace.yml` / RELEASE.md step 2; (3) editable
pedigree-graph's home. The layout-bound path deps are acceptable here by
design — fitACE and the sisters only exist inside the umbrella.

Spike artifacts: `fitACE/pixi.toml` + `fitACE/pixi.lock` (uncommitted),
`fitACE/.pixi/` env, suite logs in the session scratchpad.
