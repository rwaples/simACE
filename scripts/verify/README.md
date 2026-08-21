# Fresh-computer install verification

Three **standalone** bash scripts that mimic a brand-new user on a clean
machine. Each one clones what it needs **from GitHub** into a throwaway workdir,
creates its own throwaway env(s) (pixi for simACE since ADR 0018; conda in the stale epimight harness), runs the documented install + smoke test, asserts
concrete non-empty outputs, reports PASS/FAIL, and tears everything down.

At runtime the scripts **read no sibling repo or project files — only `lib.sh`
beside them.** Copy this `verify/` directory to a clean machine (only `git` +
`conda` required) and run from any directory.

| Script | Repos exercised | Auth | Rough cost |
|---|---|---|---|
| `verify_pedsum.sh` | pedsum | none (HTTPS) | one env solve + fast CLI smoke |
| `verify_simace.sh` | simACE | none (HTTPS) | one env solve + pytest + Snakemake smoke |
| `verify_simace_epimight.sh` | simACE + fitACE + fitACE_epimight + EPIMIGHT R pkg | SSH for the two private repos | two env solves + R pkg build + a fit |

## Prerequisites

- `git` and `conda` on `PATH`.
- For `verify_simace_epimight.sh` only: a working **GitHub SSH key** — `fitACE`
  and `fitACE_epimight` are private and clone over SSH. (simACE, pedsum, and the
  BioPsyk EPIMIGHT R package are public HTTPS, no auth.) A private-clone failure
  is an **SSH-auth** problem; the script says so explicitly when the first
  private clone fails.

## Running

```bash
bash scripts/verify/verify_pedsum.sh
bash scripts/verify/verify_simace.sh
bash scripts/verify/verify_simace_epimight.sh
```

Each prints a green `PASS` summary and exits 0 on success; a counted-failure
exit 1 on a failed check; or `ABORTED (rc=N)` if it died during setup (clone,
env solve, missing prereq, bad `--*-ref`).

**Wall-clock is conda-solve-bound and varies widely by machine and network** —
expect minutes for tests 1–2 and tens of minutes for test 3, dominated by env
solves rather than any tight range.

## Flags

All scripts accept `--keep` (retain envs + workdir for debugging) and `-h`.

Refs and URLs are overridable per repo, by flag **or** env var (a single global
`--ref` is intentionally not offered — test 3 spans repos with different default
branches):

| Repo | Flag | Env var | Default ref | Auth |
|---|---|---|---|---|
| simACE | `--simace-ref` | `SIMACE_REF` | `master` | HTTPS |
| pedsum | `--pedsum-ref` | `PEDSUM_REF` | `main` | HTTPS |
| fitACE | `--fitace-ref` | `FITACE_REF` | `main` | SSH |
| fitACE_epimight | `--fitace-epimight-ref` | `FITACE_EPIMIGHT_REF` | `main` | SSH |
| EPIMIGHT R pkg | `--r-epimight-ref` | `R_EPIMIGHT_REF` | `feature-pipeline` | HTTPS (BioPsyk/epimight) |
| pedigree-graph | — (pip `git+…@v0.5.1` at env-create) | — | `v0.5.1` | HTTPS |

Clone URLs are overridable via the matching `*_URL` env vars (e.g.
`SIMACE_URL`, `FITACE_URL`). `verify_pedsum.sh` also takes `--full` (run
pedsum's ~80 s test suite; default skips).

> **Full clones, on purpose.** The scripts never use `--depth`: `setuptools-scm`
> derives the CalVer version from git tags, and a shallow clone yields a bogus
> version that trips fitACE's `simace>=2026.05` floor.

## Part-A push requirement (test 3)

`verify_simace_epimight.sh` clones `fitACE` + `fitACE_epimight` from GitHub, so
it only sees the Part-A dependency fixes (the pandas pin realignment and the
unified `simace>=2026.05` / `fitace>=2026.04` floors) **once they are committed
and pushed** to the cloned ref. Until then, point `--fitace-epimight-ref` /
`--fitace-ref` at the branch carrying the fix. A run against a ref that lacks
the fix is not a from-a-fresh-machine pass.

## The pandas downgrade is under test — not sold as risk-free

Part A realigns `fitACE_epimight`'s pandas pin from `>=3.0,<4` down to
`>=2.2,<3` so a fresh env resolves to pandas 2.3.* (the whole rest of the stack
already targets pandas <3 for Snakemake/eido/SLURM compatibility). A fresh grep
found no pandas-3-only APIs, but pandas 3's copy-on-write default and
pyarrow-backed dtype changes can bite at the *data* level even with an unchanged
API. The **functional smoke in test 3 (run EPIMIGHT, parse a real `summary.tsv`)
is the safety net**, not the grep — the 3.0.3 → 2.3.x resolution on a fresh env
is itself the thing under test.

## Proving they fail loudly

- `verify_simace_epimight.sh --keep` then point `--conda-env` at a nonexistent
  env (edit the script) → the EPIMIGHT step fails as a counted `FAIL`.
- `verify_simace.sh --simace-ref no-such-branch` → the clone fails fast and the
  run is labelled `ABORTED (rc=N)`.
- In all cases the EXIT trap still prints the summary, labels an unexpected
  abort `ABORTED (rc=N)`, and preserves a non-zero exit.

Leftover envs (if a run is killed): `conda env list | grep _verify_`. `--keep`
retains envs + workdir deliberately.

## Round-1 scope

In: simACE (standalone), pedsum (standalone), and the simACE + fitACE +
fitACE_epimight + EPIMIGHT cross-repo path. Out: the C++ method stacks (PCGC,
iter_reml, TetraHer/LDAK), the other `fitACE_*` sisters (their `simace>=` /
`fitace>=` pins are covered by fitACE's `tests/test_dependency_floors.py`, not
by these scripts), and bootstrapping conda itself.
