# Releasing the simACE / fitACE lockstep family

This repository is the umbrella for a **lockstep family** of ten git repos that
are developed together and versioned together. A release tags all ten at one
CalVer (`vYYYY.MM[.patch]`) in a single coordinated step.

Authoritative design: [`docs/adr/0012-lockstep-family-versioning.md`](docs/adr/0012-lockstep-family-versioning.md)
(simACE side) and [`fitACE/docs/adr/0002-lockstep-family-versioning.md`](fitACE/docs/adr/0002-lockstep-family-versioning.md)
(fitACE-side mechanics). Canonical vocabulary ("Lockstep family", "Family
version", "Family floor") lives in both `CONTEXT.md` files.

---

## The ten members

A release tags every member, **including untouched ones**, so the version is
identical across the family at the tag.

| # | Repo | Path (from this root) | Versioned by |
|---|------|-----------------------|--------------|
| 1 | simACE | `.` | setuptools-scm |
| 2 | fitACE core | `fitACE` | setuptools-scm |
| 3 | fitACE_epimight | `fitACE/fitACE_epimight` | setuptools-scm |
| 4 | fitACE_pcgc | `fitACE/fitACE_pcgc` | setuptools-scm |
| 5 | fitACE_iter_reml | `fitACE/fitACE_iter_reml` | setuptools-scm |
| 6 | fitACE_tetraher | `fitACE/fitACE_tetraher` | setuptools-scm |
| 7 | fitACE_pafgrs | `fitACE/fitACE_pafgrs` | setuptools-scm |
| 8 | fitACE_stan | `fitACE/fitACE_stan` | setuptools-scm |
| 9 | fitACE_frailty | `fitACE/fitACE_frailty` | setuptools-scm |
| 10 | ace_iter_reml (C++ binary) | `fitACE/fitACE_iter_reml/ace_iter_reml` | CMake `git describe` |

> `fitace_sreml` is a *tenth import package* but **not** a separate
> distribution — it ships inside the `fitace_iter_reml` dist (one dist, two
> import packages), so it has no tag of its own.

**Excluded** (keep their own independent SemVer): `pedigree-graph`, `pedsum`,
`tetraher_simace`. These are public and/or independently consumed.

---

## Versioning scheme

- **CalVer** `vYYYY.MM`, with an optional patch for a second release in the same
  month: `v2026.06`, `v2026.06.1`. The first unified lockstep release is
  `v2026.06`.
- The nine Python repos derive their version from git tags via
  **setuptools-scm**. Between tags a repo reports a dev version
  (`2026.6.dev4+g<hash>`); members are byte-identical only *at* a tagged
  release, and between releases diverge only by their setuptools-scm
  commit-distance suffix (accepted as cosmetic).
- The `ace_iter_reml` binary embeds the **raw `git describe --tags --always
  --dirty`** string at CMake configure time (`src/version.h.in` →
  `configure_file` → `version.h`). It is deliberately *not* PEP 440-normalized,
  so the binary's provenance string matches the family tag exactly.

### Compatibility floor

One constant — `FAMILY_FLOOR` in [`fitACE/fitace/_deps.py`](fitACE/fitace/_deps.py) —
is the single minimum-compatible Family version. It is referenced with `>=`
semantics (a dev build of a *later* release still satisfies it) by:

- every family `pyproject.toml` pin (`simace>=` / `fitace>=`),
- the consistency test `fitACE/tests/test_dependency_floors.py`,
- the import-time runtime guard in `fitACE/fitace/config.py`.

simACE is upstream of the floor and does not import it. **Bumping the floor is a
single edit** in `_deps.py` per release, enforced family-wide by the
consistency test.

### Runtime version strings

- Every family Python package exposes `__version__`
  (`importlib.metadata.version("<dist>")`).
- All 14 installed console scripts accept `--version`
  (via `simace.core.cli_base.add_version_arg`).
- The binary accepts `--version` (`ace_iter_reml --version`).

### Provenance stamped into outputs

| Producer | Artifact | Keys stamped |
|----------|----------|--------------|
| simACE simulate | `params.yaml` | `simace_version` |
| Core fit-run context (`FitRunContext.base_meta`) | every Fit `*.vc.tsv.meta` | `simace_version`, `fitace_version` |
| PCGC adapter | `*.vc.tsv.meta` | `fitace_pcgc_version` |
| TetraHer adapter | `*.vc.tsv.meta` | `fitace_tetraher_version` |
| iter_reml Snakemake wrapper | `*.vc.tsv.meta` | `fitace_iter_reml_version` |
| ace_iter_reml binary (self-stamp) | `*.vc.tsv.meta` | `ace_iter_reml_version` |

> `pafgrs` and `epimight` produce no Fit-result `.meta` sidecar and are **not**
> stamped — tracked as a follow-up (see GitHub issue for "version provenance
> into pafgrs/epimight artifacts").

---

## The release helper

[`tools/release.py`](tools/release.py) tags all ten repos locally and prints the
per-repo `git push` commands. **It never pushes** (repo-wide no-`git push`
rule).

```bash
python tools/release.py v2026.06            # tag all ten repos locally
python tools/release.py v2026.06 --dry-run  # run checks + report; tag nothing
python tools/release.py v2026.06.1 -m "hotfix: <summary>"
```

It is **all-or-nothing**: it refuses (exit `1`) unless *every* member is

- present (a git work tree),
- clean (no uncommitted changes or untracked non-ignored files),
- not already tagged at the requested version.

If a tag creation fails partway, the tags already created in that run are rolled
back. The tag-format check rejects anything that isn't `vYYYY.MM[.patch]`
(exit `2`).

Because setuptools-scm reads **local** tags, the runtime version and the
`FAMILY_FLOOR` guard clear as soon as the local tags exist and the family is
reinstalled — the push only *publishes*.

---

## Cutover — step by step

This is a **hard cutover**: bumping `FAMILY_FLOOR` to a release the working tree
hasn't been tagged at makes `import fitace.config` raise (the running build is
still `2026.5.x.devN < 2026.06`). So most full fitACE test runs and any
`import fitace.config` will fail **until** the local tags are cut and the family
is reinstalled. Run the steps in this order.

### 0. Land and clean

Commit the final implementation/docs changes in each affected repo. Confirm all
ten repos are clean (the helper refuses dirty repos):

```bash
python tools/release.py v2026.06 --dry-run
```

A green dry-run (`all 10 family repos are clean and untagged`) is the gate.

### 1. Tag locally

```bash
python tools/release.py v2026.06
```

This creates the annotated tags in all ten repos. No push is needed for the
guard to clear.

### 2. Reinstall the family editable

So the new `2026.06` versions and the `>=2026.06` floor take effect together:

```bash
pip install -e .                         # simACE
pip install -e fitACE                    # fitACE core
for s in epimight pcgc iter_reml tetraher pafgrs stan frailty; do
  pip install -e "fitACE/fitACE_$s"
done
# Restore external editable deps the family reinstall may have clobbered:
# `pip install -e .` re-resolves simACE's `pedigree-graph` range (pedigree-graph
# publishes to PyPI as of v0.6.0) and installs the newest matching wheel over
# the editable link, shadowing the source checkout.
# `editable_mode=compat` writes a plain-path `.pth` that ty can follow; the
# default import-hook mode (`__editable__*.pth` + `_finder.py`) is invisible to
# ty and would surface `unresolved-import` errors for `pedigree_graph`.
pip install -e external/pedigree-graph --config-settings editable_mode=compat
# verify: pip show pedigree-graph → "Editable project location"
```

(Reinstalling also regenerates the console-script wrappers, e.g. a stale
`simace-analyze` entry point.)

### 3. Reconfigure + rebuild the binary

`configure_file` regenerates `version.h` only on the CMake **configure** step,
so reconfigure (don't just rebuild) after tagging. Build the binary in its own
conda env (`ace_iter_reml` / `ace_iter_reml_fp32`), not the simACE env. The
`build-fp*/` dirs are gitignored — rebuild, don't commit:

```bash
cd fitACE/fitACE_iter_reml/ace_iter_reml
cmake -S . -B build-fp64 && cmake --build build-fp64 -j
cmake -S . -B build-fp32 && cmake --build build-fp32 -j
```

> If a `build-fp*/` cache was created at an older source path, `cmake -S . -B`
> errors with a path-mismatch — delete the dir and configure fresh.

### 4. Verify (now that the guard can pass)

```bash
# Runtime version strings — all ten import packages (incl. fitace_sreml):
python -c "import simace, fitace, fitace_epimight, fitace_pcgc, fitace_iter_reml, \
  fitace_sreml, fitace_tetraher, fitace_pafgrs, fitace_stan, fitace_frailty as F; \
  print(simace.__version__, fitace.__version__)"

# Console-script --version spot checks:
simace-simulate --version
fitace-simple-ltm-stats --version
fitace-epimight-run --version
./fitACE/fitACE_iter_reml/ace_iter_reml/build-fp64/ace_iter_reml --version

# Floor + guard:
pytest fitACE/tests/test_dependency_floors.py -q
python -c "import fitace.config; print('guard cleared')"

# Full suites:
pytest tests/ -q                     # simACE
cd fitACE && pytest tests/           # fitACE + touched sisters

# Provenance smoke (grep the sidecars):
snakemake --cores 4 results/test/small_test/scenario.done
grep simace_version results/test/small_test/*/params.yaml
# then run a pcgc + tetraher + iter_reml fit and grep *.vc.tsv.meta for
# simace_version / fitace_version / fitace_<method>_version / ace_iter_reml_version
```

### 5. Push

The helper printed the per-repo push commands in step 1; run them (per the
repo-wide rule, the helper never pushes):

```bash
git -C <abspath> push origin v2026.06     # one per member, ten total
```

---

## Rollback

If something fails between local-tag (step 1) and push (step 5), delete the
local tags in all ten repos:

```bash
for rel in . fitACE fitACE/fitACE_epimight fitACE/fitACE_pcgc \
           fitACE/fitACE_iter_reml fitACE/fitACE_tetraher fitACE/fitACE_pafgrs \
           fitACE/fitACE_stan fitACE/fitACE_frailty \
           fitACE/fitACE_iter_reml/ace_iter_reml; do
  git -C "$rel" tag -d v2026.06
done
```

Deleting tags alone does **not** roll back a working tree that still contains
`FAMILY_FLOOR = "2026.06"`. If the committed floor/version changes must also be
backed out, reset/revert those (unpushed) commits and reinstall.

---

## Cutting the next release

1. Bump `FAMILY_FLOOR` in `fitACE/fitace/_deps.py` and the `simace>=` / `fitace>=`
   pins in every family `pyproject.toml` to the new `YYYY.MM` (the
   `test_dependency_floors` test fails if any drift).
2. Commit, then run the cutover above with the new `vYYYY.MM`.
