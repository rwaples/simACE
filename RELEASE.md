# Releasing the simACE / fitACE lockstep family

This repository is the umbrella for a **lockstep family** developed and
versioned together. Since ADR 0017 (family monorepo) a release tags **three
checkouts** at one CalVer (`vYYYY.MM[.patch]`) in a single coordinated step;
within the fitACE monorepo, all seven distributions and the C++ binary read
that one tag, so intra-monorepo lockstep is structural.

Authoritative design: [`docs/adr/0012-lockstep-family-versioning.md`](docs/adr/0012-lockstep-family-versioning.md)
(simACE side) and [`fitACE/docs/adr/0002-lockstep-family-versioning.md`](fitACE/docs/adr/0002-lockstep-family-versioning.md)
(fitACE-side mechanics). Canonical vocabulary ("Lockstep family", "Family
version", "Family floor") lives in both `CONTEXT.md` files.

---

## The three tagged checkouts

A release tags every checkout, **including untouched ones**, so the version is
identical across the family at the tag.

| # | Checkout | Path (from this root) | Versioned by |
|---|------|-----------------------|--------------|
| 1 | simACE | `.` | setuptools-scm |
| 2 | fitACE (monorepo) | `fitACE` | setuptools-scm (7 distributions: `fitace`, `fitace-pcgc`, `fitace-iter-reml`, `fitace-tetraher`, `fitace-pafgrs`, `fitace-stan`, `fitace-frailty` — each pyproject sets `[tool.setuptools_scm] root = ".."`); the `ace_iter_reml` binary via CMake `git describe` off the same tag |
| 3 | fitACE_epimight | `fitACE/fitACE_epimight` | setuptools-scm |

> `fitace_sreml` is an extra *import package* but **not** a separate
> distribution — it ships inside the `fitace_iter_reml` dist (one dist, two
> import packages).

**Excluded** (keep their own independent SemVer): `pedigree-graph`, `pedsum`.
The `tetraher_simace` LDAK fork lives inside the fitACE monorepo since ADR
0017 and simply rides its tags.

---

## Versioning scheme

- **CalVer** `vYYYY.MM`, with an optional patch for a second release in the same
  month: `v2026.06`, `v2026.06.1`. The first unified lockstep release is
  `v2026.06`.
- Every Python distribution derives its version from git tags via
  **setuptools-scm**. Between tags a checkout reports a dev version
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

[`tools/release.py`](tools/release.py) tags the three checkouts locally and prints the
per-checkout `git push` commands. **It never pushes** (repo-wide no-`git push`
rule).

```bash
python tools/release.py vYYYY.MM            # tag the three checkouts locally
python tools/release.py vYYYY.MM --dry-run  # run checks + report; tag nothing
python tools/release.py vYYYY.MM.1 -m "hotfix: <summary>"
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
still the previous release's dev version, below the new floor). So most full
fitACE test runs and any `import fitace.config` will fail **until** the local
tags are cut and the family is reinstalled. Run the steps in this order.

### 0. Land and clean

Commit the final implementation/docs changes in each affected checkout. Confirm
all three checkouts are clean (the helper refuses dirty repos):

```bash
python tools/release.py vYYYY.MM --dry-run
```

A green dry-run (`all 3 family repos are clean and untagged`) is the gate.

### 1. Tag locally

```bash
python tools/release.py vYYYY.MM
```

This creates the annotated tags in all three checkouts. No push is needed for the
guard to clear.

### 2. Refresh the pixi environments' editables

setuptools-scm bakes the version at editable-install time, so after tagging,
reinstall the editable packages in each pixi environment (ADR 0018 — the
conda family env is retired):

```bash
pixi reinstall simace                                # umbrella env
pixi reinstall --manifest-path fitACE/pixi.toml \
  simace fitace fitace-epimight fitace-pcgc fitace-iter-reml \
  fitace-tetraher fitace-pafgrs fitace-stan fitace-frailty
```

(Reinstalling also regenerates the console-script wrappers, e.g. a stale
`simace-analyze` entry point. pedigree-graph is consumed as its PyPI wheel in
these envs and keeps its own SemVer — nothing to refresh at a family release;
its dev env lives in `external/pedigree-graph/pixi.toml`.)

### 3. Reconfigure + rebuild the binary

`configure_file` regenerates `version.h` only on the CMake **configure** step,
so reconfigure (don't just rebuild) after tagging. Build the binary in its own
conda env (`ace_iter_reml` / `ace_iter_reml_fp32`) — these dedicated build envs remain after ADR 0018. The
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
fitace-observed-binary-stats --version
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
git -C <abspath> push origin vYYYY.MM     # one per member, ten total
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
  git -C "$rel" tag -d vYYYY.MM
done
```

Deleting tags alone does **not** roll back a working tree that still contains
`FAMILY_FLOOR` bumped to the new release. If the committed floor/version
changes must also be backed out, reset/revert those (unpushed) commits and
reinstall.

---

## Cutting the next release

1. Bump `FAMILY_FLOOR` in `fitACE/fitace/_deps.py` and the `simace>=` / `fitace>=`
   pins in every family `pyproject.toml` to the new `YYYY.MM` (the
   `test_dependency_floors` test fails if any drift).
2. Commit, then run the cutover above with the new `vYYYY.MM`.
