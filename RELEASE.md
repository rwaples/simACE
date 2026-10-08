# Releasing the simACE / fitACE lockstep family

This repository is the umbrella for a **lockstep family** developed and
versioned together. Since ADR 0017 (family monorepo) a release tags **three
checkouts** at one SemVer (`vMAJOR.MINOR.PATCH`) in a single coordinated step;
within the fitACE monorepo, all seven distributions and the C++ binary read
that one tag, so intra-monorepo lockstep is structural.

Authoritative design: [`docs/adr/0012-lockstep-family-versioning.md`](docs/adr/0012-lockstep-family-versioning.md)
(the lockstep mechanism) and [`docs/adr/0023-lockstep-semver.md`](docs/adr/0023-lockstep-semver.md)
(the SemVer scheme and what counts as breaking). Canonical vocabulary ("Lockstep family", "Family
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

- **SemVer** `vMAJOR.MINOR.PATCH`, all three numbers, no leading zeros. The
  first SemVer release is `v0.1.0`; releases through `v2026.09.2` used CalVer,
  and those tags stay in git.
- **Before 1.0 the minor is the breaking digit.** Bump the minor when a release
  removes or renames part of the CLI and config (subcommands, flags, scenario
  or fit config keys) or of the result files and their schemas. Additions,
  fixes, and changed output values for the same config and seed are a patch.
  ADR 0023 §3 defines the contract.
- Every Python distribution derives its version from git tags via
  **setuptools-scm**. Between tags a checkout reports a dev version
  (`0.1.1.dev4+g<hash>`); members are byte-identical only *at* a tagged
  release, and between releases diverge only by their setuptools-scm
  commit-distance suffix (accepted as cosmetic).
- The `ace_iter_reml` binary embeds the **raw `git describe --tags --always
  --dirty`** string at CMake configure time (`src/version.h.in` →
  `configure_file` → `version.h`). It is deliberately *not* PEP 440-normalized,
  so the binary's provenance string matches the family tag exactly.

### Compatibility floor

One constant — `FAMILY_FLOOR` in [`fitACE/fitace/_deps.py`](fitACE/fitace/_deps.py) —
is the single minimum-compatible Family version. It pins one minor line: for a
floor of `0.1.0`, every family member accepts `>=0.1.0,<0.2`, including dev
builds of later 0.1.x patches. It is referenced by:

- every family `pyproject.toml` pin (`simace>=0.1.0,<0.2` / `fitace>=0.1.0,<0.2`),
- the consistency test `fitACE/tests/test_dependency_floors.py`, which checks
  both bounds,
- the import-time runtime guard in `fitACE/fitace/config.py`, which rejects an
  installed simACE below the floor or outside its minor line (and so every
  CalVer-era install).

simACE is upstream of the floor and does not import it. The floor moves at
every minor release, and at a patch release only when fitACE needs a simACE
fix from it. The consistency test fails on any pin left behind.

### Runtime version strings

- Every family Python package exposes `__version__`
  (`importlib.metadata.version("<dist>")`).
- Every family console script accepts `--version`
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
pixi run python tools/release.py --next            # print the next patch and minor tags
pixi run python tools/release.py vX.Y.Z            # tag the three checkouts locally
pixi run python tools/release.py vX.Y.Z --dry-run  # run checks + report; tag nothing
pixi run python tools/release.py vX.Y.Z -m "fix: <summary>"
```

It is **all-or-nothing**: it refuses (exit `1`) unless *every* member is

- present (a git work tree),
- clean (no uncommitted changes or untracked non-ignored files),
- not already tagged at the requested version.

If a tag creation fails partway, the tags already created in that run are rolled
back. The tag-format check (exit `2`) names why it rejects a tag: not
`vMAJOR.MINOR.PATCH`, a leading zero, or a CalVer-era major (2000 and up).

Because setuptools-scm reads **local** tags, the runtime version and the
`FAMILY_FLOOR` guard clear as soon as the local tags exist and the family is
reinstalled — the push only *publishes*.

### Independently versioned checkouts

`--repo pg-phenotype` tags pg-phenotype alone, at its own version, outside the
lockstep family:

```bash
pixi run python tools/release.py --repo pg-phenotype --next
pixi run python tools/release.py --repo pg-phenotype vX.Y.Z --dry-run
pixi run python tools/release.py --repo pg-phenotype vX.Y.Z
```

It refuses (exit `1`) unless the checkout is clean and untagged, every file
that states its version (`Cargo.toml`, `r/src/rust/Cargo.toml`,
`r/DESCRIPTION`) equals the tag, and `CHANGELOG.md` has a `## vX.Y.Z` section.
Push the tag with `/push pg-phenotype`.  The tag push runs its publish
workflow, which releases to PyPI and GitHub; the procedure is in
`external/pg-phenotype/docs/releasing.md`.  To add another checkout, list
its version files in `INDEPENDENT_VERSIONS` in `tools/release.py`.

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
pixi run python tools/release.py vX.Y.Z --dry-run
```

A green dry-run (`all 3 family repos are clean and untagged`) is the gate.

### 1. Tag locally

```bash
pixi run python tools/release.py vX.Y.Z
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

(Reinstalling also regenerates the console-script wrappers, such as the
`simace` entry point. pedigree-graph is consumed as its PyPI wheel in
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

### 3b. Rebuild the TetraHer LDAK fork (when its runpath is stale)

`tetraher_simace/ldak6.2.simace` carries no version stamp (it rides fitACE's
tag), but it is dynamically linked against OpenBLAS with a `RUNPATH` baked in
at build time. A binary built under the retired `simACE` conda env (ADR 0018)
fails at runtime with `libopenblas.so.0: cannot open shared object file` once
that env is gone. Check, and rebuild against the fitACE pixi env if needed
(the binary is gitignored — rebuild, don't commit):

```bash
cd fitACE/tetraher_simace
ldd ldak6.2.simace | grep 'not found'          # any hit → rebuild
CONDA_PREFIX="$(readlink -f ../.pixi/envs/default)" bash build.sh
ldd ldak6.2.simace | grep openblas             # expect .pixi/envs/default/lib
```

`build.sh` reads `CONDA_PREFIX` for `-L`/`-rpath`; pointing it at the pixi env
is the intended use after ADR 0018.

### 4. Verify (now that the guard can pass)

Every check asserts the new version; none just prints it. `V` is the version
without the `v`.

```bash
V=0.1.0

# Versions: all ten import packages, every family console script, and both
# ace_iter_reml builds (fp64 and fp32) report $V; exits 1 on any mismatch.
pixi run --manifest-path fitACE/pixi.toml python tools/verify_release.py "$V"

# Floor + guard:
pixi run --manifest-path fitACE/pixi.toml python -m pytest fitACE/tests/test_dependency_floors.py \
  fitACE/tests/test_version_guard.py -q
pixi run --manifest-path fitACE/pixi.toml python -c "import fitace.config; print('guard cleared')"

# Full suites:
pixi run test                                                    # simACE
( cd fitACE && pixi run pytest tests/ -q )                       # fitACE core
# (method-package suites, when touched:
#   pixi run --manifest-path fitACE/pixi.toml pytest fitACE/fitACE_<x>/tests/ -q )
```

**Fresh provenance smoke.** `simace run` skips complete reps and never compares
the recorded `simace_version`, so a plain rerun would check last release's
stamps. `--force` recomputes every rep (small_test's outputs are disposable):

```bash
pixi run simace run small_test --force
pixi run --manifest-path fitACE/pixi.toml python tools/verify_release.py "$V" \
  --provenance results/test/small_test/rep*/params.yaml results/test/small_test/rep*/run.yaml
```

Then simulate one `pcgc_bias_small` cell, refit pcgc, tetraher, and iter_reml
on it (small_test leaves `tetraher_prevalence` null, which disables TetraHer),
and check the sidecars. They carry `simace_version`, `fitace_version`,
`fitace_<method>_version`, and `ace_iter_reml_version` (the binary stamps
`v$V`, which the script accepts). The `simace run` step also gives a cell built
before an output-layout change the inputs the fit rules now read:

```bash
CELL=results/pcgc_bias_small/pcgc_bias_small_A50_C00_K25/rep1
pixi run simace run pcgc_bias_small_A50_C00_K25 --rep 1 --no-plots --force
( cd fitACE && pixi run snakemake --cores 4 --force \
    $CELL/pcgc/fit.vc.tsv $CELL/tetraher/fit.vc.tsv $CELL/iter_reml_fp64/fit.vc.tsv )
pixi run --manifest-path fitACE/pixi.toml python tools/verify_release.py "$V" \
  --provenance $CELL/params.yaml $CELL/run.yaml $CELL/{pcgc,tetraher,iter_reml_fp64}/fit.vc.tsv.meta
```

### 5. Push

The helper printed the per-repo push commands in step 1; run them (per the
repo-wide rule, the helper never pushes):

```bash
git -C <abspath> push origin vX.Y.Z     # one per checkout, three total
```

---

## Rollback

If something fails between local-tag (step 1) and push (step 5), delete the
local tags in the three checkouts:

```bash
for rel in . fitACE fitACE/fitACE_epimight; do
  git -C "$rel" tag -d vX.Y.Z
done
```

Deleting tags alone does **not** roll back a working tree that still contains
`FAMILY_FLOOR` bumped to the new release. If the committed floor/version
changes must also be backed out, reset/revert those (unpushed) commits and
reinstall.

---

## Cutting the next release

1. Run `.agents/skills/coordinated-release/scripts/release-preflight.sh`; it
   prints the next patch and minor tags. Pick by ADR 0023 §3: does the release
   break the CLI/config or result-file contract?
2. For a minor release, bump `FAMILY_FLOOR` in `fitACE/fitace/_deps.py`, the
   pins in every family `pyproject.toml` to `>=0.M.0,<0.(M+1)`, and the
   matching `requires_dist` lines in `fitACE/pixi.lock` (it cannot re-solve
   until the tag exists). `test_dependency_floors` fails on any pin left
   behind.
3. Commit, then run the cutover above with the new `vX.Y.Z`.
