# ADR 0023: Lockstep SemVer for the family version

## Status

Accepted. Design interview 2026-10-02. Amends [ADR 0012](0012-lockstep-family-versioning.md)
§3 (the scheme) and §5 (the floor's `>=`-only semantics). ADR 0012's lockstep
mechanism and ADR 0017's membership are unchanged.

## Context

ADR 0012 gave the three lockstep checkouts (simACE, the fitACE monorepo,
fitACE_epimight) one CalVer `vYYYY.MM[.patch]`. Four things argue for SemVer
now:

- The family is preparing for a public release, where users read a SemVer
  number as a compatibility promise and `0.x` as "not yet stable".
- pedigree-graph and pedsum, the two independent checkouts, already use
  SemVer.
- A CalVer number says when a release was cut, not whether it breaks
  anything. The `v2026.09.2` to next-release step replaces Snakemake and
  the result-file layout, and the number would not show it.
- Releases don't follow months. May 2026 had four tags (`v2026.05` to
  `v2026.05.3`), July had none, and September had three.

ADR 0012 kept CalVer partly because every new tag sorted above the last.
SemVer's `0.1.0` sorts below `2026.9.2` under PEP 440. None of the three
packages is on PyPI, so the only comparisons that see the drop are the
`fitace.config` runtime guard and an editable reinstall.

## Decision

1. **The family version is SemVer**, tagged `vMAJOR.MINOR.PATCH` with all
   three numbers. The first SemVer release is `v0.1.0`. The CalVer tags stay
   in git as history. Lockstep is unchanged: `tools/release.py` tags all
   three checkouts at one version.
2. **Before 1.0, the minor is the breaking digit.** `0.MINOR` goes up for a
   breaking change; `0.x.PATCH` covers additions and fixes.
3. **A breaking change** is one that removes or renames part of either
   contract:
   - the command line and configuration: `simace` and fitACE subcommands and
     flags, and scenario or fit config keys;
   - result files and their schemas: the files under `results/`
     (`pedigree.parquet`, `cohort.parquet`, `report.yaml`, `params.yaml`,
     `run.yaml`) and fitACE's result sidecars.

   Adding a flag, a key with a default, or a column that old readers ignore
   is not breaking. Changes to output values for the same config and seed
   are not breaking either; a statistical fix or a pedigree-graph upgrade
   that moves the numbers ships as a patch, and `run.yaml`'s recorded
   version shows which release built a replicate. The Python API is not part
   of the contract.
4. **The family floor pins one minor line.** Family pins read
   `>=FLOOR,<0.(MINOR+1)` (for `v0.1.0`: `simace>=0.1.0,<0.2`), and
   `test_dependency_floors` checks both bounds. The `fitace.config` guard
   rejects an installed simACE below the floor or outside the floor's
   `0.MINOR`. That second check also rejects every CalVer-era install, so
   no special case for the old scheme is needed.

## Consequences

- Within a minor line, rerunning a scenario can give different numbers.
  Reproducing a result exactly means pinning the patch release recorded in
  `run.yaml`.
- The release preflight offers both candidates (next patch, next minor), and
  the maintainer picks by asking whether the release breaks either contract.
