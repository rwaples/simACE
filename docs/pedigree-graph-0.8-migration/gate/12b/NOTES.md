# 12b wheel-site gate for pedigree-graph 0.9.0 (2026-09-19)

Slice 12 (`plans/pedigree-graph-slice-12-relationship-pairs-v2.md`) commit 8:
the pre-publish half. The 0.9.0 wheel is built from a clean worktree at the
`v0.9.0` tag and every family check unit runs against it through `PYTHONPATH`,
with the consumer locks still frozen at 0.8.3. The post-publish half — the four
dependency caps, the four relocks and the byte compare — is stage `12c`.

## The artifact under test

`tools/pg08_wheel_gate.sh <work> v0.9.0`, built from `2f7e66421dab80c16207c545a544ca0ba3c0077d`:

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.9.0-cp313-abi3-linux_x86_64.whl` | `239022df639c2b1c7b8c4254c5b885a963f48e0deeaf8d74876783e88fa1d3cd` |
| `pedigree_graph-0.9.0.tar.gz` | `45013ae13b1285e08223f53a7e17fd53d076455002c556046a10df9bf9550684` |

Both install into isolated venvs and resolve `pedigree_graph` from
site-packages with `py.typed`, `_native.pyi`, and `core_version()` equal to the
distribution version; the sdist recompiles the Rust core rather than repacking
the wheel. The package's own suite against the installed wheel:
`2913 passed, 3 skipped, 9 deselected in 204.61s`, exit 0.

## Thirteen units, not eleven

The gate ran eleven units through slice 11. `tools/family_repos.py` lists
thirteen, and the two it never covered — `fitACE_stan` and `tetraher_simace` —
had joined the family without anything proving they still passed their own
checks under a new pedigree-graph. Both are test-less units, so they gate on
ruff, formatting, and the thing that would actually break: `fitace_stan`
importing alongside `fitace` and `simace`, and the LDAK fork's binary running
and carrying its own `--simace-grouping` flag. Numerical equivalence of the
fork with upstream stays where it was, in the `fitACE` unit's
`tests/tetraher/test_fork_equivalence.py`.

`tests/test_release_gate_covers_family.py` now holds the gate's unit set equal
to the manifest's, so the next family member cannot go missing the same way.

## Result

`failed units: none`, exit 0. Per-unit records and full step logs are the JSON
files and directories beside this one.

| unit | wall | peak RSS |
|---|---|---|
| pedigree-graph | 175.0 s | 980.2 MiB |
| simACE | 114.7 s | 493.9 MiB |
| fitACE | 23.6 s | 414.0 MiB |
| fitACE_pcgc | 12.8 s | 342.2 MiB |
| fitACE_iter_reml | 59.8 s | 348.1 MiB |
| ace_iter_reml | 28.8 s | 106.2 MiB |
| fitACE_tetraher | 9.2 s | 293.2 MiB |
| tetraher_simace | 1.1 s | 103.0 MiB |
| fitACE_pafgrs | 15.1 s | 717.4 MiB |
| fitACE_stan | 1.8 s | 107.4 MiB |
| fitACE_frailty | 6.6 s | 211.5 MiB |
| fitACE_epimight | 58.8 s | 737.6 MiB |
| pedsum | 212.0 s | 333.6 MiB |

Walls are not comparable with 11a's: these ran interleaved with other work on
the box. The gate is a pass/fail record, not a benchmark; slice 12's timings
are in `../12a/`.

## What the first run caught

The first attempt failed two units, neither of them the release.

`tetraher_simace` failed because its `ldak` step asserted a zero exit and
LDAK's usage path exits 1 — the manual check that chose the step had read
`head`'s exit code through a pipe rather than the binary's. Replaced by the two
probes described above.

`simACE` failed ruff and format on untracked probe scripts under `scratchpad/`,
now gitignored, and failed 22 tests on the 0.9 effective-size changes. Those 22
were confirmed to be 0.9's doing and not the working tree's: the same five
files under the locked 0.8.3 env gave `137 passed`. They are the counterpart of
pedsum's own migration, and simACE followed with `9385970`.

## Known state at this stage

simACE's `dev` is green against 0.9.0 and red against the 0.8.3 its lock still
pins, because `ne_group_coancestry` does not exist in 0.8.3. That is the same
position pedsum took deliberately, and `12c` resolves it for both.
