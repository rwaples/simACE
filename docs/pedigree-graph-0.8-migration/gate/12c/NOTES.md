# 12c post-publish gate for pedigree-graph 0.9.0 — IN PROGRESS

The second half of slice 12's commit 8, blocked on the 0.9.0 wheel reaching
PyPI. The pre-publish half is `../12b/`.

## What is already here

`byte-parity/baseline-0.8.3/`, captured before anything moved, by
`tools/pg09_byte_parity.sh`. 0.9.0 promises element-for-element identical
relationship pairs, so unlike the 0.7.1 → 0.8.0 migration the consumer outputs
must match byte for byte, and this is the side of the comparison that stops
existing once the locks move.

Both envs confirmed at `pedigree-graph 0.8.3`:

| artifact | sha256 |
|---|---|
| `results/test/small_test/rep1/report.yaml` | `6da909f7c147a83f59f354211d5691f7097d7a49b4b3b6ad4c2dab4ae5a75774` |
| `results/test/small_test/rep1/exports/pairwise_relatedness.tsv` | `505a8fd35b74925a2ce84346d4e00b16ea8a660eb0a759e65e2ba21f19d4b784` |

The rebuilt TSV came out the same 2,107,309 bytes as the copy already on disk
from 2026-09-09, so the pipeline was already reproducing byte-identically
before the version changed. The pair table is stored gzipped; gunzip it to
diff.

## What remains

1. Publish the 0.9.0 wheel and sdist whose sha256s `../12b/NOTES.md` records,
   and push the `v0.9.0` tag and the `v0.9` branch.
2. Move four `pedigree-graph>=0.8,<0.9` caps to `>=0.9,<0.10`: `pyproject.toml`,
   `fitACE/pyproject.toml`, `fitACE/fitACE_epimight/pyproject.toml`,
   `external/pedsum/pyproject.toml`. The plan said three; `fitACE_epimight`
   carries one too.
3. Relock all four trees and review each lock diff.
4. Re-run `tools/pg09_byte_parity.sh` into `byte-parity/relocked-0.9.0/` and
   compare the two manifests. Byte equality is the gate.
5. Run the thirteen-unit gate at `--routing locked`, recorded here.

Until step 2 lands, simACE's `dev` and pedsum are green against 0.9.0 and red
against the 0.8.3 their locks still pin, since `ne_group_coancestry` does not
exist in 0.8.3.
