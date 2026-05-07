#!/usr/bin/env bash
# Regenerate docs/images/rulegraph.png from the current Snakefile.
# Requires graphviz (`dot`). Run from repo root.
set -euo pipefail

TARGET="${1:-results/test/small_test/scenario.done}"
OUT="docs/images/rulegraph.png"

snakemake --rulegraph -- "$TARGET" | dot -Tpng -Gdpi=150 > "$OUT"
echo "Wrote $OUT"
