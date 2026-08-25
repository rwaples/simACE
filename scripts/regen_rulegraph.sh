#!/usr/bin/env bash
# Regenerate docs/images/rulegraph.png from the current Snakefile.
# Run from repo root. snakemake and graphviz (`dot`) both come from the umbrella
# pixi env — there is no ambient environment (ADR 0018), so neither resolves bare.
set -euo pipefail

TARGET="${1:-results/test/small_test/scenario.done}"
OUT="docs/images/rulegraph.png"

pixi run snakemake --rulegraph -- "$TARGET" | pixi run dot -Tpng -Gdpi=150 > "$OUT"
echo "Wrote $OUT"
