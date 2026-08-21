#!/usr/bin/env bash
# verify_simace.sh — fresh-computer install verification for simACE (public, no auth).
#
# Mimics a brand-new user on a clean machine: clones simACE from GitHub into a
# throwaway workdir, materializes the documented pixi environment from the
# committed lock (ADR 0016/0018), runs the documented import + pytest +
# Snakemake smoke, asserts concrete outputs, then tears everything down. The
# environment lives inside the workdir (.pixi/), so cleanup is the workdir
# removal. Reads no sibling repo or project files — only lib.sh beside it.
# Run from any directory; needs only `git` + `pixi`.
#
# Usage:
#   bash verify_simace.sh [--simace-ref REF] [--simace-url URL] [--keep]
#
# Flags (env-var equivalents in parens):
#   --simace-ref REF   git ref to check out         (SIMACE_REF;  default: master)
#   --simace-url URL   clone URL                     (SIMACE_URL;  default: rwaples/simACE)
#   --keep             keep the workdir on exit (for debugging)
#   -h, --help         show this help

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

SIMACE_URL="${SIMACE_URL:-https://github.com/rwaples/simACE.git}"
SIMACE_REF="${SIMACE_REF:-master}"

while [ $# -gt 0 ]; do
  case "$1" in
    --simace-ref) SIMACE_REF="$2"; shift 2 ;;
    --simace-url) SIMACE_URL="$2"; shift 2 ;;
    --keep)       KEEP=1; shift ;;
    -h|--help)    show_help "${BASH_SOURCE[0]}" ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

step "Preflight"
require_cmd git
require_cmd pixi
ok "git + pixi present"

make_workdir

step "Clone simACE ($SIMACE_REF)"
clone "$SIMACE_URL" "$SIMACE_REF" "$WORK/simACE"
ok "cloned simACE"

step "Materialize the locked pixi environment (pixi install --locked)"
# Mirror the documented command exactly (README quick start).
( cd "$WORK/simACE" && pixi install --locked )
ok "pixi env materialized from the committed lock"

step "Import simACE + report version"
if ver="$( cd "$WORK/simACE" && pixi run python -c 'import simace; print(simace.__version__)' )"; then
  ok "import simace OK (version: $ver)"
else
  fail "import simace failed"
fi

step "Run test suite (pixi run pytest tests/ -q)"
if ( cd "$WORK/simACE" && pixi run python -m pytest tests/ -q ); then
  ok "pytest passed"
else
  fail "pytest failed"
fi

step "Snakemake smoke (results/test/small_test/scenario.done)"
if ( cd "$WORK/simACE" && pixi run snakemake --cores 4 results/test/small_test/scenario.done ); then
  ok "snakemake smoke target built"
else
  fail "snakemake smoke target failed"
fi

step "Assert smoke outputs (rep1)"
# Canonical per-rep artifacts produced by the pipeline on the default branch:
# pedigree/trait parquet (ascertainment) + report.yaml (the curated v2
# scientific report carrying validation + stats) + plot_payload.yaml (analyze).
REP="$WORK/simACE/results/test/small_test/rep1"
assert_file "$REP/pedigree.parquet"  "pedigree parquet"
assert_file "$REP/trait.parquet"     "trait parquet"
assert_file "$REP/report.yaml"       "scientific report (validation + stats)"
assert_file "$REP/plot_payload.yaml" "plot payload"

# EXIT trap performs cleanup + prints PASS/FAIL summary and sets the exit code.
