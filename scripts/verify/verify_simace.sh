#!/usr/bin/env bash
# verify_simace.sh — fresh-computer install verification for simACE (public, no auth).
#
# Mimics a brand-new user on a clean machine: clones simACE from GitHub into a
# throwaway workdir, creates its own conda env via the documented
# envs/environment.yml, runs the documented import + pytest + Snakemake smoke,
# asserts concrete outputs, then tears everything down. Reads no sibling repo or
# project files — only lib.sh beside it. Run from any directory; needs only
# `git` + `conda`.
#
# Usage:
#   bash verify_simace.sh [--simace-ref REF] [--simace-url URL] [--keep]
#
# Flags (env-var equivalents in parens):
#   --simace-ref REF   git ref to check out         (SIMACE_REF;  default: master)
#   --simace-url URL   clone URL                     (SIMACE_URL;  default: rwaples/simACE)
#   --keep             keep the env + workdir on exit (for debugging)
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
preflight_tools

make_workdir

step "Clone simACE ($SIMACE_REF)"
clone "$SIMACE_URL" "$SIMACE_REF" "$WORK/simACE"
ok "cloned simACE"

step "Create conda env from envs/environment.yml"
ENV="$(unique_env simace)"
register_env "$ENV"                       # BEFORE create, so a partial solve is torn down
# Mirror the documented command exactly (README: run from repo root with
# `-f envs/environment.yml`); only `-n` is added to override the file's name.
( cd "$WORK/simACE" && conda env create -f envs/environment.yml -n "$ENV" )
ok "env $ENV created (editable simACE + pedigree-graph@v0.5.1)"

step "Import simACE + report version"
if ver="$( cd "$WORK/simACE" && run_in_env "$ENV" -- python -c 'import simace; print(simace.__version__)' )"; then
  ok "import simace OK (version: $ver)"
else
  fail "import simace failed"
fi

step "Run test suite (python -m pytest tests/ -q)"
if ( cd "$WORK/simACE" && run_in_env "$ENV" -- python -m pytest tests/ -q ); then
  ok "pytest passed"
else
  fail "pytest failed"
fi

step "Snakemake smoke (results/test/small_test/scenario.done)"
if ( cd "$WORK/simACE" && run_in_env "$ENV" -- snakemake --cores 4 results/test/small_test/scenario.done ); then
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
