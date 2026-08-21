#!/usr/bin/env bash
# STALE (2026-08-21): this harness predates the fitACE monorepo (ADR 0017) and
# the conda retirement (ADR 0018) — it clones the retired standalone sister
# repos and builds a conda family env that is no longer the documented flow.
# Pending rewrite around: clone simACE + fitACE (monorepo) + fitACE_epimight,
# `pixi install --locked` at fitACE/, EPIMIGHT R env unchanged.
#
# verify_simace_epimight.sh — fresh-computer cross-repo verification:
# simACE + fitACE + fitACE_epimight + the BioPsyk EPIMIGHT R package, end to end.
#
# Mimics a brand-new user on a clean machine: clones the four repos from GitHub
# into a throwaway workdir, reconstructs the nested umbrella layout, builds a
# Python env and an R env, installs the editable packages + the EPIMIGHT R
# package, produces simACE data via Snakemake, runs EPIMIGHT through the
# documented standalone CLI, and asserts a real (non-empty, correctly-shaped)
# summary.tsv. Then tears everything down. Reads no sibling repo or project
# files — only lib.sh beside it.
#
# Auth: public repos clone over HTTPS (no auth); fitACE + fitACE_epimight are
# private and clone over SSH — configure a GitHub SSH key first. A private-clone
# failure is an SSH-auth problem (the script says so explicitly).
#
# NOTE: this clones fitACE + fitACE_epimight from GitHub, so it only sees the
# Part-A dependency fixes once they are committed AND pushed to the cloned ref.
# Until then, point --fitace-epimight-ref / --fitace-ref at the branch carrying
# the fix.
#
# Usage:
#   bash verify_simace_epimight.sh [--simace-ref REF] [--fitace-ref REF] \
#       [--fitace-epimight-ref REF] [--r-epimight-ref REF] [--keep]
#
# Refs (env-var equivalents in parens); URLs override via the *_URL env vars:
#   --simace-ref REF          SIMACE_REF          (default: master)   [HTTPS]
#   --fitace-ref REF          FITACE_REF          (default: main)     [SSH]
#   --fitace-epimight-ref REF FITACE_EPIMIGHT_REF (default: main)     [SSH]
#   --r-epimight-ref REF      R_EPIMIGHT_REF      (default: feature-pipeline) [HTTPS]
#   --keep                    keep envs + workdir on exit (for debugging)
#   -h, --help                show this help

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

SIMACE_URL="${SIMACE_URL:-https://github.com/rwaples/simACE.git}"
SIMACE_REF="${SIMACE_REF:-master}"
FITACE_URL="${FITACE_URL:-git@github.com:rwaples/fitACE.git}"
FITACE_REF="${FITACE_REF:-main}"
FITACE_EPIMIGHT_URL="${FITACE_EPIMIGHT_URL:-git@github.com:rwaples/fitACE_epimight.git}"
FITACE_EPIMIGHT_REF="${FITACE_EPIMIGHT_REF:-main}"
R_EPIMIGHT_URL="${R_EPIMIGHT_URL:-https://github.com/BioPsyk/epimight.git}"
R_EPIMIGHT_REF="${R_EPIMIGHT_REF:-feature-pipeline}"

while [ $# -gt 0 ]; do
  case "$1" in
    --simace-ref)          SIMACE_REF="$2"; shift 2 ;;
    --fitace-ref)          FITACE_REF="$2"; shift 2 ;;
    --fitace-epimight-ref) FITACE_EPIMIGHT_REF="$2"; shift 2 ;;
    --r-epimight-ref)      R_EPIMIGHT_REF="$2"; shift 2 ;;
    --keep)                KEEP=1; shift ;;
    -h|--help)             show_help "${BASH_SOURCE[0]}" ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

step "Preflight"
preflight_tools
ssh_preflight   # soft: warns but does not hard-fail; the private clone is the real gate

make_workdir

# ---------------------------------------------------------------------------
# Reconstruct the nested umbrella layout under $WORK.
# ---------------------------------------------------------------------------
step "Clone + reconstruct nested layout"
clone "$SIMACE_URL" "$SIMACE_REF" "$WORK/simACE"
ok "simACE (HTTPS, $SIMACE_REF)"

# First private clone: name SSH auth as the likely cause on failure.
clone "$FITACE_URL" "$FITACE_REF" "$WORK/simACE/fitACE" \
  "fitACE clone failed — almost certainly missing/unconfigured GitHub SSH auth for the private repos; see scripts/verify/README.md"
ok "fitACE (SSH, $FITACE_REF)"

clone "$FITACE_EPIMIGHT_URL" "$FITACE_EPIMIGHT_REF" "$WORK/simACE/fitACE/fitACE_epimight"
ok "fitACE_epimight (SSH, $FITACE_EPIMIGHT_REF)"

mkdir -p "$WORK/simACE/external"
clone "$R_EPIMIGHT_URL" "$R_EPIMIGHT_REF" "$WORK/simACE/external/epimight"
ok "EPIMIGHT R package (HTTPS, $R_EPIMIGHT_REF)"

# The symlink is git-tracked in fitACE_epimight (external/epimight ->
# ../../../external/epimight); re-create defensively if a checkout dropped it,
# then assert it RESOLVES — a wrong `../` count fails loud here at setup rather
# than as a confusing "file not found" mid-pipeline.
SYMDIR="$WORK/simACE/fitACE/fitACE_epimight/external"
mkdir -p "$SYMDIR"
[ -L "$SYMDIR/epimight" ] || ln -s ../../../external/epimight "$SYMDIR/epimight"
assert_path "$SYMDIR/epimight/DESCRIPTION" "epimight symlink resolves (DESCRIPTION reachable)"

# ---------------------------------------------------------------------------
# Python env: simACE env file (editable simACE) + editable fitACE + epimight.
# ---------------------------------------------------------------------------
step "Create Python env + install editable fitACE family"
PY="$(unique_env simace_epimight_py)"
register_env "$PY"
( cd "$WORK/simACE" && conda env create -f envs/environment.yml -n "$PY" )
ok "Python env $PY created (editable simACE + pedigree-graph@v0.5.1)"

# Editable installs resolve the *packaging* floor simace>=2026.05 (tags
# preserved by the full clone). cwd = $WORK/simACE so `./fitACE...` resolves.
( cd "$WORK/simACE" && run_in_env "$PY" -- python -m pip install -e ./fitACE )
( cd "$WORK/simACE" && run_in_env "$PY" -- python -m pip install -e ./fitACE/fitACE_epimight )
ok "installed -e fitACE and -e fitACE_epimight"

pv="$( cd "$WORK/simACE" && run_in_env "$PY" -- python -c 'import pandas; print(pandas.__version__)' || true )"
log "pandas in env: ${pv:-<unknown>} (expect 2.x post-A1; informational)"

# ---------------------------------------------------------------------------
# Sanity imports — exercise the runtime guard + the PAIR_KINSHIP sync check.
#   import fitace.config       -> runtime simace>= guard (Version(MIN_SIMACE))
#   import fitace.relationships-> PAIR_KINSHIP sync check
# (Bare `import fitace` is a no-op: empty __init__.py.)
# ---------------------------------------------------------------------------
step "Sanity imports (runtime version guard + PAIR_KINSHIP sync)"
if ( cd "$WORK/simACE" && run_in_env "$PY" -- python -c \
      'import fitace.config; import fitace.relationships; import fitace_epimight, simace; from pedigree_graph import PAIR_KINSHIP' ); then
  ok "sanity imports OK"
else
  fail "sanity imports failed (runtime version guard, PAIR_KINSHIP sync, or a missing install)"
fi

# ---------------------------------------------------------------------------
# R env: build/install the EPIMIGHT R package from its absolute source path.
# ---------------------------------------------------------------------------
step "Create R env + install EPIMIGHT R package"
RENV="$(unique_env epimight_r)"
register_env "$RENV"
conda env create -f "$WORK/simACE/fitACE/fitACE_epimight/environment.yml" -n "$RENV"
ok "R env $RENV created"

# install.packages() only warns (doesn't error-exit) on failure, so confirm by
# loading the package — library() exits non-zero if it didn't install.
if run_in_env "$RENV" -- Rscript -e \
     "install.packages('$WORK/simACE/external/epimight', repos=NULL, type='source'); library(epimight)"; then
  ok "EPIMIGHT R package installed + loads"
else
  fail "EPIMIGHT R package failed to install/load"
fi

# ---------------------------------------------------------------------------
# Produce simACE data, then run EPIMIGHT end-to-end through the standalone CLI.
# ---------------------------------------------------------------------------
step "Snakemake smoke (produce simACE data)"
if ( cd "$WORK/simACE" && run_in_env "$PY" -- snakemake --cores 4 results/test/small_test/scenario.done ); then
  ok "simACE data produced"
else
  fail "snakemake smoke target failed"
fi

step "Run EPIMIGHT end-to-end (thread-pinned, throwaway R env)"
EPI_OUT="$WORK/epimight"
# fitace-epimight-run targets the throwaway R env via --conda-env (not the
# rules' hardcoded `conda run -n epimight`). Thread-pinning (R_DATATABLE/OMP=1)
# is required because EPIMIGHT is nondeterministic at fixed seed otherwise.
# Fallback if the console script is missing: `python -m fitace_epimight.cli`.
if ( cd "$WORK/simACE" \
      && export R_DATATABLE_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      && run_in_env "$PY" -- fitace-epimight-run \
           --phenotype results/test/small_test/rep1/trait.parquet \
           --output-dir "$EPI_OUT" --rels FS,PO --draws 5 --seed 42 --conda-env "$RENV" ); then
  ok "EPIMIGHT run completed"
else
  fail "EPIMIGHT run failed"
fi

# ---------------------------------------------------------------------------
# Assert a real, non-empty, correctly-shaped result.
# Determinism is *set up* (thread-pinned) but NOT asserted: we check shape +
# finiteness only, never exact h² values (cross-machine value-equality is
# brittle). Do not "strengthen" this into an equality check — it is a
# functional smoke, not a numerical regression test.
# ---------------------------------------------------------------------------
step "Assert EPIMIGHT summary.tsv (shape + finiteness)"
if ( cd "$WORK/simACE" && run_in_env "$PY" -- python -c '
import sys
import numpy as np
import pandas as pd
s = pd.read_csv(sys.argv[1], sep="\t")
assert len(s) >= 4, f"expected >=4 rows, got {len(s)}"
rels = set(s["rel"])
traits = set(s["trait"])
assert {"FS", "PO"} <= rels, f"rels missing FS/PO: {sorted(rels)}"
assert {"trait1", "trait2"} <= traits, f"traits missing trait1/2: {sorted(traits)}"
h2 = s["h2"].to_numpy()
assert len(h2) > 0 and np.isfinite(h2).all(), f"non-finite/empty h2: {h2}"
print(f"OK: {len(s)} rows; rels={sorted(rels)}; traits={sorted(traits)}")
' "$EPI_OUT/summary.tsv" ); then
  ok "summary.tsv valid (rows>=4, rels⊇{FS,PO}, traits⊇{trait1,trait2}, finite h2)"
else
  fail "summary.tsv assertion failed (missing/empty/wrong-shape EPIMIGHT output)"
fi

# EXIT trap performs cleanup (both envs + workdir) + prints PASS/FAIL summary.
