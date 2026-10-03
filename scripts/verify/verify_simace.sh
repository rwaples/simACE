#!/usr/bin/env bash
# verify_simace.sh — fresh-computer install verification for simACE (public, no auth).
#
# Mimics a brand-new user: nothing from the current machine's install is used.
# The script re-runs itself under `env -i` with a throwaway HOME inside the
# workdir, so there is no ~/.pixi, no package cache, no gitconfig, and a PATH
# of system directories only. It then follows the docs of the checked-out ref:
#
#   1. Install pixi with the installation page's pinned command.
#   2. git clone, check out the ref, pixi install --locked.
#   3. The whole Quick start: simace run small_test (and the skip on rerun),
#      the HTML and PDF atlases, coverage_scenario, simace gather test, and
#      baseline100K.
#   4. The documented development checks: pytest tests/, ruff check, ty check.
#   5. The library route: pip install "simace[plot] @ git+URL@REF" into a
#      clean venv on the minimum Python from requires-python, then simulate,
#      cohort, and read the cohort parquet. pip must refuse the two Pythons
#      below the floor (a warning when its error does not name the floor), and simace run outside a checkout must exit 2 with a
#      hint rather than a traceback.
#
# Every output is asserted, and the clone must stay clean. The workdir (clone,
# environment, package cache, about 7 GB) is deleted on exit unless --keep.
# Needs only bash, git, curl, and network access.
#
# Usage:
#   bash verify_simace.sh [--simace-ref REF] [--simace-url URL] [--quick]
#                         [--no-library] [--keep]
#
# Flags (env-var equivalents in parens):
#   --simace-ref REF   git ref to check out   (SIMACE_REF; default: master, what
#                      a new clone gets; pass the tag, e.g. v0.1.0, before a
#                      release)
#   --simace-url URL   clone URL              (SIMACE_URL; default: rwaples/simACE)
#   --quick            skip pytest and baseline100K (about 5 minutes saved)
#   --no-library       skip the pip library route
#   --keep             keep the workdir on exit (for debugging)
#   -h, --help         show this help

set -euo pipefail
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
source "$(dirname "$SCRIPT")/lib.sh"

SIMACE_URL="${SIMACE_URL:-https://github.com/rwaples/simACE.git}"
SIMACE_REF="${SIMACE_REF:-master}"
QUICK="${QUICK:-0}"
LIBRARY="${LIBRARY:-1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --simace-ref) SIMACE_REF="$2"; shift 2 ;;
    --simace-url) SIMACE_URL="$2"; shift 2 ;;
    --quick)      QUICK=1; shift ;;
    --no-library) LIBRARY=0; shift ;;
    --keep)       KEEP=1; shift ;;
    -h|--help)    show_help "$SCRIPT" ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

# Outer run: make the workdir, then re-exec in a clean environment whose HOME
# lives inside it. D-Bus and XDG_RUNTIME_DIR pass through so simace run can
# reach the user's systemd for a delegated cgroup, as in a desktop shell.
if [ "${VERIFY_ISOLATED:-0}" != "1" ]; then
  require_cmd git
  require_cmd curl
  avail_kb="$(df -Pk "${TMPDIR:-/tmp}" | awk 'NR == 2 {print $4}')"
  if [ "$avail_kb" -lt $((10 * 1024 * 1024)) ]; then
    warn "less than 10 GB free in ${TMPDIR:-/tmp}; the run needs about 7 GB"
  fi
  work="$(mktemp -d "${TMPDIR:-/tmp}/simace_verify.XXXXXX")"
  mkdir -p "$work/home"
  trap - EXIT
  exec env -i \
    HOME="$work/home" PATH=/usr/local/bin:/usr/bin:/bin \
    LANG=C.UTF-8 TERM="${TERM:-dumb}" USER="${USER:-verify}" \
    ${XDG_RUNTIME_DIR:+XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR"} \
    ${DBUS_SESSION_BUS_ADDRESS:+DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS"} \
    VERIFY_ISOLATED=1 VERIFY_WORK="$work" KEEP="$KEEP" QUICK="$QUICK" LIBRARY="$LIBRARY" \
    SIMACE_URL="$SIMACE_URL" SIMACE_REF="$SIMACE_REF" \
    bash "$SCRIPT"
fi

WORK="$VERIFY_WORK"
register_dir "$WORK"
REPO="$WORK/simACE"

step "Isolated environment"
log "workdir: $WORK"
log "HOME=$HOME PATH=$PATH"
if [ ! -e "$HOME/.pixi" ] && [ ! -e "$HOME/.cache" ] && ! command -v pixi >/dev/null 2>&1; then
  ok "no pixi, no cache, nothing inherited"
else
  fail "the sandbox inherited a pixi install or cache"
fi

step "Clone simACE ($SIMACE_REF)"
clone "$SIMACE_URL" "$SIMACE_REF" "$REPO"
ok "cloned simACE at $(git -C "$REPO" describe --tags --always)"
cd "$REPO"

step "Install pixi with the documented command"
pixi_version="$(grep -oE 'PIXI_VERSION=v[0-9.]+' docs/getting-started/installation.md | head -1 | cut -d= -f2)"
if [ -z "$pixi_version" ]; then
  err "no PIXI_VERSION=vX.Y.Z in docs/getting-started/installation.md at $SIMACE_REF"
  exit 1
fi
curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION="$pixi_version" bash
export PATH="$HOME/.pixi/bin:$PATH"
if [ "$(pixi --version)" = "pixi ${pixi_version#v}" ]; then
  ok "$(pixi --version), as the installation page pins"
else
  fail "installed $(pixi --version), the installation page pins $pixi_version"
fi

step "pixi install --locked"
pixi install --locked
ok "environment built from the committed lock"

step "Version"
installed="$(pixi run simace --version | awk '{print $NF}')"
if tag="$(git describe --tags --exact-match 2>/dev/null)"; then
  if [ "$installed" = "${tag#v}" ]; then ok "simace $installed matches tag $tag"; else fail "simace $installed, tag $tag"; fi
else
  ok "simace $installed (untagged ref)"
fi

step "Quick start: simace run small_test"
if pixi run simace run small_test; then ok "small_test ran"; else fail "simace run small_test"; fi
REP="results/test/small_test/rep1"
for f in pedigree.parquet cohort.parquet report.yaml params.yaml run.yaml timing.tsv; do
  assert_file "$REP/$f"
done
assert_file "logs/test/small_test/rep1/simulate.log"
assert_file "results/test/small_test/plots/atlas.html" "HTML atlas"

rerun="$(pixi run simace run small_test 2>&1)"
skips="$(grep -c 'skip (run.yaml matches)' <<<"$rerun" || true)"
if [ "$skips" -eq 3 ]; then ok "rerun skips all 3 reps"; else fail "rerun skipped $skips of 3 reps"; fi

if pixi run simace run small_test --format pdf; then ok "PDF atlas run"; else fail "simace run small_test --format pdf"; fi
assert_file "results/test/small_test/plots/atlas.pdf" "PDF atlas"

step "Quick start: coverage_scenario and simace gather test"
if pixi run simace run coverage_scenario && pixi run simace gather test; then
  ok "folder gathered"
else
  fail "coverage_scenario or gather"
fi
assert_file "results/test/report_summary.tsv" "folder report summary"
assert_file "results/test/plots/atlas.html" "folder validation atlas"

step "Quick start: baseline100K"
if pixi run simace run baseline100K --dry-run >/dev/null; then ok "baseline100K dry run"; else fail "baseline100K --dry-run"; fi
if [ "$QUICK" = "1" ]; then
  warn "--quick: baseline100K not run"
elif pixi run simace run baseline100K --jobs 4; then
  ok "baseline100K ran"
  assert_file "results/base/baseline100K/rep1/cohort.parquet"
else
  fail "simace run baseline100K"
fi

step "Development checks"
if [ "$QUICK" = "1" ]; then
  warn "--quick: pytest not run"
elif pixi run pytest tests/ -q; then
  ok "pytest tests/"
else
  fail "pytest tests/"
fi
if pixi run ruff check; then ok "ruff check"; else fail "ruff check"; fi
if pixi run ty check; then ok "ty check"; else fail "ty check"; fi

step "The clone is still clean"
dirty="$(git status --short)"
if [ -z "$dirty" ]; then ok "git status clean"; else fail "the run changed tracked files:"$'\n'"$dirty"; fi

if [ "$LIBRARY" = "1" ]; then
  step "Library route: pip install from git"
  floor="$(grep -oE 'requires-python *= *">=3\.[0-9]+' pyproject.toml | grep -oE '3\.[0-9]+$')"
  spec="git+${SIMACE_URL%.git}@$SIMACE_REF"
  # pixi only supplies standalone interpreters; the venvs and installs are pip.
  python_for() { pixi exec --spec "python=$1" -- python -c 'import sys; print(sys.executable)'; }
  LIB="$WORK/lib"
  mkdir -p "$LIB"
  cd "$LIB"

  for below in "3.$(( ${floor#3.} - 1 ))" "3.$(( ${floor#3.} - 2 ))"; do
    "$(python_for "$below")" -m venv "venv-$below"
    if out="$("venv-$below/bin/pip" install -q "simace @ $spec" 2>&1)"; then
      fail "pip installed simace on Python $below, below requires-python >=$floor"
    else
      ok "pip refuses Python $below"
      grep -q "requires a different Python" <<<"$out" \
        || warn "pip's error on Python $below does not name the Python floor: $(tail -1 <<<"$out")"
    fi
  done

  "$(python_for "$floor")" -m venv venv
  if venv/bin/pip install -q "simace[plot] @ $spec"; then ok "pip install simace[plot] on Python $floor"; else fail "pip install on Python $floor"; fi
  lib_version="$(venv/bin/python -c 'import simace; print(simace.__version__)' || true)"
  if [ "$lib_version" = "$installed" ]; then ok "library version $lib_version"; else fail "library version '$lib_version', pixi env $installed"; fi

  if venv/bin/simace simulate --seed 1 --N 2000 --G-ped 3 --G-sim 4 --E1 0.5 --E2 0.5 \
       --output-pedigree out/pedigree.parquet \
     && venv/bin/simace cohort --pedigree out/pedigree.parquet --output-cohort out/cohort.parquet \
       --output-phenotyped-population out/phenotyped_population.yaml --seed 1 --G-pheno 2 \
       --phenotype-model1 frailty --phenotype-params1 '{distribution: weibull, scale: 2160, rho: 0.8}' \
       --phenotype-model2 frailty --phenotype-params2 '{distribution: weibull, scale: 333, rho: 1.2}'; then
    ok "simace simulate and cohort from the pip install"
  else
    fail "simace simulate or cohort from the pip install"
  fi
  if venv/bin/python -c "import polars as pl; df = pl.read_parquet('out/cohort.parquet'); assert df.height > 0; print(df.shape)"; then
    ok "cohort.parquet reads with polars"
  else
    fail "cohort.parquet unreadable or empty"
  fi

  rc=0
  out="$(venv/bin/simace run small_test --dry-run 2>&1)" || rc=$?
  if [ "$rc" -eq 2 ] && grep -q "simACE checkout" <<<"$out" && ! grep -q Traceback <<<"$out"; then
    ok "simace run outside a checkout exits 2 with a hint"
  else
    fail "simace run outside a checkout (exit $rc): $(tail -1 <<<"$out")"
  fi
fi

# EXIT trap deletes the workdir (unless --keep) and prints the PASS/FAIL summary.
