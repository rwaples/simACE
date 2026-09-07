#!/usr/bin/env bash
# pytest xdist sweep: which --dist mode and thread split should the suite use?
#
# Runs the suite once per cell per sweep under /usr/bin/time -v, then reports a
# median wall time, peak RSS and per-worker load spread per cell, with a speedup
# against serial. A cell is an xdist worker count x per-worker numba/BLAS/polars
# thread count x --dist mode: serial, n6_loadscope, n6_worksteal, n6_load,
# n6_loadfile, n3t2_worksteal, n6_unpinned.
#
#   bash tools/bench_pytest_workers.sh
#   SWEEPS=5 CELLS="serial n6_loadscope n6_worksteal" bash tools/bench_pytest_workers.sh
#   PYTEST_ARGS='-m "not slow"' bash tools/bench_pytest_workers.sh
#
# Cost, both figures approximate: the full suite is ~800 s serial and ~160 s at
# six workers, so the default seven cells across three sweeps plus the warm-up
# run is on the order of two hours. PYTEST_ARGS='-m "not slow"' is the fast pass
# at ~139 s serial and brings the whole matrix under half an hour.
#
# Results land in $OUT (default benchmarks/pytest/): results.tsv holds one row
# per run, <cell>.<sweep>.time and <cell>.<sweep>.log hold the /usr/bin/time -v
# and pytest output per run, warmup.log holds the discarded warm-up. results.tsv
# accumulates across invocations; the printed summary covers only this one.
#
# The cells sweep --dist modes first because a profile of the original default
# (-n 6 --dist loadscope, pins on) put the bottleneck in scheduling rather than
# in the thread split: 1460 tests, 531.6 s of summed test time, 157.7 s actual
# wall against an 88.6 s perfect-packing floor (plus ~12.7 s of fixed startup
# and collection that no scheduler recovers), and a 1.83x per-worker load
# spread because one worker drew the 62.85 s
# tests/integration/test_ne_wf_monte_carlo.py::test_wf_monte_carlo_recovers_N on
# top of a normal share. Those figures came from a separate profiling run, not
# from this script; treat them as the motivation for these cells, not as a
# target to reproduce. That profile is why `pixi run test` now uses worksteal.
# n3t2_worksteal keeps one thread-split cell in reserve so the
# 6-core budget can still be re-divided if scheduling alone does not close the
# gap. The modes differ in how coarsely they pack: loadscope groups by class and
# loadfile by file, so a class-heavy module such as tests/analysis/test_validate.py
# (14 `class Test*` blocks over eight module-scoped fixtures) can be split across
# workers under loadscope and rebuild that setup once per worker, while loadfile
# keeps it on one worker at the cost of coarser balancing; load and worksteal
# schedule per test and rebuild the most, worksteal also migrating queued tests
# off a worker that falls behind.
#
# Deliberate choices, each of which cost a wrong measurement to learn:
#   * a discarded warm-up run precedes the sweeps, because numba compiles with
#     cache=True and the first invocation writes the .nbi/.nbc files -- folding
#     it into sweep 1 charges the first cell for the whole compile;
#   * every pytest exit status is checked and recorded, since a run that dies
#     during collection finishes fast and otherwise records a plausible but
#     meaningless *faster* time; only non-error runs enter the medians;
#   * the spread column is max/min of summed test seconds per worker, because
#     wall time alone cannot tell a scheduling fix from a lucky packing;
#   * every cell runs with -v --durations=0 --durations-min=0, since the spread
#     is parsed from those two line formats. They cost a little output on every
#     cell equally, so the wall times stay comparable to each other but not to a
#     bare `pytest` run;
#   * the test count is summed from pytest's outcome line rather than the
#     collection header, because serial prints `collected N items` while xdist
#     prints `N workers [N items]` and -m deselection changes both; two cells
#     that ran different counts are not comparable and the script says so;
#   * unpinned cells have the six thread variables explicitly unset, so an
#     invoking shell that already exports OMP_NUM_THREADS cannot silently pin
#     the cell that is supposed to be unpinned;
#   * the pins are passed through `env` into the child because numba and BLAS
#     read them at import time -- exporting them after the interpreter starts
#     has no effect;
#   * the command is `pixi run pytest`, never `pixi run test`, because that task
#     hardcodes one cell (-n 6 --dist worksteal, all six threads pinned to 1);
#   * /usr/bin/time -v reports the peak RSS of the largest single child, not the
#     sum across xdist workers, so the RSS column understates whole-box memory
#     by roughly the worker count;
#   * medians with a min-max range, never a single run.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1

SWEEPS=${SWEEPS:-3}
CELLS=${CELLS:-"serial n6_loadscope n6_worksteal n6_load n6_loadfile n3t2_worksteal n6_unpinned"}
PYTEST_ARGS=${PYTEST_ARGS:-}
OUT=${OUT:-$ROOT/benchmarks/pytest}

CELLS_TABLE="
serial          -  -  -
n6_loadscope    6  1  loadscope
n6_worksteal    6  1  worksteal
n6_load         6  1  load
n6_loadfile     6  1  loadfile
n3t2_worksteal  3  2  worksteal
n6_unpinned     6  -  worksteal
"

THREAD_VARS="OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS NUMBA_NUM_THREADS POLARS_MAX_THREADS"
BALANCE_ARGS="-v --durations=0 --durations-min=0"
FMT='%-16s %3s %7s  %-30s %11s %8s  %8s\n'

cell_row() {
  printf '%s\n' "$CELLS_TABLE" | awk -v c="$1" '$1==c {print $2, $3, $4}'
}

build_env_args() {
  local threads=$1 var
  env_args=()
  for var in $THREAD_VARS; do
    if [ "$threads" = "-" ]; then
      env_args+=(-u "$var")
    else
      env_args+=("$var=$threads")
    fi
  done
}

median() {
  sort -n | awk '{a[NR]=$1} END{if (NR) print a[int((NR+1)/2)]}'
}

worker_spread() {
  awk '
    /^[0-9]+\.[0-9]+s +(call|setup|teardown) +/ { cost[$3] += $1 + 0; next }
    /^\[gw[0-9]+\] \[/ {
      match($0, /^\[gw[0-9]+\]/)
      gw = substr($0, 2, RLENGTH - 2)
      rest = substr($0, RLENGTH + 1)
      sub(/^[ \t]*\[[^]]*\][ \t]*[A-Za-z]+[ \t]+/, "", rest)
      split(rest, f, /[ \t]/)
      worker[f[1]] = gw
    }
    END {
      for (node in cost) if (node in worker) load[worker[node]] += cost[node]
      n = 0
      for (gw in load) {
        if (n++ == 0) { lo = hi = load[gw] }
        if (load[gw] < lo) lo = load[gw]
        if (load[gw] > hi) hi = load[gw]
      }
      if (n > 1 && lo > 0) printf "%.2f\n", hi / lo; else print "-"
    }
  ' "$1"
}

for cell in $CELLS; do
  [ -n "$(cell_row "$cell")" ] && continue
  echo "unknown cell '$cell'" >&2
  echo "valid cells: $(printf '%s\n' "$CELLS_TABLE" | awk 'NF{printf "%s ", $1}')" >&2
  exit 2
done

PYTEST_ARGV=()
if [ -n "$PYTEST_ARGS" ]; then
  mapfile -t PYTEST_ARGV < <(printf '%s' "$PYTEST_ARGS" | xargs -n1 printf '%s\n')
fi
read -r -a BALANCE_ARGV <<< "$BALANCE_ARGS"

mkdir -p "$OUT"
SUMMARY=$OUT/results.tsv
[ -f "$SUMMARY" ] || printf 'cell\tsweep\tworkers\tthreads\tdist\twall_s\tuser_s\tsys_s\tcpu_pct\tmax_rss_mib\ttests\tspread\tstatus\n' > "$SUMMARY"

RUN_ROWS=$(mktemp) || exit 1
trap 'rm -f "$RUN_ROWS"' EXIT

echo "[$(date +%T)] WARMUP serial, unpinned, discarded"
build_env_args -
env "${env_args[@]}" pixi run pytest "${PYTEST_ARGV[@]}" > "$OUT/warmup.log" 2>&1
status=$?
if [ "$status" -gt 1 ]; then  # 1 is a test failure, which still warmed the cache
  echo "[$(date +%T)] WARMUP failed (exit $status) -- fix the environment first" >&2
  tail -40 "$OUT/warmup.log" >&2
  exit "$status"
fi

for sweep in $(seq 1 "$SWEEPS"); do
  for cell in $CELLS; do
    read -r workers threads dist <<< "$(cell_row "$cell")"
    run_log=$OUT/$cell.$sweep.log
    time_log=$OUT/$cell.$sweep.time

    xdist_args=()
    [ "$workers" = "-" ] || xdist_args=(-n "$workers" --dist "$dist")
    build_env_args "$threads"

    echo "[$(date +%T)] START $cell sweep $sweep"
    /usr/bin/time -v -o "$time_log" \
      env "${env_args[@]}" pixi run pytest \
        "${xdist_args[@]}" "${BALANCE_ARGV[@]}" "${PYTEST_ARGV[@]}" \
      > "$run_log" 2>&1
    status=$?

    case $status in
      0) state=ok ;;
      1) state=tests_failed ;;
      *) state=error:$status ;;
    esac

    wall=$(awk -F': ' '/Elapsed \(wall clock\)/{print $NF}' "$time_log" |
      awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; printf "%.2f\n", s}')  # h:mm:ss, not seconds
    user=$(awk -F': ' '/User time/{print $NF}' "$time_log")
    sys=$(awk -F': ' '/System time/{print $NF}' "$time_log")
    cpu=$(awk -F': ' '/Percent of CPU/{print $NF}' "$time_log")
    rss=$(awk -F': ' '/Maximum resident set size/{print $NF}' "$time_log" |
      awk '{printf "%.1f\n", $1/1024}')

    summary=$(grep -E '^=+ .*(passed|failed|error|no tests ran)' "$run_log" | tail -1)
    tests=$(printf '%s\n' "$summary" |
      grep -Eo '[0-9]+ (passed|failed|skipped|xfailed|xpassed|errors?)' | awk '{s+=$1} END{print s+0}')
    spread=$(worker_spread "$run_log")

    row=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "$cell" "$sweep" "$workers" "$threads" "$dist" \
      "$wall" "$user" "$sys" "$cpu" "$rss" "$tests" "$spread" "$state")
    printf '%s\n' "$row" >> "$SUMMARY"
    printf '%s\n' "$row" >> "$RUN_ROWS"
    echo "[$(date +%T)] DONE  $cell sweep $sweep  wall=${wall}s rss=${rss}MiB cpu=$cpu tests=$tests spread=$spread $state"
  done
done

usable_rows() {  # no argument: every cell
  awk -F'\t' -v c="${1:-}" '($13=="ok" || $13=="tests_failed") && (c=="" || $1==c)' "$RUN_ROWS"
}

serial_med=$(usable_rows serial | cut -f6 | median)

table=()
for cell in $CELLS; do
  n=$(usable_rows "$cell" | awk 'END{print NR+0}')
  if [ "$n" -eq 0 ]; then
    table+=("999999999"$'\t'"$(printf "$FMT" "$cell" "0" "-" "-" "-" "-" "-")")
    continue
  fi
  walls=$(usable_rows "$cell" | cut -f6)
  med_wall=$(printf '%s\n' "$walls" | median)
  min_wall=$(printf '%s\n' "$walls" | sort -n | head -1)
  max_wall=$(printf '%s\n' "$walls" | sort -n | tail -1)
  med_rss=$(usable_rows "$cell" | cut -f10 | median)
  cell_tests=$(usable_rows "$cell" | cut -f11 | sort -un | paste -sd/ -)
  med_spread=$(usable_rows "$cell" | cut -f12 | grep -v '^-$' | median)
  if [ -n "$serial_med" ]; then
    speedup=$(awk -v s="$serial_med" -v m="$med_wall" 'BEGIN{if (m>0) printf "%.2fx\n", s/m; else print "-"}')
  else
    speedup=-
  fi
  line=$(printf "$FMT" "$cell" "$n" "$cell_tests" \
    "$(printf '%s (%s-%s)' "$med_wall" "$min_wall" "$max_wall")" \
    "$med_rss" "${med_spread:--}" "$speedup")
  table+=("$med_wall"$'\t'"$line")
done

echo
echo "=== $SWEEPS sweep(s) x $(printf '%s\n' "$CELLS" | wc -w) cell(s), this invocation ==="
printf "$FMT" cell n tests "wall_s median (min-max)" "rss_mib" "spread" "speedup"
printf '%s\n' "${table[@]}" | sort -t$'\t' -k1,1n | cut -f2-

bad=$(awk -F'\t' '$13!="ok"{printf "  %s sweep %s  %s\n", $1, $2, $13}' "$RUN_ROWS")
if [ -n "$bad" ]; then
  echo
  echo "RUNS THAT DID NOT EXIT CLEAN:"
  printf '%s\n' "$bad"
fi

if [ "$(usable_rows | cut -f11 | sort -u | wc -l)" -gt 1 ]; then
  echo
  echo "######################################################################"
  echo "# INVALID COMPARISON: the cells did not run the same tests.          #"
  echo "# Wall times across different test counts mean nothing. Test counts: #"
  echo "######################################################################"
  usable_rows | awk -F'\t' '{c[$1]=c[$1] " " $11} END{for (k in c) printf "  %-16s%s\n", k, c[k]}' | sort
fi

echo
echo "[$(date +%T)] rows appended to $SUMMARY"
