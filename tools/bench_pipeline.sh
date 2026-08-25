#!/usr/bin/env bash
# Whole-pipeline wall-clock + peak-RSS benchmark over one or more scenarios.
#
# Runs simulate -> phenotype -> censor -> ascertainment -> analyze -> plot for
# each scenario, N times, forcing full re-execution every sweep. Sweep 1 is
# cold (numba compiles); later sweeps are warm. Compare warm to warm.
#
#   bash tools/bench_pipeline.sh
#   SWEEPS=5 CORES=8 SCENARIOS="baseline10K baseline1M" bash tools/bench_pipeline.sh
#
# Results land in $OUT (default bench-logs/): wallclock.tsv holds one row per
# run; tsv/sweepN/ snapshots the per-rule benchmark TSVs before the next sweep
# overwrites them; runs/ holds the snakemake log, /usr/bin/time -v output and
# CPU-clock samples per run.
#
# Deliberate choices, each of which cost a wrong measurement to learn:
#   * every snakemake exit status is checked -- a failed run otherwise records
#     a plausible but meaningless *faster* time;
#   * per-rule TSVs are snapshotted per sweep, because snakemake overwrites
#     them in place and you would otherwise aggregate n=1;
#   * the CPU clock is sampled at its FASTEST core, since a firmware power
#     clamp pins every core low and silently doubles wall times;
#   * the target is per-scenario (stats.done + the scenario atlas), NOT
#     scenario.done -- that one depends on the folder-wide report_summary.tsv
#     and will drag every other scenario in the folder into the DAG.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1

SWEEPS=${SWEEPS:-3}
CORES=${CORES:-4}
FOLDER=${FOLDER:-base}
SCENARIOS=${SCENARIOS:-"baseline10K baseline100K baseline1M"}
OUT=${OUT:-$ROOT/bench-logs}

mkdir -p "$OUT/tsv" "$OUT/runs"
SUMMARY=$OUT/wallclock.tsv
[ -f "$SUMMARY" ] || printf 'sweep\tscenario\twall_s\tmax_rss_kb\tcpu_pct\tfreq_max_khz\tfreq_med_khz\tstatus\n' > "$SUMMARY"

sample_freq() {  # $1 = output file. Fastest core: a clamp pins all cores low.
  while :; do
    sort -n /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | tail -1
    sleep 2
  done > "$1"
}

for sweep in $(seq 1 "$SWEEPS"); do
  for sc in $SCENARIOS; do
    tag="sweep${sweep}-${sc}"
    run_log=$OUT/runs/$tag.log
    freq_log=$OUT/runs/$tag.freq
    time_log=$OUT/runs/$tag.time

    echo "[$(date +%T)] START $tag"
    sample_freq "$freq_log" &
    freq_pid=$!

    /usr/bin/time -v -o "$time_log" \
      pixi run snakemake --cores "$CORES" -F \
        "results/$FOLDER/$sc/stats.done" "results/$FOLDER/$sc/plots/atlas.html" \
      > "$run_log" 2>&1
    status=$?

    kill "$freq_pid" 2>/dev/null; wait "$freq_pid" 2>/dev/null

    wall=$(awk -F': ' '/Elapsed \(wall clock\)/{print $NF}' "$time_log")
    rss=$(awk -F': ' '/Maximum resident set size/{print $NF}' "$time_log")
    cpu=$(awk -F': ' '/Percent of CPU/{print $NF}' "$time_log")
    fmax=$(sort -n "$freq_log" 2>/dev/null | tail -1)
    fmed=$(sort -n "$freq_log" 2>/dev/null | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')

    if [ "$status" -ne 0 ]; then
      echo "[$(date +%T)] FAIL $tag (exit $status) -- aborting"
      tail -40 "$run_log"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tFAILED\n' \
        "$sweep" "$sc" "$wall" "$rss" "$cpu" "$fmax" "$fmed" >> "$SUMMARY"
      exit "$status"
    fi

    dest=$OUT/tsv/sweep${sweep}/$sc
    mkdir -p "$dest"
    cp -r "benchmarks/$FOLDER/$sc/." "$dest/" 2>/dev/null

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tOK\n' \
      "$sweep" "$sc" "$wall" "$rss" "$cpu" "$fmax" "$fmed" >> "$SUMMARY"
    echo "[$(date +%T)] DONE  $tag  wall=$wall rss=${rss}KB cpu=$cpu freq_max=${fmax}kHz"
  done
done
echo "[$(date +%T)] ALL SWEEPS COMPLETE -- aggregate with: python tools/bench_aggregate.py $OUT"
