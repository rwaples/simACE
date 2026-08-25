#!/usr/bin/env bash
# Attribute a pipeline run's peak memory to the Snakemake rule that holds it.
#
#   bash tools/bench_memprobe.sh [SCENARIO]        # default baseline1M
#
# Why this exists: Snakemake's own benchmark TSVs sample peak RSS on an
# interval and miss transient spikes. On the 2026-08-20 baseline they reported
# 2.1 GB for `analyze` at 1M against a true peak of 9.0 GB -- a 3x
# understatement, on exactly the number you would use to size a SLURM mem_mb.
#
# This samples the process tree four times a second and reports three figures:
# the summed tree RSS (what the machine must hold), the largest single process
# (what one job needs), and a per-rule breakdown. Rule attribution is exact,
# not inferred from timing: Snakemake names each wrapper
# .snakemake/scripts/tmpXXXX.<rule>.py, so the rule is in the filename.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1

SCENARIO=${1:-baseline1M}
FOLDER=${FOLDER:-base}
CORES=${CORES:-4}
OUT=${OUT:-$ROOT/bench-logs}
mkdir -p "$OUT"
samples=$OUT/memprobe-$SCENARIO.txt
: > "$samples"

sample() {
  while :; do
    ps -eo rss=,args= 2>/dev/null | grep -F ".snakemake/scripts/tmp" | grep -v grep \
      | awk '{n = split($NF, p, "/"); print $1 "\t" p[n]}' >> "$samples"
    ps -eo rss=,args= 2>/dev/null | grep -E "snakemake|python" | grep -v grep \
      | awk '{s += $1} END {if (s) print s "\tTREE_TOTAL"}' >> "$samples"
    sleep 0.25
  done
}

sample & probe=$!
/usr/bin/time -v -o "$OUT/memprobe-$SCENARIO.time" \
  pixi run snakemake --cores "$CORES" -F \
    "results/$FOLDER/$SCENARIO/stats.done" "results/$FOLDER/$SCENARIO/plots/atlas.html" \
  > "$OUT/memprobe-$SCENARIO.log" 2>&1
status=$?
kill "$probe" 2>/dev/null; wait "$probe" 2>/dev/null

if [ "$status" -ne 0 ]; then
  echo "FAILED (exit $status) -- memory numbers below are from an incomplete run"
  tail -30 "$OUT/memprobe-$SCENARIO.log"
fi

echo
echo "=== $SCENARIO peak memory ==="
awk -F'\t' '
  $2 == "TREE_TOTAL" { if ($1 > tree) tree = $1; next }
  { rule = $2; sub(/^tmp[^.]*\./, "", rule); sub(/\.py$/, "", rule)
    if ($1 > peak[rule]) peak[rule] = $1
    if ($1 > single) single = $1 }
  END {
    printf "%-24s %8.2f GB\n", "peak summed tree RSS", tree / 1048576
    printf "%-24s %8.2f GB\n", "peak single process", single / 1048576
    print  "--- per rule ---"
    for (r in peak) printf "%-24s %8.2f GB\n", r, peak[r] / 1048576
  }' "$samples" | { head -3; tail -n +4 | sort -k2 -rn; }

echo
echo "snakemake's own sampled figures, for comparison:"
for f in "benchmarks/$FOLDER/$SCENARIO"/rep*/*.tsv; do
  [ -f "$f" ] || continue
  awk -v n="$(basename "$f" .tsv)" -F'\t' 'NR==2 {printf "%-24s %8.2f GB\n", n, $3/1024}' "$f"
done | sort -k2 -rn | head -5
exit "$status"
