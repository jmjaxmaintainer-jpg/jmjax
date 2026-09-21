#!/usr/bin/env bash
# Run the precision benchmark in both settings and compare.
#
# Precision is fixed at the first jax import in a process, so one R session
# cannot measure both. This runs a separate process per setting.
#
#   bash dev/bench_x64.sh
#   BENCH_SIZES=200,1000,8000 BENCH_SEEDS=1,2,3 bash dev/bench_x64.sh
#
# On a Mac, keep it awake for the long ones:
#   caffeinate -i bash dev/bench_x64.sh

set -u   # not -e: a failure in one arm should still let the other report

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/bench_x64.R"
RSCRIPT="${RSCRIPT:-Rscript}"

echo "=== float64 arm ==="
JMJAX_ENABLE_X64=1 "$RSCRIPT" "$SCRIPT"
rc64=$?

echo
echo "=== float32 arm ==="
JMJAX_ENABLE_X64=0 "$RSCRIPT" "$SCRIPT"
rc32=$?

echo
if [ $rc64 -ne 0 ] || [ $rc32 -ne 0 ]; then
  echo "!! one arm exited non-zero (float64=$rc64 float32=$rc32)."
  echo "!! comparing whatever completed - treat a half-finished table with care."
fi

echo "=== comparison ==="
"$RSCRIPT" "$SCRIPT" compare
