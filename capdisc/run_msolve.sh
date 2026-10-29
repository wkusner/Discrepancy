#!/usr/bin/env bash
# Run msolve on every system in msolve/ that has no result yet (standalone; safe alongside run_all.sh).
set -u
cd "$(dirname "$0")"; mkdir -p results logs
MSOLVE=${MSOLVE:-$(command -v msolve || echo $HOME/local/bin/msolve)}
T=${MSOLVE_THREADS:-8}
[ -x "$MSOLVE" ] || { echo "msolve not found (run build_msolve.sh)"; exit 1; }
for f in msolve/*.ms; do
  b=$(basename $f .ms); [ -s results/$b.msolve.out ] && continue
  echo "$(date '+%F %T')  msolve $b ($T threads)" | tee -a logs/run_all.log
  ( time "$MSOLVE" -f $f -o results/$b.msolve.out -P 1 -t $T -v 1 ) > logs/msolve_$b.log 2>&1
  echo "$(date '+%F %T')  msolve $b finished" | tee -a logs/run_all.log
done
