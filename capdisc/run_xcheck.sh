#!/usr/bin/env bash
# Second-kernel pipeline (VERIFICATION.md), run on w227 inside tmux:
#   local and symmetric Krawczyk tests in mpmath.iv, root isolation in sympy, elimination in msolve,
#   then pipeline B for the global part: capcert_arb (Arb ball arithmetic) + xcheck_refine (mpmath.iv).
#   THREADS=32 NS="5 6" bash run_xcheck.sh          output: results/xproof_N*, logs/xcheck.log
set -u
cd "$(dirname "$0")"
THREADS=${THREADS:-32}; NS=${NS:-"5 6"}
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
MSOLVE=${MSOLVE:-$(command -v msolve || echo $HOME/local/bin/msolve)}
mkdir -p results logs
[ -x capcert_arb ] || bash build_arb.sh
log() { echo "$(date '+%F %T')  $*" | tee -a logs/xcheck.log; }
$PY jobs/xcheck_roots.py 2>&1 | tee -a logs/xcheck.log
$PY jobs/xcheck_elim.py "$MSOLVE" 2>&1 | tee -a logs/xcheck.log
declare -A EPS=([5]=0.04 [6]=0.03)
for N in $NS; do
  nm=N${N}_best
  log "N=$N: local and symmetric Krawczyk tests in mpmath.iv"
  $PY jobs/xcheck_local.py $nm 2>&1 | tail -1 | tee -a logs/xcheck.log
  nimg=$($PY -c "import json;print(len(json.load(open('certificates/localcert_$nm.json'))['images']))")
  for k in $(seq 0 $((nimg - 1))); do $PY jobs/xcheck_sym.py $nm $k 2>&1 | tail -1 | tee -a logs/xcheck.log; done
  T=$($PY -c "import json;d=json.load(open('certificates/localcert_$nm.json'));print(repr(max(i['D_enclosure'][1] for i in d['images'])))")
  log "N=$N: capcert_arb T=$T MIN_EPS=${EPS[$N]} ($THREADS threads)"
  rm -f results/xproof_N$N.surv.* results/xproof_N$N.done
  ./capcert_arb $N $T ${EPS[$N]} results/xproof_N$N --threads $THREADS > results/xproof_N$N.out 2> logs/xproof_capcert_N$N.log
  cat results/xproof_N$N.out | tee -a logs/xcheck.log
  [ -f results/xproof_N$N.done ] || { log "capcert_arb did not complete for N=$N"; continue; }
  log "N=$N: refinement in mpmath.iv"
  $PY jobs/xcheck_refine.py $N results/xproof_N$N --workers $THREADS 2>&1 | tee logs/xrefine_N$N.log | tail -1 | tee -a logs/xcheck.log
done
log "run_xcheck finished"
