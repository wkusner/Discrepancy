#!/usr/bin/env bash
# Second-kernel pipeline B (VERIFICATION.md), run on w227 inside tmux:
#   local and symmetric Krawczyk tests in mpmath.iv, root isolation in sympy, elimination in msolve,
#   mesh in Arb (capcert_arb) and refinement of its survivors in mpmath.iv (xcheck_refine.py);
#   finally jobs/xcheck_verdict.py states, per N, whether pipeline B proves Theorem 10 on its own.
#   THREADS=32 NS="5 6" bash run_xcheck.sh          output: results/xproof_N*, logs/xcheck.log
set -u
cd "$(dirname "$0")"
THREADS=${THREADS:-32}; NS=${NS:-"5 6"}
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
MSOLVE=${MSOLVE:-$(command -v msolve || echo $HOME/local/bin/msolve)}
mkdir -p results logs
log() { echo "$(date '+%F %T')  $*" | tee -a logs/xcheck.log; }
step() { local what=$1; shift; "$@" > logs/xcheck_step.out 2>&1; local rc=$?; tail -2 logs/xcheck_step.out | tee -a logs/xcheck.log
         [ $rc -eq 0 ] || log "FAILED: $what (exit $rc)"; return $rc; }
[ -x capcert_arb ] || bash build_arb.sh || { log "could not build capcert_arb"; exit 1; }
rm -f certificates/xcheck_*.json                     # never pass on stale results
step roots "$PY" jobs/xcheck_roots.py
step elimination "$PY" jobs/xcheck_elim.py "$MSOLVE"
declare -A EPS=([5]=0.04 [6]=0.03)
for N in $NS; do
  nm=N${N}_best
  log "N=$N: local and symmetric Krawczyk tests in mpmath.iv"
  step "local N=$N" "$PY" jobs/xcheck_local.py $nm || continue
  nimg=$("$PY" -c "import json;print(len(json.load(open('certificates/localcert_$nm.json'))['images']))")
  ok=1; for k in $(seq 0 $((nimg - 1))); do step "symmetric N=$N image $k" "$PY" jobs/xcheck_sym.py $nm $k || ok=0; done
  [ $ok = 1 ] || continue
  T=$("$PY" -c "import json;d=json.load(open('certificates/localcert_$nm.json'));print(repr(max(i['D_enclosure'][1] for i in d['images'])))")
  log "N=$N: capcert_arb T=$T MIN_EPS=${EPS[$N]} ($THREADS threads)"
  rm -f results/xproof_N$N.surv.* results/xproof_N$N.done results/xproof_N$N.xrefine.json
  ./capcert_arb $N $T ${EPS[$N]} results/xproof_N$N --threads $THREADS > results/xproof_N$N.out 2> logs/xproof_capcert_N$N.log
  rc=$?; cat results/xproof_N$N.out | tee -a logs/xcheck.log
  [ $rc -eq 0 ] && [ -f results/xproof_N$N.done ] || { log "capcert_arb did not complete for N=$N"; continue; }
  log "N=$N: refinement in mpmath.iv"
  "$PY" jobs/xcheck_refine.py $N results/xproof_N$N --workers $THREADS > logs/xrefine_N$N.log 2>&1
  tail -1 logs/xrefine_N$N.log | tee -a logs/xcheck.log
done
"$PY" jobs/xcheck_verdict.py 2>&1 | tee -a logs/xcheck.log
log "run_xcheck finished"
