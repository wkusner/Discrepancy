#!/usr/bin/env bash
# Additional w227 jobs, to run beside run_all.sh and run_msolve.sh (resumable; separate tags).
#   n8cert   first N=8 certification rung (13-dim feasibility), tag results/early_N8_*   MORE_CERT_THREADS (8)
#   n8sys    build the 2-fold-reduced N=8 area system (sympy) and solve it with msolve  MORE_MSOLVE_THREADS (4)
#   search   candidate search for N=9 and N=10, then identification                     MORE_SEARCH_WORKERS (4)
# Usage (inside tmux):  bash run_more.sh        JOBS="n8cert search" bash run_more.sh
set -u
cd "$(dirname "$0")"; mkdir -p results logs
PY=${PY:-$HOME/sphcap-env/bin/python}
MSOLVE=${MSOLVE:-$(command -v msolve || echo $HOME/local/bin/msolve)}
JOBS=${JOBS:-"n8cert n8sys search"}
CT=${MORE_CERT_THREADS:-8}; MT=${MORE_MSOLVE_THREADS:-4}; SW=${MORE_SEARCH_WORKERS:-4}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date '+%F %T')  [more] $*" | tee -a logs/run_all.log; }
[ -x ./capcert ] || bash build.sh

job_n8cert() {
  for eta in 5e-2 2e-2; do
    T=$(python3 -c "print(repr(0.23079914617978103 - $eta))"); me=$(python3 -c "print($eta/5)")
    tag=results/early_N8_eta$eta; grep -q '^RESULT' $tag.out 2>/dev/null && continue
    log "early N=8 certification eta=$eta ($CT threads)"
    timeout 48h ./capcert 8 $T $me $tag --threads $CT > $tag.out 2> $tag.log
    grep -q '^CERTIFIED' $tag.out || { log "early N=8 stopped at eta=$eta"; break; }
  done
}
job_n8sys() {
  [ -s msolve/N8_C2.ms ] || { log "building N8_C2.ms"; "$PY" jobs/make_msolve_c2.py N8_best msolve/N8_C2.ms > logs/make_N8_C2.log 2>&1; }
  [ -s msolve/N8_C2.ms ] || { log "N8 system build failed (logs/make_N8_C2.log)"; return; }
  [ -s results/N8_C2.msolve.out ] && return
  log "msolve N8_C2 ($MT threads)"
  ( time "$MSOLVE" -f msolve/N8_C2.ms -o results/N8_C2.msolve.out -P 1 -t $MT -v 1 ) > logs/msolve_N8_C2.log 2>&1
  log "msolve N8_C2 finished"
}
job_search() {
  for N in 9 10; do
    log "search N=$N ($SW workers)"
    "$PY" jobs/search_parallel.py $N --workers $SW --rounds 15 >> logs/search_N$N.log 2>&1
    "$PY" jobs/identify.py $N 2 600 > logs/identify_N$N.log 2>&1
  done
}
pids=()
for j in $JOBS; do job_$j & pids+=($!); done
wait "${pids[@]}"; python3 status.py; log "run_more finished"
