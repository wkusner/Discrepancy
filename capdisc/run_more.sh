#!/usr/bin/env bash
# Additional w227 jobs, to run beside run_all.sh and run_msolve.sh (resumable; separate tags).
#   trackA   deeper floating-point check rungs for N=5 (to 1e-5) and N=6 (to 1e-4), then the atlas for N=4..6   MORE_CERT_THREADS (8)
#   n8cert   first N=8 floating-point rung (optional; run_all reaches N=8 later)
#   n8sys    build the 2-fold-reduced N=8 area system (sympy) and solve it with msolve  MORE_MSOLVE_THREADS (4)
#   search   candidate search for N=9..12, then identification                           MORE_SEARCH_WORKERS (4)
# Usage (inside tmux):  bash run_more.sh        JOBS="n8cert search" bash run_more.sh
set -u
cd "$(dirname "$0")"; mkdir -p results logs
PY=${PY:-$HOME/sphcap-env/bin/python}
MSOLVE=${MSOLVE:-$(command -v msolve || echo $HOME/local/bin/msolve)}
JOBS=${JOBS:-"trackA n8sys search"}
CT=${MORE_CERT_THREADS:-8}; MT=${MORE_MSOLVE_THREADS:-4}; SW=${MORE_SEARCH_WORKERS:-4}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date '+%F %T')  [more] $*" | tee -a logs/run_all.log; }
[ -x ./capcert ] || bash build.sh

job_n8cert() {
  for eta in 5e-2 2e-2; do
    T=$(python3 -c "print(repr(0.23079914617978103 - $eta))"); me=$(python3 -c "print($eta/5)")
    tag=results/early_N8_eta$eta; grep -q '^RESULT' $tag.out 2>/dev/null && continue
    log "early N=8 floating-point check eta=$eta ($CT threads)"
    timeout 48h ./capcert 8 $T $me $tag --threads $CT > $tag.out 2> $tag.log
    grep -qE '^(CERTIFIED|FLOAT-CHECKED|INTERVAL-VERIFIED)' $tag.out || { log "early N=8 stopped at eta=$eta"; break; }
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
  for N in 9 10 11 12; do
    log "search N=$N ($SW workers)"
    "$PY" jobs/search_parallel.py $N --workers $SW --rounds 12 >> logs/search_N$N.log 2>&1
    "$PY" jobs/identify.py $N 2 600 > logs/identify_N$N.log 2>&1
  done
}
job_trackA() {
  declare -A DB=( [5]=0.32332565099805260 [6]=0.27867881517624233 )
  for spec in "5 3e-5" "5 1e-5" "6 3e-4" "6 1e-4"; do
    set -- $spec; N=$1; eta=$2
    T=$(python3 -c "print(repr(${DB[$N]} - $eta))"); me=$(python3 -c "print($eta/5)")
    tag=results/ladder_N${N}_eta$eta; grep -q '^RESULT' $tag.out 2>/dev/null && continue
    log "deeper floating-point check N=$N eta=$eta ($CT threads)"
    timeout 24h ./capcert $N $T $me $tag --threads $CT > $tag.out 2> $tag.log
  done
  declare -A DELTA=( [4]=0.005 [5]=0.01 [6]=0.02 ) ; declare -A DA=( [4]=0.375 [5]=0.32332565099805260 [6]=0.27867881517624233 )
  for N in 4 5 6; do
    tag=results/atlas_N$N; grep -q '^RESULT' $tag.out 2>/dev/null && continue
    T=$(python3 -c "print(repr(${DA[$N]} + ${DELTA[$N]}))")
    log "atlas N=$N T=$T ($CT threads)"
    timeout 24h ./capcert $N $T 3e-3 $tag --threads $CT --keep-prob 1e-3 > $tag.out 2> $tag.log
    "$PY" jobs/atlas.py $N $tag --sample 4000 --workers $SW > logs/atlas_N$N.log 2>&1
  done
}
pids=()
for j in $JOBS; do job_$j & pids+=($!); done
wait "${pids[@]}"; python3 status.py; log "run_more finished"
