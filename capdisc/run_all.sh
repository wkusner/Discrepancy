#!/usr/bin/env bash
# =============================================================================
# capdisc orchestrator for w227: N = 3..8, all phases, with a thread budget.
#
#   ladder   C certifier: D >= D_best - eta for a decreasing ladder of eta     (CERT_THREADS)
#   search   parallel Python search for minimiser candidates                    (SEARCH_WORKERS)
#   identify high-precision polish + integer relations for the best classes    (after search)
#   msolve   exact elimination of symmetry-reduced systems (if msolve present) (MSOLVE_THREADS)
#   atlas    optional (ATLAS=1): sublevel-set survivors + descent, N <= 6        (after ladder)
#   status   STATUS.md every STATUS_EVERY seconds; pushed to GitHub if PUSH=1
#
# Usage (inside tmux):   bash run_all.sh
# Knobs (environment):   NS="3 4 5 6 7 8"  CERT_THREADS=32  SEARCH_WORKERS=12  MSOLVE_THREADS=8
#                        STATUS_EVERY=600  PUSH=0  ATLAS=0  PY=~/sphcap-env/bin/python
# Everything is resumable: rerunning skips finished work.
# =============================================================================
set -u
cd "$(dirname "$0")"
NS=${NS:-"3 4 5 6 7 8"}
CERT_THREADS=${CERT_THREADS:-32}
SEARCH_WORKERS=${SEARCH_WORKERS:-12}
MSOLVE_THREADS=${MSOLVE_THREADS:-8}
STATUS_EVERY=${STATUS_EVERY:-600}
PUSH=${PUSH:-0}
ATLAS=${ATLAS:-0}
PY=${PY:-$HOME/sphcap-env/bin/python}
MSOLVE=${MSOLVE:-$HOME/sphcap-env/bin/msolve}
mkdir -p results logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # Python workers stay single-threaded

log() { echo "$(date '+%F %T')  $*" | tee -a logs/run_all.log; }

[ -x ./capcert ] || bash build.sh || { log "build failed"; exit 1; }
HAVE_PY=0; "$PY" -c "import numpy, scipy, mpmath, flint" 2>/dev/null && HAVE_PY=1
HAVE_MSOLVE=0; [ -x "$MSOLVE" ] && HAVE_MSOLVE=1
log "start: NS=[$NS] cert=$CERT_THREADS search=$SEARCH_WORKERS msolve=$MSOLVE_THREADS python=$HAVE_PY msolve=$HAVE_MSOLVE push=$PUSH"

# ---------------------------------------------------------------- best known values
declare -A DBEST=( [3]=0.5 [4]=0.375 [5]=0.32332565099805260 [6]=0.27867881517624233
                   [7]=0.26085880727033208 [8]=0.23079914617978103 )
declare -A LADDER=( [3]="1e-3 1e-5" [4]="1e-3 1e-4 1e-5 1e-6" [5]="1e-2 3e-3 1e-3 3e-4 1e-4"
                    [6]="2e-2 1e-2 3e-3 1e-3" [7]="3e-2 1e-2 3e-3" [8]="5e-2 2e-2 1e-2" )
declare -A ROUNDS=( [3]=2 [4]=2 [5]=4 [6]=6 [7]=20 [8]=40 )

# ---------------------------------------------------------------- phases
phase_ladder() {
  for N in $NS; do
    for eta in ${LADDER[$N]}; do
      T=$(python3 -c "print(repr(${DBEST[$N]} - $eta))"); me=$(python3 -c "print($eta/5)")
      tag=results/ladder_N${N}_eta${eta}
      grep -q '^RESULT' $tag.out 2>/dev/null && continue
      log "ladder N=$N eta=$eta T=$T"
      timeout ${LADDER_TIMEOUT:-12h} ./capcert $N $T $me $tag --threads $CERT_THREADS > $tag.out 2> $tag.log
      grep -q '^CERTIFIED' $tag.out || { log "ladder N=$N stopped at eta=$eta"; break; }
    done
  done
  log "ladder phase done"
}

phase_search_identify() {
  [ $HAVE_PY = 1 ] || { log "search skipped: no python env (run setup_env.sh)"; return; }
  for N in $NS; do
    log "search N=$N rounds=${ROUNDS[$N]}"
    "$PY" jobs/search_parallel.py $N --workers $SEARCH_WORKERS --rounds ${ROUNDS[$N]} >> logs/search_N$N.log 2>&1
    [ $N -ge 5 ] && "$PY" jobs/identify.py $N 3 600 > logs/identify_N$N.log 2>&1
  done
  for N in $NS; do
    [ $N -ge 7 ] || continue
    [ -f results/minpoly_N$N.json ] && continue
    log "deep identify N=$N"
    "$PY" jobs/deep_identify.py $N 4000 120,160 > logs/deep_identify_N$N.log 2>&1
  done
  log "search/identify phase done"
}

phase_msolve() {
  [ $HAVE_MSOLVE = 1 ] || { log "msolve skipped: not installed (setup_env.sh route A provides it)"; return; }
  for f in msolve/*.ms; do
    b=$(basename $f .ms); [ -s results/$b.msolve.out ] && continue
    log "msolve $b"
    ( time "$MSOLVE" -f $f -o results/$b.msolve.out -P 1 -t $MSOLVE_THREADS -v 1 ) > logs/msolve_$b.log 2>&1
  done
  log "msolve phase done"
}

phase_atlas() {
  [ $ATLAS = 1 ] && [ $HAVE_PY = 1 ] || return
  declare -A DELTA=( [3]=0.02 [4]=0.005 [5]=0.01 [6]=0.02 )
  for N in 3 4 5 6; do
    tag=results/atlas_N$N; grep -q '^RESULT' $tag.out 2>/dev/null && continue
    T=$(python3 -c "print(repr(${DBEST[$N]} + ${DELTA[$N]}))")
    log "atlas N=$N T=$T"
    ./capcert $N $T 3e-3 $tag --threads $CERT_THREADS --keep-prob 1e-3 > $tag.out 2> $tag.log
    "$PY" jobs/atlas.py $N $tag --sample 4000 --workers $SEARCH_WORKERS > logs/atlas_N$N.log 2>&1
  done
}

status_loop() {
  while true; do
    python3 status.py > /dev/null 2>&1
    [ $PUSH = 1 ] && bash push_status.sh >> logs/push.log 2>&1
    sleep $STATUS_EVERY
  done
}

# ---------------------------------------------------------------- run
status_loop & SPID=$!
( phase_ladder; phase_atlas ) & P1=$!
phase_search_identify & P2=$!
phase_msolve & P3=$!
wait $P1 $P2 $P3
python3 status.py; [ $PUSH = 1 ] && bash push_status.sh >> logs/push.log 2>&1
kill $SPID 2>/dev/null
log "all phases finished"
