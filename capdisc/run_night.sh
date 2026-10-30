#!/usr/bin/env bash
# Overnight jobs beside run_xcheck.sh and the atlas (inside tmux):
#   probe7  feasibility probe for an N=7 proof: capcert_rig (interval-verified) at T = upper end of the
#           Krawczyk enclosure of D*_7 (certificates/localcert_N7_best.json), cells down to MIN_EPS,
#           keeping every survivor; stops after PROBE_HOURS.  Output: results/probe_N7.*
#   search  N=9..12 searches again, now saving coordinates and worst caps (for the atlas/website)
#   THREADS=24 SW=8 bash run_night.sh            JOBS="probe7 search" by default
set -u
cd "$(dirname "$0")"
THREADS=${THREADS:-24}; SW=${SW:-8}; JOBS=${JOBS:-"probe7 search"}; PROBE_HOURS=${PROBE_HOURS:-11}; EPS7=${EPS7:-0.05}
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
mkdir -p results logs
log() { echo "$(date '+%F %T')  [night] $*" | tee -a logs/run_all.log; }
[ -x capcert_rig ] || bash build.sh
job_probe7() {
  T=$("$PY" -c "import json;d=json.load(open('certificates/localcert_N7_best.json'));print(repr(max(i['D_enclosure'][1] for i in d['images'])))")
  log "probe N=7: capcert_rig T=$T MIN_EPS=$EPS7 ($THREADS threads, at most $PROBE_HOURS h)"
  timeout ${PROBE_HOURS}h ./capcert_rig 7 "$T" "$EPS7" results/probe_N7 --threads "$THREADS" > results/probe_N7.out 2> logs/probe_N7.log
  log "probe N=7 ended: $(head -1 results/probe_N7.out 2>/dev/null) | last progress: $(tail -1 logs/probe_N7.log)"
}
job_search() {
  for N in 9 10 11 12; do
    log "search N=$N with coordinates ($SW workers)"
    "$PY" jobs/search_parallel.py $N --workers $SW --rounds 12 >> logs/search_N$N.log 2>&1
  done
}
pids=(); for j in $JOBS; do job_$j & pids+=($!); done
wait "${pids[@]}"; log "run_night finished"
