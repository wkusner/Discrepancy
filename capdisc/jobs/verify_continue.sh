#!/usr/bin/env bash
# jobs/verify_continue.sh N  --  continue a verify.sh run whose first mesh pass was started by an older verify.sh
# (one without the second pass, Step 2b).  Used once, for N = 7 (first pass started at 768cefc, 2026-10-31).
#
# 1. waits for results/verify_N<N>/mesh.done (the running first pass writes it when it completes);
# 2. stops the old script's Python refinement of the first-pass survivors as soon as it starts (it would read
#    ~10^9 cells into memory) -- this script polls every 10 s;
# 3. re-runs Step 4 (local verification, seconds) at the current commit, so that every Python certificate of the
#    chain comes from one commit (chain check C8), and checks that it gives the threshold of the first pass;
# 4. runs Step 2b (second mesh pass), Step 3 (refinement) and the chain checker, logging to verify.log.
# The first-pass binary was built from an older commit; the rigorous code path it ran (cell bounds, first-order
# bound, splitting) is unchanged since (git diff 768cefc -- csrc/ only adds --cells), which verify.log records.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:?usage: jobs/verify_continue.sh N}
THREADS=${THREADS:-4}
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
declare -A NAME=([7]=N7_best) RBOX=([7]=1e-4) EPS2=([7]=0.0078125)
nm=${NAME[$N]}; OUT=results/verify_N$N
MESH_OPTS=${MESH_OPTS---first-order --fo-emax 0.1}
log() { echo "$(date '+%F %T')  $*" | tee -a "$OUT/verify.log"; }
until [ -f "$OUT/mesh.done" ]; do sleep 10; done
sleep 2; bash build.sh > "$OUT/build_continue.log" 2>&1   # the first pass has exited; rebuild before anything else starts
for i in $(seq 1 30); do pkill -f "[p]roof.refine $N " && break; sleep 10; done || true
sleep 5; pkill -9 -f "[p]roof.refine $N " || true
log "continuing with jobs/verify_continue.sh at commit $(git rev-parse --short HEAD); first pass: $(cat $OUT/mesh.done)"
log "csrc changes since the first pass was built: $(git diff --stat 768cefc HEAD -- csrc | tail -1)"
log "Step 4 (again, at this commit): local verification"
$PY -m proof.local "$nm" "${RBOX[$N]}" --out "$OUT" 2>&1 | tee -a "$OUT/verify.log"
T=$($PY -c "import json,math;d=json.load(open('$OUT/localcert_$nm.json'));print(repr(math.nextafter(max(i['D_enclosure'][1] for i in d['images']),math.inf)))")
$PY -c "s=dict(kv.split('=',1) for kv in open('$OUT/mesh.done').read().split()); assert float.fromhex(s['T'])==$T, 'threshold changed'"
rm -f "$OUT"/mesh.refine.json
log "Step 2b: second mesh pass on the survivors, MIN_EPS=${EPS2[$N]}"
tail -1 "$OUT/build_continue.log" | tee -a "$OUT/verify.log"
./capcert_rig "$N" "$T" "${EPS2[$N]}" "$OUT/mesh2" --threads "$THREADS" --cells "$OUT/mesh" $MESH_OPTS > "$OUT/mesh2.out" 2> "$OUT/mesh2.err"
cat "$OUT/mesh2.out" | tee -a "$OUT/verify.log"
[ -f "$OUT/mesh2.done" ] || { log "second mesh pass did not complete"; exit 1; }
log "Step 3: refinement"
$PY -m proof.refine "$N" "$OUT/mesh2" "$OUT/localcert_$nm.json" --workers "$THREADS" --min-width 1e-7 --T "$T" > "$OUT/refine.log" 2>&1 || true
tail -2 "$OUT/refine.log" | tee -a "$OUT/verify.log"
$PY -c "import json;d=json.load(open('$OUT/mesh2.refine.json'));assert d['verified'] and d['failed']==0" \
    && log "pipeline A, N=$N: all steps verified (level: verified)" || { log "pipeline A, N=$N: NOT VERIFIED"; exit 1; }
python3 check/check_chain.py "$N" "$OUT" 2>&1 | tee -a "$OUT/verify.log"
