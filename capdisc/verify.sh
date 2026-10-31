#!/usr/bin/env bash
# verify.sh N  --  pipeline A of the N = 5, 6 proof in proof mode (PROOF.md Steps 1-6), with the trusted core proof/.
#
#   THREADS=16 bash verify.sh 5          outputs in results/verify_N5/ ; nothing is resumed or reused
#
# Steps: local verification at every image (Step 4) -> symmetric verification per image (Step 5) -> congruence
# (Step 6) -> mesh capcert_rig at T = largest upper end of the D* enclosures (Step 2) -> refinement with the
# local boxes excluded (Step 3).  Exact data (Step 7) and pipeline B are separate (jobs/gram_verify.py, run_xcheck.sh).
# The chain checker (check/check_chain.py, to come) then checks that the certificates fit together.
set -euo pipefail
cd "$(dirname "$0")"
N=${1:?usage: verify.sh N}
THREADS=${THREADS:-4}
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
declare -A NAME=([5]=N5_best [6]=N6_best) RBOX=([5]=3e-4 [6]=1e-4) EPS=([5]=0.04 [6]=0.02)
nm=${NAME[$N]}; OUT=results/verify_N$N
rm -rf "$OUT"; mkdir -p "$OUT"                     # proof mode: start clean
log() { echo "$(date '+%F %T')  $*" | tee -a "$OUT/verify.log"; }
log "verify N=$N ($nm), commit $(git rev-parse --short HEAD)$(git diff --quiet -- proof csrc || echo ' (proof/ or csrc/ modified)')"

log "Step 4: local verification"
$PY -m proof.local "$nm" "${RBOX[$N]}" --out "$OUT" 2>&1 | tee -a "$OUT/verify.log"
nimg=$($PY -c "import json;d=json.load(open('$OUT/localcert_$nm.json'));assert d['verified'];print(len(d['images']))")

log "Step 5: symmetric verification ($nimg images)"
for k in $(seq 0 $((nimg - 1))); do $PY -m proof.symmetric "$nm" "$k" --dir "$OUT" 2>&1 | tail -2 | tee -a "$OUT/verify.log"; done

log "Step 6: congruence"
$PY -m proof.congruence "$nm" --dir "$OUT" 2>&1 | tee -a "$OUT/verify.log"

T=$($PY -c "import json;d=json.load(open('$OUT/localcert_$nm.json'));print(repr(max(i['D_enclosure'][1] for i in d['images'])))")
log "Step 2: mesh capcert_rig N=$N T=$T MIN_EPS=${EPS[$N]} ($THREADS threads)"
[ -x capcert_rig ] || bash build.sh
./capcert_rig "$N" "$T" "${EPS[$N]}" "$OUT/mesh" --threads "$THREADS" > "$OUT/mesh.out" 2> "$OUT/mesh.err"
cat "$OUT/mesh.out" | tee -a "$OUT/verify.log"
[ -f "$OUT/mesh.done" ] || { log "mesh did not complete"; exit 1; }

log "Step 3: refinement"
$PY -m proof.refine "$N" "$OUT/mesh" "$OUT/localcert_$nm.json" --workers "$THREADS" --min-width 1e-7 \
    > "$OUT/refine.log" 2>&1 || true
tail -2 "$OUT/refine.log" | tee -a "$OUT/verify.log"
$PY -c "import json;d=json.load(open('$OUT/mesh.refine.json'));assert d['verified'] and d['failed']==0" \
    && log "pipeline A, N=$N: all steps verified" || { log "pipeline A, N=$N: NOT VERIFIED"; exit 1; }
