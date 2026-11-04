#!/usr/bin/env bash
# verify.sh N  --  pipeline A of the N = 5, 6 proof in proof mode (PROOF.md Steps 1-6), with the trusted core proof/.
#
#   THREADS=16 bash verify.sh 5          outputs in results/verify_N5/ ; nothing is resumed or reused
#
# Proof mode: the run aborts if the proof code (proof/, csrc/, verify.sh, build.sh, check/) differs from the commit
# (untracked files included), rebuilds capcert_rig (build.sh bakes the commit into it) and checks that the binary
# is from this commit.  VERIFY_ALLOW_DIRTY=1 runs anyway, as a TEST run: the stamps then say tree_dirty and the
# chain checker rejects the chain (C8).
#
# Steps: local verification at every image (Step 4) -> symmetric verification per image (Step 5) -> congruence
# (Step 6) -> mesh capcert_rig at T = largest upper end of the D* enclosures (Step 2) -> refinement with the
# local boxes excluded (Step 3).  Exact data (Step 7) and pipeline B are separate (jobs/gram_verify.py, run_xcheck.sh).
# The chain checker (check/check_chain.py) then checks that the certificates fit together; its result goes to
# results/verify_N<N>/chain_N<N>.json.
set -euo pipefail
cd "$(dirname "$0")"
N=${1:?usage: verify.sh N}
THREADS=${THREADS:-4}
# mesh bounds: zeroth order everywhere, the first-order bound (csrc/fo.h) on cells with motion bound < 0.1
# (measured on N = 7 root cells: same time as zeroth order alone, 36x fewer survivors; MESH_OPTS= turns it off)
MESH_OPTS=${MESH_OPTS---first-order --fo-emax 0.1}
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
declare -A NAME=([5]=N5_best [6]=N6_best [7]=N7_best) RBOX=([5]=3e-4 [6]=1e-4 [7]=1e-4) EPS=([5]=0.04 [6]=0.02 [7]=0.0625)
# second mesh pass (capcert_rig --cells) on the survivors of the first, down to a smaller cell size, before the
# Python refinement: needed for N = 7, where the first pass leaves ~10^8-10^9 survivors far from the minimizer
declare -A EPS2=([7]=0.0078125)
[ -n "${MESH2_EPS:-}" ] && EPS2[$N]=$MESH2_EPS     # e.g. MESH2_EPS=0.01 bash verify.sh 5 to test the two-pass chain
# Levels (PROOF.md Section 3a):
# N = 5, 6: full (Steps 4-6, T = largest upper end of the D* enclosures, D* irrational by Step 5).
# N = 7: verified (Step 4 only; T = the next double above that upper end, so D >= T > D* outside the boxes
#        without the exact polynomial; congruence of the images is not checked; no image may have extra caps).
LEVEL=full; [ "$N" -ge 7 ] && LEVEL=verified
nm=${NAME[$N]}; OUT=results/verify_N$N
DIRTY=$(git status --porcelain --untracked-files=all -- proof csrc verify.sh build.sh check)
if [ -n "$DIRTY" ] && [ "${VERIFY_ALLOW_DIRTY:-}" != 1 ]; then
  echo "verify.sh: the proof code differs from commit $(git rev-parse --short HEAD); commit first (or VERIFY_ALLOW_DIRTY=1 for a test run):"
  echo "$DIRTY"; exit 1
fi
rm -rf "$OUT"; mkdir -p "$OUT"                     # proof mode: start clean
log() { echo "$(date '+%F %T')  $*" | tee -a "$OUT/verify.log"; }
log "verify N=$N ($nm), commit $(git rev-parse --short HEAD)${DIRTY:+ -- TEST RUN: proof code modified, not a proof run}"
bash build.sh > "$OUT/build.log" 2>&1 || { log "build.sh failed"; exit 1; }
PROV=$(./capcert_rig --version); log "capcert_rig: $PROV"
case "$PROV" in *"commit=$(git rev-parse HEAD) tree_dirty=0 "*) ;; *) [ -n "$DIRTY" ] || { log "capcert_rig is not built from this commit"; exit 1; } ;; esac

log "Step 4: local verification (level: $LEVEL)"
$PY -m proof.local "$nm" "${RBOX[$N]}" --out "$OUT" --level "$LEVEL" 2>&1 | tee -a "$OUT/verify.log"
nimg=$($PY -c "import json;d=json.load(open('$OUT/localcert_$nm.json'));assert d['verified'];print(len(d['images']))")

if [ "$LEVEL" = full ]; then
  log "Step 5: symmetric verification ($nimg images)"
  for k in $(seq 0 $((nimg - 1))); do $PY -m proof.symmetric "$nm" "$k" --dir "$OUT" 2>&1 | tail -2 | tee -a "$OUT/verify.log"; done
  log "Step 6: congruence"
  $PY -m proof.congruence "$nm" --dir "$OUT" 2>&1 | tee -a "$OUT/verify.log"
  T=$($PY -c "import json;d=json.load(open('$OUT/localcert_$nm.json'));print(repr(max(i['D_enclosure'][1] for i in d['images'])))")
else
  log "Steps 5-6 skipped (level: verified)"
  T=$($PY -c "import json,math;d=json.load(open('$OUT/localcert_$nm.json'));print(repr(math.nextafter(max(i['D_enclosure'][1] for i in d['images']),math.inf)))")
fi
log "Step 2: mesh capcert_rig N=$N T=$T MIN_EPS=${EPS[$N]} ($THREADS threads) $MESH_OPTS"
./capcert_rig "$N" "$T" "${EPS[$N]}" "$OUT/mesh" --threads "$THREADS" $MESH_OPTS > "$OUT/mesh.out" 2> "$OUT/mesh.err"
cat "$OUT/mesh.out" | tee -a "$OUT/verify.log"
[ -f "$OUT/mesh.done" ] || { log "mesh did not complete"; exit 1; }
MESH=mesh
if [ -n "${EPS2[$N]:-}" ]; then
  MESH=mesh2
  log "Step 2b: second mesh pass on the survivors, MIN_EPS=${EPS2[$N]}"
  ./capcert_rig "$N" "$T" "${EPS2[$N]}" "$OUT/mesh2" --threads "$THREADS" --cells "$OUT/mesh" $MESH_OPTS > "$OUT/mesh2.out" 2> "$OUT/mesh2.err"
  cat "$OUT/mesh2.out" | tee -a "$OUT/verify.log"
  [ -f "$OUT/mesh2.done" ] || { log "second mesh pass did not complete"; exit 1; }
fi

log "Step 3: refinement"
$PY -m proof.refine "$N" "$OUT/$MESH" "$OUT/localcert_$nm.json" --workers "$THREADS" --min-width 1e-7 --T "$T" \
    > "$OUT/refine.log" 2>&1 || true
tail -2 "$OUT/refine.log" | tee -a "$OUT/verify.log"
$PY -c "import json;d=json.load(open('$OUT/$MESH.refine.json'));assert d['verified'] and d['failed']==0" \
    && log "pipeline A, N=$N: all steps verified (level: $LEVEL)" || { log "pipeline A, N=$N: NOT VERIFIED"; exit 1; }
log "chain check"
if ! $PY check/check_chain.py "$N" "$OUT" --out "$OUT/chain_N$N.json" 2>&1 | tee -a "$OUT/verify.log"; then
  log "chain check: CHAIN INCOMPLETE"; exit 1
fi
