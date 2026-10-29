#!/usr/bin/env bash
# Global minimality proof pipeline for N=5 and N=6 (run on w227, inside tmux):
#   1. local certificate around each image of the minimizer (Krawczyk + orthant test)
#   2. capcert_rig (interval arithmetic) at T = upper end of the D* enclosure, coarse MIN_EPS, keeping every survivor
#   3. rigorous ball-arithmetic refinement of the survivors (jobs/refine.py), with the
#      certified boxes of step 1 excluded
# Output: results/proof_N{5,6}.* and logs/proof.log ; status.py picks up the .refine.json files.
#   THREADS=16 NS="5 6" bash run_proof.sh
set -u
cd "$(dirname "$0")"
THREADS=${THREADS:-16}
NS=${NS:-"5 6"}
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
mkdir -p results logs
[ -x capcert_rig ] || bash build.sh
declare -A NAME=([5]=N5_best [6]=N6_best)
declare -A RBOX=([5]=3e-4 [6]=1e-4)
declare -A EPS=([5]=0.02 [6]=0.02)
for N in $NS; do
  nm=${NAME[$N]}
  echo "=== N=$N ($nm) $(date)" | tee -a logs/proof.log
  $PY jobs/local_cert.py "$nm" "${RBOX[$N]}" 2>&1 | tee -a logs/proof.log
  lc=results/localcert_$nm.json
  T=$($PY -c "import json;d=json.load(open('$lc'));print(repr(max(i['D_enclosure'][1] for i in d['images'])) if d['certified'] else 'FAIL')")
  [ "$T" = FAIL ] && { echo "local certificate failed for $nm" | tee -a logs/proof.log; continue; }
  rm -f results/proof_N$N.surv.* results/proof_N$N.progress
  ./capcert_rig "$N" "$T" "${EPS[$N]}" "results/proof_N$N" --threads "$THREADS" \
      > "results/proof_N$N.out" 2> "logs/proof_capcert_N$N.log"
  cat "results/proof_N$N.out" | tee -a logs/proof.log
  $PY jobs/refine.py "$N" "results/proof_N$N" "$lc" --workers "$THREADS" --min-width 1e-7 \
      2>&1 | tee "logs/proof_refine_N$N.log" | tail -3 | tee -a logs/proof.log
done
echo "=== done $(date)" | tee -a logs/proof.log
