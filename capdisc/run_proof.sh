#!/usr/bin/env bash
# Global minimality proof pipeline for N=5 and N=6 (run on w227, inside tmux):
#   1. local verification around each image of the minimizer (Krawczyk + orthant test, Arb), and the
#      symmetric verification (minimizer symmetric; N*D* is the root of P_N in the enclosure)
#   2. capcert_rig (interval-verified) at T = upper end of the D* enclosure, coarse MIN_EPS, keeping every survivor
#   3. Arb-verified refinement of the survivors (jobs/refine.py), with the
#      Krawczyk-verified boxes of step 1 excluded
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
  mkdir -p certificates && cp "$lc" certificates/
  nimg=$($PY -c "import json;print(len(json.load(open('$lc'))['images']))")
  for k in $(seq 0 $((nimg - 1))); do
    $PY jobs/sym_cert.py "$nm" "$k" 2>&1 | tail -4 | tee -a logs/proof.log
  done
  T=$($PY -c "import json;d=json.load(open('$lc'));print(repr(max(i['D_enclosure'][1] for i in d['images'])) if d.get('verified', d.get('certified')) else 'FAIL')")
  [ "$T" = FAIL ] && { echo "local verification failed for $nm" | tee -a logs/proof.log; continue; }
  [ "${RESUME:-0}" = 1 ] || rm -f results/proof_N$N.surv.* results/proof_N$N.progress
  rm -f results/proof_N$N.done
  ./capcert_rig "$N" "$T" "${EPS[$N]}" "results/proof_N$N" --threads "$THREADS" \
      > "results/proof_N$N.out" 2> "logs/proof_capcert_N$N.log"
  rc=$?
  cat "results/proof_N$N.out" | tee -a logs/proof.log
  if [ $rc -ne 0 ] || [ ! -f "results/proof_N$N.done" ]; then
    echo "capcert_rig did not complete for N=$N (exit $rc); rerun the same command to resume" | tee -a logs/proof.log; continue
  fi
  $PY jobs/refine.py "$N" "results/proof_N$N" "$lc" --workers "$THREADS" --min-width 1e-7 \
      2>&1 | tee "logs/proof_refine_N$N.log" | tail -3 | tee -a logs/proof.log
done
echo "=== done $(date)" | tee -a logs/proof.log
