#!/usr/bin/env bash
# jobs/run_identify.sh -- server job: numerical minimal-polynomial search for N D* (jobs/identify_poly.py) at N = 10, 12.
# A first look here (dps 600, degrees <= 48; dps 4000, degrees 72, 96 for N = 10) found no relation: u = N D* is
# generic up to degree 96 for heights below ~10^41.  This job goes to degree 240 at 14,000 digits.
# Rough cost: polish ~30 min per N; LLL ~1-6 h per degree near 200 (one core each).  Memory is small.
#   tmux new -s ident 'bash jobs/run_identify.sh 2>&1 | tee results/identify.log'
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
for N in 10 12; do
  $PY jobs/identify_poly.py $N 14000 120,144,168,192,216,240 --workers 6 &
done
wait
