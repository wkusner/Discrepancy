#!/usr/bin/env bash
# jobs/run_atlas_search.sh -- server job: grow the atlas to N = 13..20 (jobs/search_parallel.py; numerical).
# 8 workers at low priority (nice 10), so that with the 52-thread N = 7 mesh a few cores stay free for others.
# Rough cost (measured on a laptop core): one search run takes ~15 s at N = 13, ~25 s at N = 16, ~75 s at N = 20;
# 10 rounds x 8 workers x 12 runs = 960 runs per N, i.e. about 10 hours in total.
# Outputs results/search_N<N>.{pkl,json}; the export and stuck-point classes are added afterwards
# (jobs/atlas_export.py, jobs/atlas_stuck.py).
#   tmux new -d -s atlas 'bash jobs/run_atlas_search.sh > results/atlas_search.log 2>&1'
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/sphcap-env/bin/python}; [ -x "$PY" ] || PY=python3
WORKERS=${WORKERS:-8}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one BLAS thread per worker (numpy default: all cores)
for N in ${NS:-13 14 15 16 17 18 19 20}; do
  nice -n 10 $PY jobs/search_parallel.py $N --workers $WORKERS --rounds 10 --runs-per-task 12 | grep -v '^{'
done
echo "atlas search done"
