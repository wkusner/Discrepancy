# Search pass 3 for N = 21, 22, 24 and pass 1 for N = 26, 27 (numerical, level 0)

`jobs/search_parallel.py`, seed 0, 15 rounds x 12 runs per task, 2026-11-04/05. N = 21, 22: 2 workers, resuming the
pass-2 `.pkl` (hits cumulative). N = 24: 4 workers, resuming the pass-2 `.pkl`. N = 26, 27: 4 workers, first pass.
The logs contain every pass on that machine (machine names replaced by machine-A / machine-B); pass 3 is the block
after the third `start` line. Registered in `KNOWN_PASSES` of `jobs/atlas_pipeline.py`.

| N | best D | hits |
|---|---|---|
| 21 | 0.1198129842 | 3 (cumulative) |
| 22 | 0.1134466644 | 1 (unchanged through pass 3) |
| 24 | 0.1097190160 | 1 |
| 26 | 0.1041367600 | 1 (first pass) |
| 27 | 0.1015545641 | 1 (first pass) |
