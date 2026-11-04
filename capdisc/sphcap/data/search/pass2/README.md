# Search passes for N = 21..25 (numerical, level 0)

`jobs/search_parallel.py` (10 rounds x 12 runs per task, seed 0), first pass 2026-11-03; second pass 2026-11-03/04,
15 more rounds resuming the first pass's `.pkl` (so hits are cumulative). Workers: 2 for N = 21, 22; 4 for N = 23..25.
The JSON files here are the state after the second pass (top 15 classes each); the logs have the best value per round
of both passes (machine names replaced by machine-A / machine-B).

To add them to the atlas export as pass 2: copy each file over `sphcap/data/search/search_N<N>.json` and run
`jobs/atlas_pipeline.py` with `--pass-info` (rounds 15, workers as above, runs_per_task 12, resumes_previous true) and
`--search-log` pointing at the matching log (see `claude/run-2026-11-04-pipeline.md`).

| N | best D after pass 2 | hits | pass-1 best |
|---|---|---|---|
| 21 | 0.1198690429 | 1 | 0.1205125917 |
| 22 | 0.1134466644 | 1 | 0.1134885779 |
| 23 | 0.1125606267 | 3 | 0.1127779833 |
| 24 | 0.1097414757 | 1 | 0.1097448120 |
| 25 | 0.1069742689 | 2 | 0.1069742689 |

Not converged at N = 21, 22, 24 (best found once); a third pass is running.
