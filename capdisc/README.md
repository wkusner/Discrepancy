# capdisc — minimal spherical cap discrepancy configurations, N = 3..8

Code, data and paper for the "cap capturing and balancing" program
(W. Kusner; computations assisted by Claude). This folder lives on the `capdisc`
branch of `wkusner/Discrepancy`; the original 2014 Mathematica implementation of the
enumeration algorithm (`MatrixSphericalDiscrepancy.nb`) is on `master`.

| path | what |
|---|---|
| `sphcap/` | Python package: discrepancy (exact enumeration), `balance` (branches, forces, KKT/balance test, SLP descent), `gram` (Gram formulas, N=3/4 exact solutions, N=5 elimination), `highprec` (mpmath polishing, LLL identification), `search` (candidate search), `bnb` (Python branch-and-bound) |
| `csrc/capcert.c` | C/OpenMP certifier: proves D >= T cell by cell over the reduced configuration space |
| `jobs/` | server jobs: parallel search, identification, deep identification, atlas |
| `msolve/` | symmetry-reduced polynomial systems (area equations) for exact elimination |
| `paper/` | the paper source and figures |
| `run_all.sh` | orchestrator for all phases, N = 3..8 |
| `run_proof.sh` | global proof for N = 5, 6: local certificate, capcert at T = D*_hi, rigorous refinement of survivors |
| `status.py`, `push_status.sh` | STATUS.md writer and GitHub status push |
| `DATA.md` | data guide: file formats, conventions, what is where (for readers and the website) |

## Running on w227

```bash
ssh -J wkusner@100.105.162.104 woden@w227
git clone -b capdisc https://github.com/wkusner/Discrepancy.git ~/Discrepancy && cd ~/Discrepancy/capdisc
bash setup_env.sh          # once: user-space Python env (+ msolve if conda-forge is reachable)
bash build.sh              # compiles capcert, prints two CERTIFIED lines
tmux new -s capdisc
bash run_all.sh            # Ctrl-b d to detach;  tmux attach -t capdisc  to come back
```

Defaults: certifier 32 threads, search 12 workers, msolve 8 threads (52 of 64 — the
machine is shared). Change with `CERT_THREADS=… SEARCH_WORKERS=… MSOLVE_THREADS=…`.
`NS="5 6"` restricts N. Everything is resumable: rerun `bash run_all.sh` after an
interruption. `cat STATUS.md` any time for a summary (rewritten every 10 minutes).

To update the code later: `git pull` in `~/Discrepancy`, then `bash build.sh`.

### Status pushed to GitHub (optional)

With `PUSH=1 bash run_all.sh`, `STATUS.md` and the small result files are pushed every
10 minutes to the branch **`w227-status`** of this repo, so progress can be checked at
https://github.com/wkusner/Discrepancy/blob/w227-status/STATUS.md and pulled into the paper.
One-time setup on w227 — a deploy key restricted to this repo:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/capdisc_deploy -C "w227 capdisc"
cat ~/.ssh/capdisc_deploy.pub     # GitHub -> wkusner/Discrepancy -> Settings -> Deploy keys -> Add, tick "Allow write access"
cat >> ~/.ssh/config <<'EOF'
Host github-capdisc
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/capdisc_deploy
  IdentitiesOnly yes
EOF
ssh -T github-capdisc     # expect "Hi wkusner/Discrepancy! You've successfully authenticated"
export STATUS_REMOTE=github-capdisc:wkusner/Discrepancy.git
```

(Port 443 is used in case outbound port 22 is blocked.) Put the `export` line in
`~/.bashrc` or prefix the run: `STATUS_REMOTE=github-capdisc:wkusner/Discrepancy.git PUSH=1 bash run_all.sh`.

## What the phases establish

* **ladder**: `CERTIFIED` for (N, eta) is a proof that D(X) >= D_best(N) - eta for
  *every* configuration (floating point with explicit 1e-12 slack on all angles and
  bounds; an interval-arithmetic recheck is planned). The cost per rung shows what full
  certification (eta -> 0 plus local certificates at the minimisers) will cost.
* **search**: candidates for the minimisers (structured starts + basin hopping).
* **identify**: 600-digit polishing, minimal polynomials of D and of the Gram entries
  (numerical identification by LLL); `deep_identify` tries degrees up to 160 at 4000 digits.
* **msolve**: exact elimination of the symmetry-reduced area equations — the exact,
  certified minimal polynomial of D (`N5_C2v` reproduces the known degree-12 polynomial).
* **atlas** (`ATLAS=1`): all local minima below D_best + delta, N <= 6.

## Tests

```bash
~/sphcap-env/bin/python -m pytest -q tests
```
