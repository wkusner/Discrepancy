"""Parallel candidate search on w227.

Rounds of independent search tasks (multiprocessing); after each round the classes
are merged and the best ones seed basin hopping in the next round.

usage:  python jobs/search_parallel.py N [--workers 16] [--rounds 10] [--runs-per-task 25]
writes: results/search_N{N}.pkl   (list of (D, X, hits, sources), best first)
        results/search_N{N}.json  (summary of the top classes)
"""
import argparse, json, os, pickle, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np
from sphcap import search as S

MIX = dict(riesz=0.3, rings=0.25, spiral=0.05, uniform=0.05, hop=0.35)


def task(args):
    N, runs, seed, init = args
    cl = S.search(N, runs, seed=seed, mix=MIX, keep_top=8, verbose=False, init=init)
    return [(k.D, k.X, k.hits, k.sources) for k in cl[:40]]


def merge(store, items):
    for D, X, hits, src in items:
        key = (round(D, 8), S.invariant(X, 4))
        if key in store:
            store[key][2] += hits
            for s, c in src.items():
                store[key][3][s] = store[key][3].get(s, 0) + c
        else:
            store[key] = [D, X, hits, dict(src)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--runs-per-task', type=int, default=25)
    a = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    out = f'results/search_N{a.N}'
    store = {}
    if os.path.exists(out + '.pkl'):
        merge(store, pickle.load(open(out + '.pkl', 'rb')))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for r in range(a.rounds):
            best = sorted(store.values(), key=lambda v: v[0])[:6]
            init = [v[1] for v in best]
            jobs = [(a.N, a.runs_per_task, 1000003 * r + 7919 * w + a.N, init) for w in range(a.workers)]
            for items in pool.imap_unordered(task, jobs):
                merge(store, items)
            cl = sorted(store.values(), key=lambda v: v[0])
            pickle.dump([tuple(v) for v in cl], open(out + '.pkl', 'wb'))
            print(f"[{time.time()-t0:7.0f}s] N={a.N} round {r+1}/{a.rounds}: {len(cl)} classes; "
                  f"best {cl[0][0]:.12f} (hits {cl[0][2]}); next {[round(v[0], 10) for v in cl[1:4]]}", flush=True)
    rows = []
    from sphcap import balance as B
    for D, X, hits, src in sorted(store.values(), key=lambda v: v[0])[:15]:
        rep = B.balance_report(X)
        rows.append(dict(D=D, hits=hits, active=len(rep.active), rank=rep.rank, dimq=rep.dim_quot,
                         min_lambda=float(rep.min_lambda), sharp=bool(rep.sharp),
                         aut=S.gram_automorphisms(X), sources=src))
    json.dump(rows, open(out + '.json', 'w'), indent=1)
    for r in rows:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
