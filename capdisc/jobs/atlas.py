"""Atlas post-processing: from the cells capcert could not certify at level T,
find every local minimiser of D below T.

capcert run with T = D_best + delta leaves survivor cells covering the whole sublevel
set {D < T + O(min_eps)}; every critical configuration with value < T lies in them.
We descend (SLP) from a sample of survivor centres (spread over the survivors) and
cluster the end points.  Local minima below T are found this way; saddles are not
(they are not attractors) -- they show up only through the survivor geometry.

usage: python jobs/atlas.py N PREFIX [--sample 3000] [--workers 16]
       (PREFIX as given to capcert, e.g. results/atlas_N5)
writes PREFIX.atlas.json, PREFIX.atlas.pkl
"""
import argparse, glob, json, os, pickle, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np
from sphcap import balance as B, bnb, search as S


def descend(args):
    N, q = args
    X = bnb.points(np.asarray(q)[None], N)[0]
    rng = np.random.default_rng(abs(hash(tuple(np.round(q, 9)))) % (2**32))
    Y, D = S.descend(X, rng, max_kicks=0)
    return D, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int); ap.add_argument('prefix')
    ap.add_argument('--sample', type=int, default=3000); ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    N, NQ = a.N, 2 * a.N - 3
    files = sorted(glob.glob(a.prefix + '.surv.*'))
    arr = np.concatenate([np.fromfile(f, dtype=np.float64) for f in files]).reshape(-1, 2 * NQ + 1)
    print(f"{len(arr):,} survivor cells; lower bounds in [{arr[:, -1].min():.10f}, {arr[:, -1].max():.10f}]", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(arr), size=min(a.sample, len(arr)), replace=False)
    with Pool(a.workers) as pool:
        res = pool.map(descend, [(N, arr[i, :NQ]) for i in idx], chunksize=4)
    store = {}
    for D, Y in res:
        key = (round(D, 8), S.invariant(Y, 4))
        if key in store:
            store[key][2] += 1
        else:
            store[key] = [D, Y, 1]
    cl = sorted(store.values(), key=lambda v: v[0])
    rows = []
    for D, Y, hits in cl:
        rep = B.balance_report(Y)
        rows.append(dict(D=D, hits=hits, active=len(rep.active), rank=rep.rank, dimq=rep.dim_quot,
                         min_lambda=float(rep.min_lambda), sharp=bool(rep.sharp), aut=S.gram_automorphisms(Y)))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(dict(N=N, survivors=int(len(arr)), sample=int(len(idx)), classes=rows), open(a.prefix + '.atlas.json', 'w'), indent=1)
    pickle.dump([(D, Y, h) for D, Y, h in cl], open(a.prefix + '.atlas.pkl', 'wb'))


if __name__ == '__main__':
    main()
