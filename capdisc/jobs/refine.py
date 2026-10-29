"""Rigorous refinement of the cells left by capcert, using the second-order (LP-combined)
ball-arithmetic bound of sphcap.certify, with exclusion boxes from jobs/local_cert.py.

A cell is closed when (a) it lies inside a certified box, or (b) the rigorous lower bound is >= T.
Otherwise it is bisected, down to --min-width; cells that still fail are reported.

usage: python jobs/refine.py N PREFIX LOCALCERT.json [--workers 16] [--min-width 1e-7] [--T value]
       (PREFIX as given to capcert; survivors are read from PREFIX.surv.*)
writes PREFIX.refine.json
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np
from sphcap import certify as Cf

G = {}


def init(N, T, boxes, min_width):
    G.update(N=N, T=T, boxes=boxes, minw=min_width, caps=Cf.all_caps(N))


def inside_box(c, w):
    for (m, r) in G['boxes']:
        if np.all(np.abs(c - m) + w <= r):
            return True
    return False


def split_dim(c, w, N):
    s = w.copy()
    for i in range(2, N):                       # phi coordinates weighted by sin(theta)
        s[2 * i - 2] = w[2 * i - 2] * (abs(np.sin(c[2 * i - 3])) + 1e-3)
    return int(np.argmax(s))


def process(cell):
    c0, w0 = cell
    N, T = G['N'], G['T']
    stack = [(np.array(c0), np.array(w0))]
    st = dict(cells=0, certified=0, boxed=0, failed=0, fails=[])
    while stack:
        c, w = stack.pop(); st['cells'] += 1
        if inside_box(c, w): st['boxed'] += 1; continue
        lb, info = Cf.cell_lower_bound(c, w, N, T, G['caps'])
        if lb >= T: st['certified'] += 1; continue
        if w.max() < G['minw']:
            st['failed'] += 1
            if len(st['fails']) < 20: st['fails'].append(dict(c=c.tolist(), w=w.tolist(), lb=float(lb.lower())))
            continue
        j = split_dim(c, w, N); h = w.copy(); h[j] /= 2
        a = c.copy(); a[j] -= h[j]; b = c.copy(); b[j] += h[j]
        stack += [(a, h), (b, h)]
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int); ap.add_argument('prefix'); ap.add_argument('localcert')
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--min-width', type=float, default=1e-7)
    ap.add_argument('--T', type=float, default=None)
    a = ap.parse_args()
    N, nq = a.N, 2 * a.N - 3
    lc = json.load(open(a.localcert))
    assert lc.get('certified'), "local certificate not certified"
    boxes = [(np.array(im['q_center']), lc['box_radius']) for im in lc['images']]
    T = a.T if a.T is not None else max(im['D_enclosure'][1] for im in lc['images'])
    files = sorted(glob.glob(a.prefix + '.surv.*'))
    arr = np.concatenate([np.fromfile(f, dtype=np.float64) for f in files]).reshape(-1, 2 * nq + 1) if files else np.zeros((0, 2 * nq + 1))
    cells = [(row[:nq], row[nq:2 * nq]) for row in arr]
    print(f"N={N}: {len(cells):,} cells from capcert, T={T!r}, {len(boxes)} exclusion box(es) of half-width {lc['box_radius']}", flush=True)
    t0 = time.time(); tot = dict(cells=0, certified=0, boxed=0, failed=0); fails = []
    with Pool(a.workers, initializer=init, initargs=(N, T, boxes, a.min_width)) as pool:
        for k, st in enumerate(pool.imap_unordered(process, cells, chunksize=8)):
            for key in tot: tot[key] += st[key]
            fails += st['fails'][:5]
            if (k + 1) % 1000 == 0 or k + 1 == len(cells):
                print(f"[{time.time()-t0:7.0f}s] {k+1:,}/{len(cells):,} input cells; {tot}", flush=True)
    res = dict(N=N, T=T, input_cells=len(cells), **tot, fails=fails[:50], time=time.time() - t0,
               proved=(tot['failed'] == 0))
    json.dump(res, open(a.prefix + '.refine.json', 'w'), indent=1)
    print("PROVED: no cell left" if res['proved'] else f"NOT PROVED: {tot['failed']} cells failed")


if __name__ == '__main__':
    main()
