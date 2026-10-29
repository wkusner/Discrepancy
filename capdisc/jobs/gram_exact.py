"""Exact descriptions of the Gram matrices of the critical configurations.

For each configuration: polish to high precision (sphcap.highprec.polish_values), compute the
Gram matrix, group equal entries, and for each distinct entry find (numerically, by LLL)
  * its minimal polynomial over Q, and
  * an expression a*g = sum_i b_i D^i in the field Q(D), when the minimal polynomial of D is known.
These are numerical identifications, checked to (dps - 10) digits.

usage: python jobs/gram_exact.py [NAME ...] [--dps 800]      writes sphcap/data/gram_exact.json
"""
import sys, os, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np, mpmath as mp
from sphcap import balance as B, highprec as H

ap = argparse.ArgumentParser(); ap.add_argument('names', nargs='*'); ap.add_argument('--dps', type=int, default=800)
ap.add_argument('--max-degree', type=int, default=24)
a = ap.parse_args()
DATA = os.path.join(HERE, '..', 'sphcap', 'data')
cfg = {c['name']: c for c in json.load(open(os.path.join(DATA, 'critical_configurations.json')))['classes']}
names = a.names or ['N5_best', 'N5_valley', 'N5_third', 'N6_best', 'N6_second']
outf = os.path.join(DATA, 'gram_exact.json')
out = json.load(open(outf)) if os.path.exists(outf) else {}

for name in names:
    c = cfg[name]; X = np.array(c['points']); N = len(X)
    rep = B.balance_report(X)
    sol = H.polish_values(X, report=rep, dps=a.dps)
    mp.mp.dps = a.dps
    Xm, _ = H.build_points(sol['q'], N)
    G = [[mp.fsum(Xm[i][k] * Xm[j][k] for k in range(3)) for j in range(N)] for i in range(N)]
    D = sol['D']
    tol = mp.mpf(10) ** (-a.dps // 2)
    vals, pattern = [], [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            k = next((k for k, v in enumerate(vals) if abs(G[i][j] - v) < tol), None)
            if k is None:
                vals.append(G[i][j]); k = len(vals) - 1
            pattern[i][j] = pattern[j][i] = k
    idD = H.identify_lll(D, max_degree=a.max_degree)
    rec = dict(name=name, N=N, dps=a.dps, residual=mp.nstr(sol['residual'], 3),
               D=mp.nstr(D, 80), D_minpoly_ascending=idD[1] if idD else None,
               gram_pattern=pattern,
               note="gram_pattern[i][j] = index into 'entries' (diagonal = -1, meaning 1); "
                    "identifications are numerical (LLL), checked to dps-10 digits; "
                    "in_QD: denom * g = sum_i coeffs[i] * D^i",
               entries=[])
    for v in vals:
        e = dict(value=mp.nstr(v, 60))
        r = H.identify_lll(v, max_degree=a.max_degree)
        e['minpoly_ascending'] = r[1] if r else None
        if idD:
            q = H.express_in_field(v, D, idD[0])
            e['in_QD'] = dict(denom=q[0], coeffs=q[1]) if q else None
        rec['entries'].append(e)
    out[name] = rec
    print(f"{name}: N={N}, D degree {idD[0] if idD else None}, {len(vals)} distinct Gram entries; "
          f"minpolys found {sum(e['minpoly_ascending'] is not None for e in rec['entries'])}, "
          f"in Q(D) {sum(bool(e.get('in_QD')) for e in rec['entries'])}", flush=True)
    json.dump(out, open(outf, 'w'), indent=1)
