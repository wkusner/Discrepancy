"""Polish the best classes to high precision and identify D and the Gram entries.

usage: python identify_classes.py N [top] [dps]
"""
import sys, json, pickle, time
import numpy as np
import mpmath as mp
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sphcap import balance as B, highprec as H, search as S

N = int(sys.argv[1]); top = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
cl = pickle.load(open(f'results/search_N{N}.pkl', 'rb'))
out = []
for (D, X, hits, src) in cl[:top]:
    t0 = time.time()
    rep = B.balance_report(X)
    rec = dict(N=N, D_float=D, hits=hits, sources=src, active=len(rep.active), rank=rep.rank,
               dimq=rep.dim_quot, sharp=bool(rep.sharp), min_lambda=float(rep.min_lambda),
               aut=S.gram_automorphisms(X))
    if not rep.sharp:
        rec['note'] = 'not sharp: skipped high-precision polish'
        out.append(rec); print(json.dumps(rec), flush=True); continue
    try:
        sol = H.polish_values(X, report=rep, dps=dps)
    except Exception as e:
        rec['note'] = f'polish failed: {e}'
        out.append(rec); print(json.dumps(rec), flush=True); continue
    mp.mp.dps = dps
    Dm = sol['D']
    rec['residual'] = mp.nstr(sol['residual'], 3)
    rec['extra_residual'] = mp.nstr(sol['extra_residual'], 3)
    rec['D_digits'] = mp.nstr(Dm, 80)
    # minimal polynomial of D
    idD = H.identify_lll(Dm, max_degree=min(40, int(dps / 12)))
    rec['minpoly_D'] = idD[1] if idD else None
    # Gram entries
    Xm, _ = H.build_points(sol['q'], N)
    G = [[sum(Xm[i][k] * Xm[j][k] for k in range(3)) for j in range(N)] for i in range(N)]
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            if not any(abs(G[i][j] - v) < mp.mpf(10) ** (-dps // 2) for v in vals):
                vals.append(G[i][j])
    rec['gram'] = [[mp.nstr(G[i][j], 30) for j in range(N)] for i in range(N)]
    rec['gram_distinct'] = [mp.nstr(v, 40) for v in vals]
    polys = []
    for v in vals:
        r = H.identify_lll(v, max_degree=min(40, int(dps / 12)))
        polys.append(r[1] if r else None)
    rec['gram_minpolys'] = polys
    if idD:
        dD = idD[0]
        rec['gram_in_QD'] = []
        for v in vals:
            r = H.express_in_field(v, Dm, dD)
            rec['gram_in_QD'].append(dict(denom=r[0], coeffs=r[1]) if r else None)
    rec['time'] = round(time.time() - t0, 1)
    out.append(rec)
    print(json.dumps(rec), flush=True)
json.dump(out, open(f'results/identify_N{N}.json', 'w'), indent=1)
