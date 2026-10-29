"""High-degree identification: polish to many digits, one LLL at large degree, factor.

usage: python deep_identify.py N dps deg1,deg2,...
"""
import sys, json, pickle, time
import numpy as np
import mpmath as mp
import flint
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sphcap import balance as B, highprec as H

N = int(sys.argv[1]); dps = int(sys.argv[2]); degs = [int(d) for d in sys.argv[3].split(',')]
D0, X, hits, src = pickle.load(open(f'results/search_N{N}.pkl', 'rb'))[0]
t0 = time.time()
rep = B.balance_report(X)
# bootstrap: polish at moderate precision, then raise
sol = H.polish_values(X, report=rep, dps=200)
mp.mp.dps = dps
nq = 2 * N - 3
spec = sol['spec']
def F(*u):
    q = list(u[:nq]); s = u[nq]
    Xm, _ = H.build_points(q, N)
    out = []
    for (I, kind, ncl, nop) in spec:
        t, w, beta = H.cap_geometry_mp(Xm, I)
        V = (1 - t) / 2
        out.append((mp.mpf(ncl) / N - V if kind == 'up' else V - mp.mpf(nop) / N) - s)
    return out
x0 = [mp.mpf(v) for v in sol['q']] + [mp.mpf(sol['D'])]
z = mp.findroot(F, x0, tol=mp.mpf(10) ** (-dps + 10), maxsteps=40, verify=False)
Dm = z[nq]
res = max(abs(e) for e in F(*z))
print(f"N={N} polished to dps={dps}, residual {mp.nstr(res, 3)}, {time.time()-t0:.0f}s", flush=True)
open(f'results/D{N}_{dps}digits.txt', 'w').write(mp.nstr(Dm, dps - 20))

digits = dps - 30
scale = mp.mpf(10) ** digits
for d in degs:
    t1 = time.time()
    pw = [mp.mpf(1)]
    for _ in range(d):
        pw.append(pw[-1] * Dm)
    rows = []
    for i in range(d + 1):
        r = [0] * (d + 2); r[i] = 1; r[d + 1] = int(mp.nint(pw[i] * scale)); rows.append(r)
    M = flint.fmpz_mat(rows).lll()
    found = None
    for k in range(min(3, M.nrows())):
        c = [int(M[k, i]) for i in range(d + 1)]
        if all(v == 0 for v in c):
            continue
        p = flint.fmpz_poly(c)
        for f, m in p.factor()[1]:
            fc = [int(f[i]) for i in range(f.degree() + 1)]
            val = mp.fsum(mp.mpf(ci) * Dm ** i for i, ci in enumerate(fc))
            h = max(abs(v) for v in fc)
            if abs(val) < mp.mpf(10) ** (-(dps // 2)) and (f.degree() + 1) * mp.log10(h + 1) < 0.7 * digits:
                found = fc
                break
        if found:
            break
    print(f"deg {d}: {'FOUND degree ' + str(len(found)-1) if found else 'none'}  [{time.time()-t1:.0f}s]", flush=True)
    if found:
        json.dump(dict(N=N, dps=dps, minpoly_ascending=found), open(f'results/minpoly_N{N}.json', 'w'))
        print(found, flush=True)
        break
