"""Build the msolve input for a configuration with a 2-fold rotation symmetry.

The rotation axis becomes the z-axis; points on the axis are (0,0,+-1); swapped pairs are
(a,b,c) and (-a,-b,c); the first pair representative is put in the xz-plane (gauge).
One area equation per orbit of worst caps (see the paper, eq. (area-eq)):
    3-point cap:  det[x_i,x_j,x_k]^2 = tau^2 |(x_j-x_i) x (x_k-x_i)|^2
    2-point cap:  (1 + <x_i,x_j>)/2 = tau^2,      tau = 1 - 2 n_C/N + 2D
(n_C = points in the worst cap = n_closed for 'up' branches, N - n_open for 'low').
A saturation variable z removes degenerate components (coincident points, a0 = 0).

usage:  python jobs/make_msolve_c2.py NAME OUT.ms      (NAME from critical_configurations.json)
"""
import sys, os, json, itertools
import numpy as np, sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
from sphcap import balance as B

name, out = sys.argv[1], sys.argv[2]
d = {c['name']: c for c in json.load(open(os.path.join(HERE, '..', 'sphcap/data/critical_configurations.json')))['classes']}
X = np.array(d[name]['points']); N = len(X); G = X @ X.T
perm = None
for p in itertools.permutations(range(N)):
    if p == tuple(range(N)) or not np.allclose(G[np.ix_(p, p)], G, atol=1e-7):
        continue
    M = np.linalg.lstsq(X, X[list(p)], rcond=None)[0].T
    ev = np.sort(np.linalg.eigvals(M).real)
    if np.linalg.det(M) > 0 and np.allclose(ev, [-1, -1, 1], atol=1e-6):
        perm, R = p, M; break
assert perm is not None, "no 2-fold rotation symmetry"
w, v = np.linalg.eig(R); ax = np.real(v[:, np.argmin(abs(w - 1))]); ax /= np.linalg.norm(ax)
fixed = [i for i in range(N) if perm[i] == i]
pairs = sorted({tuple(sorted((i, perm[i]))) for i in range(N) if perm[i] != i})
# frame: e3 = axis, e1 = horizontal part of the first pair representative
r0 = pairs[0][0]; h = X[r0] - (X[r0] @ ax) * ax; e1 = h / np.linalg.norm(h); e2 = np.cross(ax, e1)
Y = X @ np.stack([e1, e2, ax]).T
syms = {}; P = {}; num = {}; unit = []; satf = 1
for k, (i, j) in enumerate(pairs):
    a, c = sp.symbols(f'a{k} c{k}')
    if k == 0:
        b = 0; num[a], num[c] = Y[i, 0], Y[i, 2]; unit.append(a**2 + c**2 - 1); satf *= a
    else:
        b = sp.Symbol(f'b{k}'); num[a], num[b], num[c] = Y[i]; unit.append(a**2 + b**2 + c**2 - 1)
    P[i] = sp.Matrix([a, b, c]); P[j] = sp.Matrix([-a, -b, c])
for i in fixed:
    P[i] = sp.Matrix([0, 0, int(round(Y[i, 2]))])
D, z = sp.symbols('D z'); num[D] = d[name]['D_float']
rep = B.balance_report(X)
seen, eqs = set(), []
dot = lambda u, v: (u.T * v)[0]
for br in rep.active:
    I = tuple(sorted(br.I))
    if I in seen: continue
    seen |= {I, tuple(sorted(perm[i] for i in I))}
    nC = br.n_cl if br.kind == 'up' else N - br.n_op
    tau = 1 - sp.Rational(2 * nC, N) + 2 * D
    if len(I) == 2:
        e = (1 + dot(P[I[0]], P[I[1]])) / 2 - tau**2
    else:
        A, Bv, C = (P[i] for i in I); n = (Bv - A).cross(C - A)
        e = sp.Matrix.hstack(A, Bv, C).det()**2 - tau**2 * dot(n, n)
    eqs.append(sp.expand(e))
# no coincident points: one factor per pair occurring together in a worst cap, up to the symmetry
seenp = set()
for br in rep.active:
    for i, j in itertools.combinations(sorted(br.I), 2):
        key = min((i, j), tuple(sorted((perm[i], perm[j]))))
        if key in seenp: continue
        seenp.add(key); satf *= (1 - dot(P[i], P[j]))
print("saturation factors:", len(seenp))
nunk = len(num)
print(f"{name}: N={N}, pairs={pairs}, fixed={fixed}, {len(eqs)} area eqs + {len(unit)} unit eqs, {nunk} unknowns")
vals = [float(e.subs(num)) for e in unit + eqs]
print("residuals at the numerical configuration:", max(abs(v) for v in vals))
allv = [str(s) for s in sorted(num, key=lambda s: str(s)) if str(s) != 'D']
def clear(e):
    e = sp.expand(e); Pp = sp.Poly(e, *sorted(e.free_symbols, key=str))
    den = sp.ilcm(*[sp.fraction(cf)[1] for cf in Pp.coeffs()]); return sp.expand(e * den)
polys = [clear(e) for e in unit + eqs] + [sp.expand(z * satf - 1)]
open(out, 'w').write(','.join(['z'] + allv + ['D']) + "\n0\n" + ",\n".join(str(p).replace('**', '^') for p in polys) + "\n")
print("wrote", out)
