"""Second-kernel check of the root isolation (sympy, exact rational arithmetic; the first kernel is FLINT).
P_N irreducible over Q, and exactly one real root of P_N in the interval of Theorem 10 (u = N D).
usage: python jobs/xcheck_roots.py            writes certificates/xcheck_roots.json"""
import json, os, sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); CERT = os.path.join(HERE, '..', 'certificates')
u = sp.Symbol('u')
P = {5: [64, -640, -352, 15344, -23004, -23384, -120504, 243968, 562963, -698538, -670529, 545392, 173504],
     6: [25, -496, 3524, -9120, -5298, 69920, -102348, -58272, 278817, -207344, -29312, 43520, 25600]}
I = {5: ('1.616628254990254', '1.616628254990268'), 6: ('1.672072891057447', '1.672072891057457')}
out = {}
for N in (5, 6):
    p = sp.Poly(sum(c * u ** (12 - i) for i, c in enumerate(P[N])), u)
    irr = p.is_irreducible
    lo, hi = sp.Rational(I[N][0]), sp.Rational(I[N][1])
    nroots = p.count_roots(lo, hi)
    out[N] = dict(irreducible=bool(irr), roots_in_interval=int(nroots), interval=I[N], kernel='sympy (exact, Sturm/real root isolation)')
    print(f"P_{N}: irreducible {irr}; real roots in [{lo}, {hi}]: {nroots}")
json.dump(out, open(os.path.join(CERT, 'xcheck_roots.json'), 'w'), indent=1)
