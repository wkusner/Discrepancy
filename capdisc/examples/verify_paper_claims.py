"""Re-derive the headline claims of the paper in about a minute.  Run: python examples/verify_paper_claims.py"""
import itertools, numpy as np, mpmath as mp
from sphcap import balance as B, gram as GM, highprec as H
from sphcap.data import load_critical

def ok(msg, cond):
    print(("PASS  " if cond else "FAIL  ") + msg)

# N=3
ok("N=3: great-circle triples with all arcs >= g* have D = 1/2",
   all(abs(B.discrepancy(GM.n3_points(a, b)) - .5) < 1e-12 for a, b in [(2, 2), (1.7, 2.5), (1.9, 2.2)]))
# N=4
ok("N=4: arc endpoints lie on the cubic and have D = 3/8",
   all(abs(GM.n4_cubic(*e)) < 1e-12 and abs(B.discrepancy(GM.disphenoid_points(*e)) - .375) < 1e-12
       for e in (GM.N4_ARC_END_SYMMETRIC, GM.N4_ARC_END_EDGE)))
pts = GM.n4_cubic_curve_points(1500)
ds = np.array([B.discrepancy(GM.disphenoid_points(*p)) for p in pts[::10]])
mx = pts[::10].max(1)
ok("N=4: on the cubic, D = 3/8 exactly where max(x,y,z) <= 9/16 (else D > 3/8)",
   np.all((ds < .375 + 1e-9) == (mx < 9/16 + 1e-3)) or np.all(ds[mx < .5622] < .375 + 1e-9) and np.all(ds[mx > .5628] > .375 + 1e-9))
# N=5
C = load_critical()
for name in ("N5_best", "N5_third"):
    c = C[name]; X = np.array(c["points"]); r = B.balance_report(X, act_tol=1e-7)
    ok(f"{name}: sharp minimiser (|A|={len(r.active)}, rank {r.rank}/{r.dim_quot}, min lambda {r.min_lambda:.3g}, D={r.D:.12f})", r.sharp)
mp.mp.dps = 60
D5 = mp.mpf(C["N5_best"]["D_60digits"])
ok("N=5: D5 is a root of the degree-12 polynomial (|P(D5)| < 1e-40)",
   abs(sum(mp.mpf(a) * D5 ** i for i, a in enumerate(GM.MINPOLY["N5_best"]))) < mp.mpf(10) ** -40)
Dv = mp.mpf(C["N5_valley"]["D_60digits"])
ok("N=5 valley value is a root of 2500x^4-2000x^3-3200x^2+940x+71",
   abs(sum(mp.mpf(a) * Dv ** i for i, a in enumerate(GM.MINPOLY["N5_valley"]))) < mp.mpf(10) ** -40)
import sympy as sp
D = sp.Symbol("D")
P = sp.Poly(sum(a * D ** i for i, a in enumerate(GM.MINPOLY["N5_best"])), D)
fl = GM.n5_eliminate()
ok("N=5: exact elimination of the C2v-reduced system contains P as a factor",
   any(sp.Poly(f, D).degree() == 12 and sp.div(sp.Poly(f, D), P)[1].is_zero for f, m in fl))
# N=6
c = C["N6_best"]; r = B.balance_report(np.array(c["points"]), act_tol=1e-7)
ok(f"N=6: best class sharp (|A|={len(r.active)}, rank {r.rank}/{r.dim_quot}), D={r.D:.12f}", r.sharp)
D6 = mp.mpf(C["N6_best"]["D_60digits"])
ok("N=6: D6 is a root of the degree-12 polynomial",
   abs(sum(mp.mpf(a) * D6 ** i for i, a in enumerate(GM.MINPOLY["N6_best"]))) < mp.mpf(10) ** -40)
