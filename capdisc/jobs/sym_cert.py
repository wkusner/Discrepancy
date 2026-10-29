"""Symmetric certificate: the minimizer has the expected symmetry, and D* is a root of P_N.

For N5_best (mirror group of order 4) and N6_best (dihedral rotation group of order 6):
 1. A symmetric family Y(s) of configurations, s = (c, e) for N=5, s = (a, b, c) on the unit sphere
    for N=6, is fixed by explicit orthogonal maps g (Y(s) is invariant as a set, exactly).
 2. The worst caps A of the local certificate (jobs/local_cert.py) split into orbits under the
    group; for one representative cap R per orbit we solve phi_R(Y(s)) = D (plus |s| = 1 for N=6),
    a square system, by a Krawczyk test in ball arithmetic (128-bit).  By symmetry every cap of A
    then has phi_C(Y(s*)) = D_s exactly.
 3. The gauge coordinates q(Y(s*)) and D_s lie in the Krawczyk box of the local certificate, whose
    zero is unique; so (q*, D*) = (q(Y(s*)), D_s): the minimizer is symmetric, and redundant worst
    caps (N=6) also have value D*.
 4. The squared representative equations, cleared of square roots, are polynomials in s and D;
    their eliminant R(D) (iterated resultants) vanishes at D*.  P_N(N D) divides R, and P_N is the
    only irreducible factor of R with a root in the enclosure of D*; so N D* is that root of P_N.
usage: python jobs/sym_cert.py N5_best|N6_best      writes certificates/symcert_<NAME>.json
"""
import sys, os, json, itertools, re
import numpy as np, sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import flint
from flint import arb, fmpz_poly
from sphcap import certify as Cf
flint.ctx.prec = 128

name = sys.argv[1]
CERT = os.path.join(HERE, '..', 'certificates')
lc = json.load(open(os.path.join(CERT, f'localcert_{name}.json')))
N = lc['N']; nq = 2 * N - 3
P_N = {5: [64, -640, -352, 15344, -23004, -23384, -120504, 243968, 562963, -698538, -670529, 545392, 173504],
       6: [25, -496, 3524, -9120, -5298, 69920, -102348, -58272, 278817, -207344, -29312, 43520, 25600]}[N]


# ------------------------------------------------------------------ the symmetric family
def family(s, lib):
    """points of Y(s); lib provides sqrt and the constant sqrt(3) (arb/AD, or sympy)."""
    sqrt = lib['sqrt']
    if N == 5:
        c, e = s
        sc, se = sqrt(1 - c * c), sqrt(1 - e * e)
        return [[0 * c, sc, c], [se, 0 * c, e], [0 * c, -sc, c], [0 * c, 0 * c, 0 * c + 1], [-se, 0 * c, e]]
    a, b, c = s
    r3 = lib['sqrt3']; half = lib['half']
    def rot(p):   # rotation by 120 degrees about z
        return [-half * p[0] - half * r3 * p[1], half * r3 * p[0] - half * p[1], p[2]]
    x0 = [a, b, c]; x1 = rot(x0); x2 = rot(x1)
    h = lambda p: [p[0], -p[1], -p[2]]        # half-turn about x
    return [x0, x1, x2, h(x0), h(x1), h(x2)]


if N == 5:
    s_guess = [0.05603, -0.51441]
    G_mats = [np.diag([1, -1, 1.0]), np.diag([-1, 1, 1.0])]           # mirrors y -> -y, x -> -x
else:
    s_guess = None
    c3, s3 = -0.5, np.sqrt(3) / 2
    G_mats = [np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1.0]]), np.diag([1, -1, -1.0])]


def fam_float(s):
    lib = dict(sqrt=np.sqrt, sqrt3=np.sqrt(3), half=0.5)
    return np.array(family(s, lib), float)


# ------------------------------------------------------------------ numeric solution and labels
def gauge_float(Y):
    a = Y[0]; b = Y[1]; e1 = b - (a @ b) * a; e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    q = [np.arccos(np.clip(a @ b, -1, 1))]
    for y in Y[2:]:
        q += [np.arccos(np.clip(a @ y, -1, 1)), np.arctan2(e2 @ y, e1 @ y)]
    return np.array(q)


img = lc['images'][0]
qc = np.array(img['q_center'])
from sphcap import bnb
X_img = bnb.points(qc[None], N)[0]
Gimg = X_img @ X_img.T


def solve_numeric():
    from scipy.optimize import least_squares
    if N == 5:
        x0 = np.array(s_guess)
    else:
        D0 = 0.278679
        a0 = (1 - 6 * D0) / 3; c0 = 2 * D0; b0 = np.sqrt(1 - a0 ** 2 - c0 ** 2)
        x0 = np.array([a0, b0, c0])
    best = None
    for sg in itertools.product((1, -1), repeat=len(x0)):
        x = x0 * np.array(sg)
        Y = fam_float(x); G = Y @ Y.T
        for perm in itertools.permutations(range(N)):
            if np.allclose(G[np.ix_(perm, perm)], Gimg, atol=2e-4):
                best = (x, perm); break
        if best: break
    assert best, "no labeling matches"
    x, perm = best
    def res(z):
        Y = fam_float(z); G = Y @ Y.T
        r = (G[np.ix_(perm, perm)] - Gimg)[np.triu_indices(N, 1)]
        return np.append(r, (z @ z - 1) if N == 6 else [])
    z = least_squares(res, x, xtol=1e-15, ftol=1e-15, gtol=1e-15).x
    return z, list(perm)


s_num, perm = solve_numeric()          # image point k = family point perm[k]
inv = [perm.index(j) for j in range(N)]  # family point j = image point inv[j]
Yn = fam_float(s_num)[perm]
assert np.allclose(Yn @ Yn.T, Gimg, atol=1e-9), "symmetric family does not reach the minimizer"
print(f"{name}: symmetric parameters {s_num}, labels {perm}")

# group as permutations of image labels (exact by construction: g maps family point j to point sigma(j))
def perm_of(g):
    Yf = fam_float(s_num); Z = Yf @ g.T
    sig = [int(np.argmin(np.linalg.norm(Yf - z, axis=1))) for z in Z]
    assert sorted(sig) == list(range(N)) and np.allclose(Yf[sig], Z, atol=1e-9)
    return [inv[sig[perm[k]]] for k in range(N)]     # image label k -> image label
gens = [(perm_of(g), int(round(np.linalg.det(g)))) for g in G_mats]
group = {(tuple(range(N)), 1)}
while True:
    new = {(tuple(p[q] for q in range(N)) if False else tuple(g[0][p[k]] for k in range(N)), d * g[1]) for (p, d) in group for g in gens}
    if new <= group: break
    group |= new
print(f"  group of order {len(group)}")


def parse_cap(sx):
    m = re.match(r"Cap\(([\d, ]+)\)([+-])", sx)
    I = tuple(int(v) for v in m.group(1).split(',')); return Cf.CapSpec(I, 1 if m.group(2) == '+' else -1)


def image_cap(cap, g):
    sig, det = g
    J = [sig[i] for i in cap.I]
    if len(J) == 2: return (tuple(sorted(J)), 0)
    order = sorted(range(3), key=lambda k: J[k])
    sgn = np.linalg.det(np.eye(3)[order])           # sign of the sorting permutation
    return (tuple(sorted(J)), int(round(cap.side * det * sgn)))


key = lambda c: (c.I, c.side if len(c.I) == 3 else 0)
caps = [parse_cap(x) for x in img['caps']]; counts = img['counts']
cnt = {key(c): n for c, n in zip(caps, counts)}
orbits, seen = [], set()
for c in caps:
    if key(c) in seen: continue
    orb = {image_cap(c, g) for g in group}
    orbits.append((c, orb)); seen |= orb
cover = all(key(c) in set().union(*[o for _, o in orbits]) for c in caps)
same_counts = all(cnt.get(k, cnt[key(r)]) == cnt[key(r)] for r, o in orbits for k in o if k in cnt)
print(f"  {len(caps)} caps of the square system in {len(orbits)} orbits; representatives {[repr(r) for r, _ in orbits]}; counts consistent: {same_counts}")
assert len(orbits) == len(s_num) + (1 if N == 5 else 0), "orbit count does not match the number of unknowns"


# ------------------------------------------------------------------ Krawczyk in symmetric coordinates
def system(sv, Dv):
    """AD values: [phi_R(Y(s)) - D for representatives R] (+ |s|^2 - 1 for N=6); variables s..., D."""
    k = len(sv)
    S = [Cf.AD(sv[i], {i: arb(1)}) for i in range(k)]; Dad = Cf.AD(Dv, {k: arb(1)})
    lib = dict(sqrt=lambda x: x.sqrt(), sqrt3=Cf.AD(arb(3).sqrt()), half=Cf.AD(arb(1) / 2))
    Y = family(S, lib)
    P = [Y[perm[kk]] for kk in range(N)]            # image labels
    out = []
    for r, _ in orbits:
        t, _ = Cf.cap_terms(P, r)
        out.append(t * 0.5 + (arb(cnt[key(r)]) / N - arb(1) / 2) - Dad)
    if N == 6:
        out.append(S[0] * S[0] + S[1] * S[1] + S[2] * S[2] - 1)
    return out


k = len(s_num); n = k + 1
m = np.append(s_num, np.mean(lc['images'][0]['D_enclosure']))
for _ in range(8):                                   # float Newton
    F = system([arb(float(x)) for x in m[:-1]], arb(float(m[-1])))
    J = np.array([[float(f.d.get(j, arb(0)).mid()) for j in range(n)] for f in F])
    m = m - np.linalg.solve(J, np.array([float(f.v.mid()) for f in F]))
Minv = np.linalg.inv(J)
mb = [arb(float(x)) for x in m]
for _ in range(6):                                   # Newton at 128 bits (midpoints only)
    F = system(mb[:-1], mb[-1])
    mb = [arb((mb[i] - sum((arb(float(Minv[i, j])) * F[j].v for j in range(n)), arb(0))).mid()) for i in range(n)]
ok = False
for rk in (1e-30, 1e-25, 1e-20):
    Z = [arb(x.mid(), rk) for x in mb]
    Fm = [f.v for f in system(mb[:-1], mb[-1])]
    FZ = system(Z[:-1], Z[-1])
    JZ = [[f.d.get(j, arb(0)) for j in range(n)] for f in FZ]
    K = []
    ok = True
    for i in range(n):
        acc = mb[i]
        for j in range(n): acc -= arb(float(Minv[i, j])) * Fm[j]
        for j in range(n):
            cij = arb(1 if i == j else 0)
            for l in range(n): cij -= arb(float(Minv[i, l])) * JZ[l][j]
            acc += cij * (Z[j] - mb[j])
        K.append(acc)
        ok = ok and acc.lower() > Z[i].lower() and acc.upper() < Z[i].upper()
    if ok: break
assert ok, "Krawczyk test failed in symmetric coordinates"
s_box, D_s = K[:-1], K[-1]
print(f"  Krawczyk (symmetric, radius {rk}): D_s = {D_s.str(30)}")

# ------------------------------------------------------------------ gauge coordinates in the local box
lib = dict(sqrt=lambda x: x.sqrt(), sqrt3=arb(3).sqrt(), half=arb(1) / 2)
Yb = family(s_box, lib); Yb = [Yb[perm[kk]] for kk in range(N)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
a, b = Yb[0], Yb[1]
ab = dot(a, b); e1 = [b[i] - ab * a[i] for i in range(3)]; ne = dot(e1, e1).sqrt(); e1 = [x / ne for x in e1]
e2 = [a[1] * e1[2] - a[2] * e1[1], a[2] * e1[0] - a[0] * e1[2], a[0] * e1[1] - a[1] * e1[0]]
qg = [ab.acos()]
for y in Yb[2:]:
    qg += [dot(a, y).acos(), arb.atan2(dot(e2, y), dot(e1, y))]
flip = 1 if float(qg[2].mid()) * qc[2] >= 0 else -1      # reflection y -> -y if needed
for i in range(2, N): qg[2 * i - 2] = qg[2 * i - 2] * flip
rk_loc = img['krawczyk_radius']
inK = all((qg[j] - arb(qc[j])).abs_upper() < rk_loc for j in range(nq))
Dlo, Dhi = img['D_enclosure']
inD = D_s.lower() > Dlo and D_s.upper() < Dhi
print(f"  q(Y(s*)) in the local Krawczyk box: {inK};  D_s inside the local enclosure: {inD}")

# ------------------------------------------------------------------ eliminant and P_N
syms = sp.symbols('s0:%d' % k); Dsym = sp.Symbol('D')
aux = sp.symbols('w0:2')                      # square roots in the N=5 family
if N == 5:
    libs = dict(sqrt=None)
    c_, e_ = syms; w0, w1 = aux
    Ysym = [[0, w0, c_], [w1, 0, e_], [0, -w0, c_], [0, 0, 1], [-w1, 0, e_]]
    rel = [w0 ** 2 - (1 - c_ ** 2), w1 ** 2 - (1 - e_ ** 2)]
else:
    r3 = sp.Symbol('r3')
    Ysym = family(list(syms), dict(sqrt=sp.sqrt, sqrt3=r3, half=sp.Rational(1, 2)))
    rel = []
Ysym = [list(map(sp.nsimplify, Ysym[perm[kk]])) if False else Ysym[perm[kk]] for kk in range(N)]
polys = []
for r, _ in orbits:
    I = r.I
    tau = 1 - sp.Rational(2 * cnt[key(r)], N) + 2 * Dsym
    if len(I) == 2:
        S = [Ysym[I[0]][i] + Ysym[I[1]][i] for i in range(3)]
        e = sp.expand(sum(x * x for x in S) / 4 - tau ** 2)        # t^2 = |S|^2/4
    else:
        A_, B_, C_ = (sp.Matrix(Ysym[i]) for i in I)
        nvec = (B_ - A_).cross(C_ - A_)
        e = sp.expand(nvec.dot(A_) ** 2 - tau ** 2 * nvec.dot(nvec))
    for rr in rel:
        e = sp.expand(sp.rem(sp.Poly(e, *aux), sp.Poly(rr, *aux)).as_expr()) if False else e
    if N == 5:
        e = sp.expand(e.subs({aux[0] ** 2: 1 - syms[0] ** 2, aux[1] ** 2: 1 - syms[1] ** 2}))
        e = sp.expand(e.subs({aux[0] ** 2: 1 - syms[0] ** 2, aux[1] ** 2: 1 - syms[1] ** 2}))
    polys.append(e)
if N == 6:
    polys.append(sum(x ** 2 for x in syms) - 1)
free_aux = [p for p in polys if p.has(*aux)]
assert not free_aux, "square roots survive in the squared equations"
# sanity: the squared equations vanish at the numerical solution
vals = [float(p.subs(dict(zip(syms, s_num))).subs({Dsym: float(D_s.mid()), sp.Symbol("r3"): 3 ** 0.5})) for p in polys]
print(f"  squared equations at the numerical solution: max |.| = {max(abs(v) for v in vals):.1e}")


def arb_eval(expr):
    f = sp.lambdify(list(syms) + [Dsym, sp.Symbol('r3')], expr, modules=[{'sqrt': lambda x: arb(x).sqrt()}])
    return f(*s_box, D_s, arb(3).sqrt())


def vanishing_factor(p):
    # the product of the factors vanishes at the true solution; a factor whose enclosure excludes 0
    # does not vanish there, so if exactly one factor's enclosure contains 0, that factor vanishes
    fs = [f for f, _ in sp.factor_list(p)[1] if sp.Poly(f, *syms, Dsym).total_degree() > 0]
    cand = [f for f in fs if arb(arb_eval(f)).contains(0)]
    assert len(cand) == 1, ("ambiguous factor", len(cand))
    return sp.expand(cand[0])


if N == 6:   # remove sqrt(3): replace p by its norm Res_r3(p, r3^2 - 3) (vanishes when p does)
    r3 = sp.Symbol('r3')
    polys = [sp.expand(sp.resultant(p, r3 ** 2 - 3, r3)) if p.has(r3) else p for p in polys]
polys = [vanishing_factor(p) for p in polys]
print("  degrees of the vanishing factors:", [sp.Poly(p, *syms, Dsym).total_degree() for p in polys])


def eliminate(ps, vars_):
    ps = [sp.Poly(p, *vars_, Dsym) for p in ps]
    for v in vars_:
        if len(ps) == 1: break
        base = ps[0]; new = []
        for p in ps[1:]:
            r = sp.resultant(base.as_expr(), p.as_expr(), v)
            if r == 0: continue
            new.append(sp.Poly(vanishing_factor(sp.expand(r)), *[w for w in vars_ if w != v], Dsym))
        ps = new
    return ps


R = None
# eliminate the parameters in an order that keeps degrees small
order = list(syms)
cand = eliminate(polys, order)
Rs = [sp.Poly(p.as_expr(), Dsym) for p in cand if p.free_symbols <= {Dsym}]
assert Rs, "elimination did not produce a univariate polynomial"
R = Rs[0]
for extra in Rs[1:]: R = sp.Poly(sp.gcd(R.as_expr(), extra.as_expr()), Dsym)
u = sp.Symbol('u'); P = sp.Poly(sum(cf * u ** (12 - i) for i, cf in enumerate(P_N)), u)
PD = sp.Poly(P.as_expr().subs(u, N * Dsym), Dsym)
divides = sp.rem(R, PD).is_zero
facs = sp.factor_list(R.as_expr())[1]
hits = []
for f, mult in facs:
    fp = sp.Poly(f, Dsym)
    if fp.degree() == 0: continue
    cs = [int(x) for x in (fp * sp.ilcm(*[sp.fraction(c)[1] for c in fp.all_coeffs()])).all_coeffs()][::-1]
    fl = fmpz_poly(cs)
    for r_, mlt in fl.complex_roots():
        if r_.imag.contains(0) and r_.real.upper() >= Dlo and r_.real.lower() <= Dhi:
            hits.append((str(f)[:60], fp.degree()))
print(f"  eliminant degree {R.degree()}; P_N(N D) divides it: {divides}; factors with a root in the enclosure: {hits}")
only_P = divides and len(hits) == 1 and sp.Poly(sp.factor_list(PD.as_expr())[1][0][0], Dsym).degree() == hits[0][1]

union = sorted(set().union(*[o for _, o in orbits]))
print(f"  orbits of the representatives contain {len(union)} caps (worst caps at the minimizer, including redundant ones): {union}")
res = dict(name=name, N=N, group_order=len(group), orbits=[repr(r) for r, _ in orbits], orbit_caps=[list(map(list, [c[0]])) + [c[1]] for c in union],
           counts_consistent=bool(same_counts), krawczyk_symmetric=bool(ok), krawczyk_radius=rk,
           D_s=[float(D_s.lower()), float(D_s.upper())], gauge_in_local_box=bool(inK), D_in_local_enclosure=bool(inD),
           eliminant_degree=R.degree(), P_divides_eliminant=bool(divides), factors_meeting_enclosure=hits,
           certified=bool(ok and inK and inD and same_counts and cover and only_P))
json.dump(res, open(os.path.join(CERT, f'symcert_{name}.json'), 'w'), indent=1)
print("CERTIFIED: minimizer symmetric, N*D* is the root of P_N in the enclosure" if res['certified'] else "NOT CERTIFIED")
