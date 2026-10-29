"""Second-kernel check of the symmetric verification (jobs/sym_cert.py), in mpmath interval arithmetic.

Re-proves, independently of Arb/python-flint and sphcap.certify:
  S1  the representative equations phi_R(Y(s)) = D (one per orbit; plus |s|^2 = 1 for N=6) have a
      unique zero (s*, D_s) in a small box (Krawczyk, forward AD over mpmath.iv);
  S2  the gauge coordinates q(Y(s*)) and D_s lie in the Krawczyk box of the *mpmath* local check
      (certificates/xcheck_local_<NAME>.json), so (q*, D*) = (q(Y(s*)), D_s).
The orbit structure (which caps are images of which under the exact symmetry group) is exact
combinatorics and is taken from certificates/symcert_<NAME>_img<k>.json; the elimination to P_N is
cross-checked separately (sympy resultants vs msolve Groebner bases).
usage: python jobs/xcheck_sym.py NAME IMAGE        writes certificates/xcheck_sym_<NAME>_img<k>.json
"""
import sys, os, json, itertools, re
import numpy as np
from mpmath import iv, mp
iv.dps = 40
HERE = os.path.dirname(os.path.abspath(__file__)); CERT = os.path.join(HERE, '..', 'certificates')
sys.path.insert(0, HERE)
name, IMG = sys.argv[1], int(sys.argv[2])
lc = json.load(open(os.path.join(CERT, f'localcert_{name}.json')))
xl = json.load(open(os.path.join(CERT, f'xcheck_local_{name}.json')))
sc_ = json.load(open(os.path.join(CERT, f'symcert_{name}_img{IMG}.json')))
N = lc['N']; nq = 2 * N - 3
im = lc['images'][IMG]; xim = xl['images'][IMG]
qc = np.array(im['q_center'])


def parse(s):
    m = re.match(r"Cap\(([\d, ]+)\)([+-])", s)
    return tuple(int(v) for v in m.group(1).split(',')), (1 if m.group(2) == '+' else -1)


cnt = {parse(s): n for s, n in zip(im['caps'], im['counts'])}
extra = {parse(s): n for s, n in zip(im.get('extra_caps', []), im.get('extra_counts', []))}
reps = [parse(s) for s in sc_['orbits']]
T = max(i['D_enclosure'][1] for i in lc['images'])


# ---- symmetric family (same parametrization as the paper, Section 8)
def family(s, sqrt, r3, half):
    if N == 5:
        c, e = s; scc, se = sqrt(1 - c * c), sqrt(1 - e * e)
        z = c * 0
        return [[z, scc, c], [se, z, e], [z, -scc, c], [z, z, z + 1], [-se, z, e]]
    a, b, c = s
    rot = lambda p: [-half * p[0] - half * r3 * p[1], half * r3 * p[0] - half * p[1], p[2]]
    x0 = [a, b, c]; x1 = rot(x0); x2 = rot(x1); h = lambda p: [p[0], -p[1], -p[2]]
    return [x0, x1, x2, h(x0), h(x1), h(x2)]


def gauge_float(Y):
    a = Y[0]; b = Y[1]; e1 = b - (a @ b) * a; e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    q = [np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)]
    for y in Y[2:]: q += [np.arctan2(np.linalg.norm(np.cross(a, y)), a @ y), np.arctan2(e2 @ y, e1 @ y)]
    return np.array(q)


# ---- labeling (selection only): float search for a relabeling with the same Gram matrix and orientation
def points_float(q):
    X = [[0, 0, 1.0], [np.sin(q[0]), 0, np.cos(q[0])]]
    for i in range(2, N):
        th, ph = q[2 * i - 3], q[2 * i - 2]
        X.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return np.array(X)


X_img = points_float(qc); Gimg = X_img @ X_img.T
fam_f = lambda s: np.array(family(s, np.sqrt, np.sqrt(3), 0.5), float)
if N == 5:
    x0 = np.array([0.05603, -0.51441])
else:
    D0 = 0.278679; a0 = (1 - 6 * D0) / 3; c0 = 2 * D0; x0 = np.array([a0, np.sqrt(1 - a0 ** 2 - c0 ** 2), c0])
found = None
for sg in itertools.product((1, -1), repeat=len(x0)):
    Y = fam_f(x0 * np.array(sg)); G = Y @ Y.T
    for perm in itertools.permutations(range(N)):
        if np.allclose(G[np.ix_(perm, perm)], Gimg, atol=2e-4):
            dq = (gauge_float(Y[list(perm)]) - qc + np.pi) % (2 * np.pi) - np.pi
            if np.abs(dq).max() < 1e-3: found = (x0 * np.array(sg), list(perm)); break
    if found: break
assert found, "no labeling"
s0, perm = found

# ---- exact symmetry: each generator g maps the symbolic family Y(s) to itself, g Y_j = Y_sigma(j)
import sympy as sp
ssym = sp.symbols('s0:%d' % len(s0))
Ysym = family(list(ssym), sp.sqrt, sp.sqrt(3), sp.Rational(1, 2))
if N == 5:
    gens = [sp.diag(1, -1, 1), sp.diag(-1, 1, 1)]
else:
    gens = [sp.Matrix([[-sp.Rational(1, 2), -sp.sqrt(3) / 2, 0], [sp.sqrt(3) / 2, -sp.Rational(1, 2), 0], [0, 0, 1]]), sp.diag(1, -1, -1)]
inv = [perm.index(j) for j in range(N)]
gens_img = []
for g in gens:
    sig = []
    for j in range(N):
        gy = g * sp.Matrix(Ysym[j])
        hit = [m for m in range(N) if all(sp.simplify(gy[r] - Ysym[m][r]) == 0 for r in range(3))]
        assert len(hit) == 1, "generator does not permute the family exactly"
        sig.append(hit[0])
    gens_img.append((tuple(inv[sig[perm[k]]] for k in range(N)), int(g.det())))
group = {(tuple(range(N)), 1)}
while True:
    new = {(tuple(gp[0][p[k]] for k in range(N)), d * gp[1]) for (p, d) in group for gp in gens_img}
    if new <= group: break
    group |= new


def image_cap(I, side, g):
    sig, det = g
    J = [sig[i] for i in I]
    if len(J) == 2: return (tuple(sorted(J)), 1)
    order = sorted(range(3), key=lambda k: J[k])
    sgn = round(float(np.linalg.det(np.eye(3)[order])))
    return (tuple(sorted(J)), side * det * sgn)


orbit_of = {}
for r in reps:
    for g in group:
        orbit_of.setdefault(image_cap(r[0], r[1], g), r)
all_caps = dict(cnt); all_caps.update(extra)
orbits_ok = all(c in orbit_of and all_caps[c] == cnt[orbit_of[c]] for c in all_caps)
print(f"  exact symmetry group of order {len(group)}; every worst cap (incl. {len(extra)} redundant) is the image of a representative with the same count: {orbits_ok}")


# ---- forward AD over mpmath.iv (independent of the Arb code)
class V:
    __slots__ = ('v', 'd')
    def __init__(s, v, d=None): s.v = v; s.d = d or {}
    @staticmethod
    def c(x): return x if isinstance(x, V) else V(iv.mpf(x))
    def __add__(a, b):
        b = V.c(b); d = dict(a.d)
        for k, x in b.d.items(): d[k] = d[k] + x if k in d else x
        return V(a.v + b.v, d)
    __radd__ = __add__
    def __neg__(a): return V(-a.v, {k: -x for k, x in a.d.items()})
    def __sub__(a, b): return a + (-V.c(b))
    def __rsub__(a, b): return V.c(b) + (-a)
    def __mul__(a, b):
        b = V.c(b); d = {k: x * b.v for k, x in a.d.items()}
        for k, x in b.d.items(): d[k] = d[k] + a.v * x if k in d else a.v * x
        return V(a.v * b.v, d)
    __rmul__ = __mul__
    def __truediv__(a, b):
        b = V.c(b); q = a.v / b.v; d = {k: x / b.v for k, x in a.d.items()}
        for k, x in b.d.items():
            t = -q * x / b.v; d[k] = d[k] + t if k in d else t
        return V(q, d)
    def sqrt(a):
        s = iv.sqrt(a.v); return V(s, {k: x / (2 * s) for k, x in a.d.items()})


dot = lambda a, b: a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
sub = lambda a, b: [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
cross = lambda a, b: [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def tlevel(P, I, side):
    if len(I) == 2:
        S = [P[I[0]][k] + P[I[1]][k] for k in range(3)]; return dot(S, S).sqrt() * 0.5
    a, b, c = (P[i] for i in I); n = cross(sub(b, a), sub(c, a))
    return dot(n, a) * side / dot(n, n).sqrt()


k = len(s0); n = k + 1


def system(sv, Dv):
    S = [V(sv[i], {i: iv.mpf(1)}) for i in range(k)]; Dd = V(Dv, {k: iv.mpf(1)})
    Y = family(S, lambda x: x.sqrt(), V.c(iv.sqrt(3)), V.c(iv.mpf(1) / 2))
    P = [Y[perm[j]] for j in range(N)]
    out = [tlevel(P, I, sd) * 0.5 + (iv.mpf(cnt[(I, sd)]) / N - iv.mpf(1) / 2) - Dd for (I, sd) in reps]
    if N == 6: out.append(S[0] * S[0] + S[1] * S[1] + S[2] * S[2] - 1)
    return out


mp.dps = 50
x = [mp.mpf(v) for v in s0] + [mp.mpf(sum(im['D_enclosure']) / 2)]
for _ in range(10):
    F = system([iv.mpf(v) for v in x[:-1]], iv.mpf(x[-1]))
    J = np.array([[float(mp.mpf(f.d.get(j, iv.mpf(0)).mid)) for j in range(n)] for f in F])
    r = np.array([float(mp.mpf(f.v.mid)) for f in F])
    # Newton in high precision: solve with the float Jacobian, residual in mpmath
    dx = np.linalg.solve(J, r)
    x = [x[i] - mp.mpf(dx[i]) for i in range(n)]
    for _ in range(3):
        F = system([iv.mpf(v) for v in x[:-1]], iv.mpf(x[-1]))
        r = [mp.mpf(f.v.mid) for f in F]
        Ji = np.linalg.inv(J)
        x = [x[i] - sum(mp.mpf(Ji[i, j]) * r[j] for j in range(n)) for i in range(n)]
Minv = np.linalg.inv(J)
ok = False
for rk in ('1e-30', '1e-25', '1e-20'):
    Z = [iv.mpf([x[i] - mp.mpf(rk), x[i] + mp.mpf(rk)]) for i in range(n)]
    Fm = [f.v for f in system([iv.mpf(v) for v in x[:-1]], iv.mpf(x[-1]))]
    FZ = system(Z[:-1], Z[-1])
    K = []; ok = True
    for i in range(n):
        acc = iv.mpf(x[i])
        for j in range(n): acc = acc - iv.mpf(Minv[i, j]) * Fm[j]
        for j in range(n):
            cij = iv.mpf(1 if i == j else 0)
            for l in range(n): cij = cij - iv.mpf(Minv[i, l]) * FZ[l].d.get(j, iv.mpf(0))
            acc = acc + cij * (Z[j] - iv.mpf(x[j]))
        K.append(acc); ok = ok and acc.a > Z[i].a and acc.b < Z[i].b
    if ok: break
assert ok, "Krawczyk (mpmath.iv) failed in symmetric coordinates"
Ds = K[-1]
# ---- gauge coordinates of Y(s*) (the whole box K) and inclusion in the mpmath local box
Y = family(K[:-1], iv.sqrt, iv.sqrt(3), iv.mpf(1) / 2); Y = [Y[perm[j]] for j in range(N)]
a, b = Y[0], Y[1]
ab = dot(a, b); e1 = [b[i] - ab * a[i] for i in range(3)]; ne = iv.sqrt(dot(e1, e1)); e1 = [v / ne for v in e1]
e2 = cross(a, e1)
norm = lambda v: iv.sqrt(dot(v, v))
qg = [iv.atan2(norm(cross(a, b)), ab)]
for y in Y[2:]:
    qg += [iv.atan2(norm(cross(a, y)), dot(a, y)), iv.atan2(dot(e2, y), dot(e1, y))]
Zb = [iv.mpf([mp.make_mpf(tuple(lo_)), mp.make_mpf(tuple(hi_))]) for lo_, hi_ in xim['krawczyk_box']]   # exact
inK = all((qg[j].a > Zb[j].a) is True and (qg[j].b < Zb[j].b) is True for j in range(nq))
inD = (Ds.a > Zb[-1].a) is True and (Ds.b < Zb[-1].b) is True
belowT = (Ds.b < iv.mpf(T)) is True
# N*D_s lies in the interval stated in Theorem 10 (endpoints exact rationals -> outward intervals)
THM = {5: ('1.616628254990254', '1.616628254990268'), 6: ('1.672072891057447', '1.672072891057457')}[N]
uN = Ds * N
in_thm = (uN.a > iv.mpf(THM[0])) is True and (uN.b < iv.mpf(THM[1])) is True
print(f"{name} image {IMG}: Krawczyk (mpmath.iv, radius {rk}): D_s in [{mp.nstr(Ds.a, 25)}, {mp.nstr(Ds.b, 25)}]; "
      f"q(Y(s*)) in the mpmath local box: {inK}; D_s in it: {inD}; D_s < T: {belowT}")
res = dict(name=name, image=IMG, kernel='mpmath.iv (interval arithmetic, 40 digits)', method='Krawczyk test in symmetric coordinates + gauge inclusion',
           independent_of='arb/python-flint and sphcap.certify', krawczyk_radius=float(rk),
           D_s=[mp.nstr(mp.mpf(Ds.a), 30), mp.nstr(mp.mpf(Ds.b), 30)], gauge_in_local_box=bool(inK), D_in_local_box=bool(inD),
           T=T, D_below_T=bool(belowT), group_order=len(group), orbits_exact=bool(orbits_ok),
           theorem_interval=THM, N_D_in_theorem_interval=bool(in_thm),
           verified=bool(ok and inK and inD and belowT and orbits_ok and in_thm))
json.dump(res, open(os.path.join(CERT, f'xcheck_sym_{name}_img{IMG}.json'), 'w'), indent=1)
print("KRAWCZYK-VERIFIED (mpmath.iv): second kernel agrees" if res['verified'] else "SECOND KERNEL DOES NOT CONFIRM")
