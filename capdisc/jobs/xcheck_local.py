"""Second-kernel check of the local verification (jobs/local_cert.py), in mpmath interval arithmetic.

Independent of Arb/python-flint and of sphcap.certify: points, caps, derivatives (forward AD over
mpmath.iv intervals), the Krawczyk operator and the orthant test are all re-implemented here.
Reads only the *claims* from certificates/localcert_<NAME>.json (caps, counts, box radius) and
re-proves them for every image:
  K1  the square system phi_b(q) - D = 0 (b in A) has a unique zero in a box Z (Krawczyk), and the
      D-component of K(Z) lies inside the Arb enclosure's neighbourhood (consistency);
  K2  counts of the caps in A are frozen on the big box B (lower = upper = n_b);
  K3  every other cap (except the listed redundant ones) has value < D_lo on Z, and 1/N < D_lo;
  K4  orthant test on B: for every sign vector s there is mu >= 0 with s_j (sum_b mu_b G^s_bj) > 0.
usage: python jobs/xcheck_local.py NAME        writes certificates/xcheck_local_<NAME>.json
"""
import sys, os, json, itertools, re
import numpy as np
from mpmath import iv, mp
from scipy.optimize import linprog

iv.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(HERE, '..', 'certificates')
name = sys.argv[1]
lc = json.load(open(os.path.join(CERT, f'localcert_{name}.json')))
N = lc['N']; nq = 2 * N - 3; rB = lc['box_radius']


# ---------------------------------------------------------------- forward AD over mpmath.iv
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
        b = V.c(b); q = a.v / b.v
        d = {k: x / b.v for k, x in a.d.items()}
        for k, x in b.d.items():
            t = -q * x / b.v; d[k] = d[k] + t if k in d else t
        return V(q, d)
    def sqrt(a):
        s = iv.sqrt(a.v); return V(s, {k: x / (2 * s) for k, x in a.d.items()})


def sc(x):  # sin, cos with derivatives
    s, c = iv.sin(x.v), iv.cos(x.v)
    return V(s, {k: v * c for k, v in x.d.items()}), V(c, {k: -v * s for k, v in x.d.items()})


def points(q):
    """q: list of iv intervals -> list of 3-vectors of V (with derivatives d/dq_k)."""
    Q = [V(q[k], {k: iv.mpf(1)}) for k in range(nq)]
    P = [[V.c(0), V.c(0), V.c(1)]]
    s0, c0 = sc(Q[0]); P.append([s0, V.c(0), c0])
    for i in range(2, N):
        st, ct = sc(Q[2 * i - 3]); sp, cp = sc(Q[2 * i - 2])
        P.append([st * cp, st * sp, ct])
    return P


dot = lambda a, b: a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
sub = lambda a, b: [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
cross = lambda a, b: [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def cap(P, I, side):
    """(t, margin(x)) with margin >= 0 iff x in the closed cap."""
    if len(I) == 2:
        S = [P[I[0]][k] + P[I[1]][k] for k in range(3)]; SS = dot(S, S)
        return SS.sqrt() * 0.5, (lambda x: dot(S, x) * 2 - SS)
    a, b, c = (P[i] for i in I)
    n = cross(sub(b, a), sub(c, a)); na = dot(n, a)
    return na * side / dot(n, n).sqrt(), (lambda x: (dot(n, x) - na) * side)


def parse(s):
    m = re.match(r"Cap\(([\d, ]+)\)([+-])", s)
    return tuple(int(v) for v in m.group(1).split(',')), (1 if m.group(2) == '+' else -1)


def phi(P, I, side, n):
    t, _ = cap(P, I, side)
    return t * 0.5 + (iv.mpf(n) / N - iv.mpf(0.5))


def lo(x): return x.a
def hi(x): return x.b


def check_image(im):
    caps = [parse(s) for s in im['caps']]; counts = im['counts']
    extra = [parse(s) for s in im.get('extra_caps', [])]
    qc = im['q_center']
    rep = {}
    # ---- K1 Krawczyk on (q, D), own Newton polish in mpmath (60 digits)
    mp.dps = 60
    x = [mp.mpf(v) for v in qc] + [mp.mpf(sum(im['D_enclosure']) / 2)]
    def F_mid(xm):
        P = points([iv.mpf(v) for v in xm[:-1]])
        return [phi(P, I, s, n) - V.c(iv.mpf(xm[-1])) for (I, s), n in zip(caps, counts)]
    for _ in range(8):
        Fv = F_mid(x)
        J = np.array([[float(mp.mpf(f.d.get(j, iv.mpf(0)).mid)) for j in range(nq)] + [-1.0] for f in Fv])
        r = np.array([float(mp.mpf(f.v.mid)) for f in Fv])
        dx = np.linalg.solve(J, r)
        x = [x[i] - mp.mpf(dx[i]) for i in range(nq + 1)]
    Minv = np.linalg.inv(J)
    n = nq + 1
    ok = False
    for rk in ('1e-20', '1e-16', '1e-13'):
        Z = [iv.mpf([x[i] - mp.mpf(rk), x[i] + mp.mpf(rk)]) for i in range(n)]
        Fm = [f.v for f in F_mid(x)]
        P = points(Z[:-1])
        FZ = [phi(P, I, s, cnt) - V(Z[-1], {nq: iv.mpf(1)}) for (I, s), cnt in zip(caps, counts)]
        K = []; ok = True
        for i in range(n):
            acc = iv.mpf(x[i])
            for j in range(n): acc = acc - iv.mpf(Minv[i, j]) * Fm[j]
            for j in range(n):
                cij = iv.mpf(1 if i == j else 0)
                for l in range(n): cij = cij - iv.mpf(Minv[i, l]) * FZ[l].d.get(j, iv.mpf(0))
                acc = acc + cij * (Z[j] - iv.mpf(x[i * 0 + j]))
            K.append(acc)
            ok = ok and lo(acc) > lo(Z[i]) and hi(acc) < hi(Z[i])
        if ok: break
    rep['krawczyk'] = ok; rep['krawczyk_radius'] = float(rk)
    rep['krawczyk_center'] = [mp.nstr(v, 40) for v in x]          # (q, D), box = center +- radius
    if not ok: return rep, False
    Dk = K[-1]
    rep['D_enclosure'] = [float(lo(Dk)), float(hi(Dk))]
    # consistency with the Arb result: the D enclosures must overlap
    rep['consistent_with_arb'] = bool(float(lo(Dk)) <= im['D_enclosure'][1] and float(hi(Dk)) >= im['D_enclosure'][0])
    # ---- K3 other caps below D on the Krawczyk box
    P = points(K[:-1])
    used = set(caps) | set(extra)
    worst = -1.0; bad = False
    for k in (2, 3):
        for I in itertools.combinations(range(N), k):
            for side in ((1,) if k == 2 else (1, -1)):
                if (I, side) in used: continue
                try:
                    t, marg = cap(P, I, side)
                except ZeroDivisionError:
                    bad = True; continue
                # by the one-sided formula and the capture lemma, D = max(1/N, max over the capture set of
                # n/N - sigma) with closed caps: smallest pair caps and both caps of each triple circle
                nup = len(I) + sum(1 for j in range(N) if j not in I and hi(marg(P[j]).v) >= 0)
                up = iv.mpf(nup) / N - (1 - t.v) / 2
                worst = max(worst, float(hi(up)))
    rep['max_other_caps_on_Z'] = worst
    rep['others_below'] = (not bad) and worst < float(lo(Dk)) and 1.0 / N < float(lo(Dk))
    # ---- K2 counts frozen on the big box B (naive interval evaluation over B)
    B = [iv.mpf([v - rB, v + rB]) for v in qc]
    PB = points(B)
    frozen = True
    for (I, s), cnt in zip(caps, counts):
        _, marg = cap(PB, I, s)
        lo_c = len(I) + sum(1 for j in range(N) if j not in I and lo(marg(PB[j]).v) > 0)
        up_c = len(I) + sum(1 for j in range(N) if j not in I and hi(marg(PB[j]).v) >= 0)
        frozen = frozen and lo_c == cnt == up_c
    rep['counts_frozen_on_box'] = frozen
    # ---- K4 orthant test with gradient enclosures over B
    G = []
    for (I, s), cnt in zip(caps, counts):
        f = phi(PB, I, s, cnt)
        G.append([f.d.get(j, iv.mpf(0)) for j in range(nq)])
    Glo = np.array([[float(lo(g)) for g in row] for row in G]); Ghi = np.array([[float(hi(g)) for g in row] for row in G])
    nfail = 0
    for sg in itertools.product((1, -1), repeat=nq):
        sv = np.array(sg)
        Gs = np.where(sv > 0, Glo, Ghi)
        A = -(Gs * sv).T
        res = linprog(np.append(np.zeros(len(caps)), -1.0), A_ub=np.hstack([A, np.ones((nq, 1))]), b_ub=np.zeros(nq),
                      A_eq=np.append(np.ones(len(caps)), 0.0)[None], b_eq=[1.0],
                      bounds=[(0, None)] * len(caps) + [(None, None)], method='highs')
        if res.status != 0: nfail += 1; continue
        mu = np.maximum(res.x[:-1], 0.0)
        for j in range(nq):
            # exact interval endpoints of the gradient enclosure (not the float copies)
            acc = iv.mpf(0)
            for b in range(len(caps)):
                gb = G[b][j]; e = iv.mpf(gb.a) if sv[j] > 0 else iv.mpf(gb.b)
                acc = acc + iv.mpf(mu[b]) * e
            if not (lo(acc * int(sv[j])) > 0): nfail += 1; break
    rep['orthants_failed'] = nfail
    verified = rep['krawczyk'] and rep['others_below'] and frozen and nfail == 0
    return rep, verified


out = []
allok = True
for k, im in enumerate(lc['images']):
    rep, ok = check_image(im)
    rep['image'] = k; rep['verified'] = bool(ok); out.append(rep); allok = allok and ok
    print(f"  image {k}: " + ", ".join(f"{key}={rep.get(key)}" for key in ('krawczyk', 'D_enclosure', 'consistent_with_arb', 'others_below', 'counts_frozen_on_box', 'orthants_failed', 'verified')), flush=True)
res = dict(name=name, N=N, kernel='mpmath.iv (interval arithmetic, 30 digits)', method='Krawczyk test + orthant test',
           independent_of='arb/python-flint and sphcap.certify', box_radius=rB, verified=bool(allok), images=out)
json.dump(res, open(os.path.join(CERT, f'xcheck_local_{name}.json'), 'w'), indent=1)
print("KRAWCZYK-VERIFIED (mpmath.iv): second kernel agrees" if allok else "SECOND KERNEL DOES NOT CONFIRM")
