"""Exact Gram matrices of the minimizers (N = 5, 6): from "numerically identified" to exact.

Input: the entries of sphcap/data/gram_exact.json, each written as  g = (sum_i c_i D^i) / denom  in the
field K = Q(D), D the root of the minimal polynomial m(x) stored there.  This script proves that the
Gram matrix G_e built from these entries is the Gram matrix of the minimizer q* of Theorem 10:

  E1  m(x) is proportional to P_N(N x) (P_N as in xcheck_roots.py), so the root D_e of m in I_N / N is
      D* (N D* is the only root of P_N in the theorem interval I_N; certificates/xcheck_roots.json).
  E2  G_e has rank 3: det(A) C - B^T adj(A) B = 0 in K, where A = G_e[:3,:3] (exact).
  E3  every worst cap of the local verification satisfies its equation phi_b = D exactly, in the
      squared form  det G_I = tau^2 * |(b-a) x (c-a)|^2  (three points) or  tau^2 = (2 + 2 g_ab)/4
      (two points), tau = 1 - 2 n_b / N + 2 D  (exact in K).
  I1  A is positive definite (leading minors > 0), so G_e is positive semidefinite of rank 3 and is
      realized by points on S^2, unique up to O(3) (interval).
  I2  the signs: s * det[a, b, c] and tau have the same strict sign for each three-point cap, tau > 0
      for each two-point cap; with E3 this gives phi_b = D exactly (interval).
  I3  the realization in the gauge of the local verification lies in its box B (interval).

E3 + I2 say that q_e is a zero of the square system of the local verification with D = D*, and
I3 puts it in B.  The orthant test there shows max_b phi_b(Y) > D* for every Y in B other than q*,
so q_e = q*: G_e is the Gram matrix of the minimizer.

Kernels: the exact steps run twice, in sympy (Poly arithmetic mod m) and in FLINT (fmpq_poly);
the interval steps run twice, in Arb (python-flint) and in mpmath.iv.  D_e is enclosed by an exact
rational isolating interval (sympy) of width < 10^-120.

usage: python jobs/gram_verify.py [NAME ...]     writes certificates/gramcert_<NAME>.json
"""
import sys, os, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np, sympy as sp, mpmath as mp
from flint import fmpq_poly, fmpq, arb, ctx
from sphcap import certify as Cf

ROOT = os.path.join(HERE, '..'); CERT = os.path.join(ROOT, 'certificates')
P = {5: [64, -640, -352, 15344, -23004, -23384, -120504, 243968, 562963, -698538, -670529, 545392, 173504],
     6: [25, -496, 3524, -9120, -5298, 69920, -102348, -58272, 278817, -207344, -29312, 43520, 25600]}
I_TH = {5: ('1.616628254990254', '1.616628254990268'), 6: ('1.672072891057447', '1.672072891057457')}
x = sp.Symbol('x')


# ------------------------------------------------------------------ exact field arithmetic, two kernels
class SympyK:
    name = 'sympy'
    def __init__(s, m): s.m = sp.Poly(list(reversed(m)), x, domain=sp.QQ)
    def elem(s, coeffs, denom=1): return sp.Poly(list(reversed([sp.Rational(c, denom) for c in coeffs])) or [0], x, domain=sp.QQ).rem(s.m)
    def const(s, r): return sp.Poly(r, x, domain=sp.QQ)
    def gen(s): return sp.Poly(x, x, domain=sp.QQ)
    def mul(s, a, b): return (a * b).rem(s.m)
    def is_zero(s, a): return a.rem(s.m).is_zero


class FlintK:
    name = 'flint'
    def __init__(s, m): s.m = fmpq_poly(m)
    def elem(s, coeffs, denom=1): return fmpq_poly([fmpq(c, denom) for c in coeffs]) % s.m
    def const(s, r): return fmpq_poly([fmpq(r.p, r.q) if isinstance(r, sp.Rational) else fmpq(r)])
    def gen(s): return fmpq_poly([0, 1])
    def mul(s, a, b): return (a * b) % s.m
    def is_zero(s, a): return (a % s.m) == 0


def det(K, M):
    n = len(M)
    if n == 1: return M[0][0]
    out = K.const(0)
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in M[1:]]
        t = K.mul(M[0][j], det(K, minor))
        out = out + t if j % 2 == 0 else out - t
    return out


def exact_checks(K, m, G, caps, N):
    """E2 and E3 in the field K; returns dict of booleans."""
    D = K.gen(); one = K.const(1)
    # E2: rank 3 via the Schur complement (multiplied through by det A)
    A = [[G[i][j] for j in range(3)] for i in range(3)]
    dA = det(K, A)
    adj = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            minor = [[A[r][c] for c in range(3) if c != i] for r in range(3) if r != j]
            v = det(K, minor); adj[i][j] = v if (i + j) % 2 == 0 else K.const(0) - v
    rank3 = True
    for i in range(3, N):
        for j in range(3, N):
            s = K.const(0)
            for k in range(3):
                for l in range(3):
                    s = s + K.mul(K.mul(G[k][i], adj[k][l]), G[l][j])
            if not K.is_zero(K.mul(dA, G[i][j]) - s): rank3 = False
    # E3: cap equations
    cap_ok = []
    for (I, side, n) in caps:
        tau = one - K.const(sp.Rational(2 * n, N)) + D + D
        tau2 = K.mul(tau, tau)
        if len(I) == 2:
            a, b = I
            lhs = tau2; rhs = K.mul(K.const(sp.Rational(1, 2)), one + G[a][b])
        else:
            a, b, c = I
            gI = [[G[i][j] for j in I] for i in I]
            nab = K.const(2) - G[a][b] - G[a][b]; nac = K.const(2) - G[a][c] - G[a][c]
            dot = G[b][c] - G[a][b] - G[a][c] + one
            Q = K.mul(nab, nac) - K.mul(dot, dot)
            lhs = det(K, gI); rhs = K.mul(tau2, Q)
        cap_ok.append(bool(K.is_zero(lhs - rhs)))
    return dict(rank3=rank3, caps=cap_ok)


# ------------------------------------------------------------------ interval checks, two kernels
class ArbI:
    name = 'arb (python-flint)'
    def __init__(s): ctx.prec = 600
    def q(s, r): r = sp.Rational(r); return arb(fmpq(int(r.p), int(r.q)))
    def hull(s, lo, hi): a, b = s.q(lo), s.q(hi); return a.union(b)
    def sqrt(s, v): return v.sqrt()
    def atan2(s, y, xx): return arb.atan2(y, xx)
    def pi(s): return arb.pi()
    def pos(s, v): return bool(v > 0)
    def neg(s, v): return bool(v < 0)


class MpI:
    name = 'mpmath.iv'
    def __init__(s): mp.iv.dps = 180
    def q(s, r): r = sp.Rational(r); return mp.iv.mpf(int(r.p)) / int(r.q)
    def hull(s, lo, hi): a, b = s.q(lo), s.q(hi); return mp.iv.mpf([a.a, b.b])
    def sqrt(s, v): return mp.iv.sqrt(v)
    def atan2(s, y, xx): return mp.iv.atan2(y, xx)
    def pi(s): return mp.iv.pi
    def pos(s, v): return bool(v.a > 0)
    def neg(s, v): return bool(v.b < 0)


def interval_checks(IV, Dlo, Dhi, entries, pattern, caps, N, q_center, r_big, sigma):
    Dv = IV.hull(Dlo, Dhi)
    vals = []
    for e in entries:
        acc = IV.q(0); p = IV.q(1)
        for c in e['in_QD']['coeffs']:
            acc = acc + IV.q(c) * p; p = p * Dv
        vals.append(acc / IV.q(e['in_QD']['denom']))
    G = [[IV.q(1) if i == j else vals[pattern[i][j]] for j in range(N)] for i in range(N)]
    one = IV.q(1)
    # I1: A positive definite
    m2 = one - G[0][1] * G[0][1]
    dA = (G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1]) - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
          + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]))
    pd = IV.pos(m2) and IV.pos(dA)
    # realization in the gauge: x0 = pole, x1 = (sin th1, 0, cos th1), y2 = sigma * |y2|
    X = [[IV.q(0), IV.q(0), one]]
    s1 = IV.sqrt(m2); X.append([s1, IV.q(0), G[0][1]])
    for i in range(2, N):
        z = G[0][i]; xx = (G[1][i] - G[0][1] * z) / s1
        if i == 2:
            y = IV.sqrt(one - z * z - xx * xx) * sigma
        else:
            y = (G[2][i] - X[2][0] * xx - X[2][2] * z) / X[2][1]
        X.append([xx, y, z])
    # I3: parameters in B (angles compared modulo 2 pi)
    qe = [IV.atan2(X[1][0], X[1][2])]
    for i in range(2, N):
        r = IV.sqrt(X[i][0] * X[i][0] + X[i][1] * X[i][1])
        qe += [IV.atan2(r, X[i][2]), IV.atan2(X[i][1], X[i][0])]
    inB = True; twopi = IV.pi() * 2
    for k, (v, c) in enumerate(zip(qe, q_center)):
        d = v - IV.q(sp.Rational(c))
        if k >= 2 and k % 2 == 0:           # longitude
            dm = mid(d); kk = -round(dm / (2 * np.pi)); d = d + twopi * kk
        if not (IV.neg(d - IV.q(sp.Rational(r_big))) and IV.pos(d + IV.q(sp.Rational(r_big)))): inB = False
    # I2: signs
    signs = []
    for (I, side, n) in caps:
        tau = one - IV.q(sp.Rational(2 * n, N)) + Dv + Dv
        if len(I) == 2:
            signs.append(IV.pos(tau)); continue
        a, b, c = (X[i] for i in I)
        dt = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) * side
        signs.append((IV.pos(dt) and IV.pos(tau)) or (IV.neg(dt) and IV.neg(tau)))
    return dict(positive_definite=pd, in_box_B=inB, signs=signs)


def mid(v):
    if isinstance(v, arb): return float(v.mid())
    return float(v.mid) if hasattr(v, 'mid') else float((v.a + v.b) / 2)


def parse_cap(s):
    # 'Cap(0, 1, 2)+'
    I = tuple(int(t) for t in s[s.index('(') + 1:s.index(')')].split(',') if t.strip())
    return I, (1 if s.endswith('+') else -1)


def run(name):
    ge = json.load(open(os.path.join(ROOT, 'sphcap/data/gram_exact.json')))[name]
    lc = json.load(open(os.path.join(CERT, f'localcert_{name}.json')))
    N = ge['N']; m = ge['D_minpoly_ascending']; img = lc['images'][0]
    rep = dict(name=name, N=N, image=img['image'], local_certificate=f'certificates/localcert_{name}.json')
    # E1: m(x) ~ P_N(N x); D_e in I_N / N
    PN = sp.Poly(sum(c * x ** (12 - i) for i, c in enumerate(P[N])), x)
    PNx = sp.Poly(PN.as_expr().subs(x, N * x), x)
    mm = sp.Poly(list(reversed(m)), x)
    e1 = (PNx.primitive()[1] == mm.primitive()[1]) or (PNx.primitive()[1] == -mm.primitive()[1])
    pm, pf = fmpq_poly(m), fmpq_poly([int(c) for c in reversed(PNx.all_coeffs())])
    e1_flint = (pm * pf.coeffs()[-1] == pf * pm.coeffs()[-1])
    lo, hi = sp.Rational(I_TH[N][0]) / N, sp.Rational(I_TH[N][1]) / N
    nroots = mm.count_roots(lo, hi)
    iv = [iv for iv, _ in mm.intervals(eps=sp.Rational(1, 10 ** 120)) if iv[0] >= lo and iv[1] <= hi]
    assert nroots == 1 and len(iv) == 1, (nroots, iv)
    Dlo, Dhi = iv[0]
    rep['E1_minpoly_is_P_N'] = dict(sympy=bool(e1), flint=bool(e1_flint), roots_in_theorem_interval=int(nroots))
    # labelling: permutation pi with Gram(image 0)[i][j] = G_e[pi i][pi j]
    nq = 2 * N - 3
    Pq = Cf.points_ad([arb(v) for v in img['q_center']], N)
    Xc = np.array([[float(c.v.mid()) for c in p] for p in Pq]); Gc = Xc @ Xc.T
    vals = [float(e['value']) for e in ge['entries']]
    Gn = np.array([[1.0 if i == j else vals[ge['gram_pattern'][i][j]] for j in range(N)] for i in range(N)])
    perm = next(p for p in itertools.permutations(range(N)) if np.allclose(Gn[np.ix_(p, p)], Gc, atol=1e-9))
    pattern = [[ge['gram_pattern'][perm[i]][perm[j]] for j in range(N)] for i in range(N)]
    sigma = 1 if Xc[2][1] > 0 else -1
    rep['relabelling'] = list(perm); rep['orientation_y2'] = sigma
    caps = [parse_cap(c) + (n,) for c, n in zip(img['caps'] + img['extra_caps'], img['counts'] + img['extra_counts'])]
    rep['caps'] = [f"Cap{I}{'+' if s > 0 else '-'} n={n}" for I, s, n in caps]
    rep['square_system'] = len(img['caps'])
    # exact checks
    for Kc in (SympyK, FlintK):
        K = Kc(m)
        el = [K.elem(e['in_QD']['coeffs'], e['in_QD']['denom']) for e in ge['entries']]
        G = [[K.const(1) if i == j else el[pattern[i][j]] for j in range(N)] for i in range(N)]
        r = exact_checks(K, m, G, caps, N)
        rep[f'exact_{K.name}'] = dict(E2_rank3=r['rank3'], E3_caps=r['caps'], all=bool(r['rank3'] and all(r['caps'])))
        print(f"{name}: exact ({K.name}): rank 3 {r['rank3']}, cap equations {sum(r['caps'])}/{len(r['caps'])}", flush=True)
    for IVc in (ArbI, MpI):
        IV = IVc()
        r = interval_checks(IV, Dlo, Dhi, ge['entries'], pattern, caps, N, img['q_center'], lc['box_radius'], sigma)
        r['all'] = bool(r['positive_definite'] and r['in_box_B'] and all(r['signs']))
        rep[f'interval_{IV.name.split()[0]}'] = r
        print(f"{name}: interval ({IV.name}): PD {r['positive_definite']}, in B {r['in_box_B']}, signs {sum(r['signs'])}/{len(r['signs'])}", flush=True)
    ok_e = [rep['exact_sympy']['all'], rep['exact_flint']['all']]
    ok_i = [rep['interval_arb']['all'], rep['interval_mpmath.iv']['all']]
    e1ok = rep['E1_minpoly_is_P_N']['sympy'] and rep['E1_minpoly_is_P_N']['flint']
    rep['verified'] = bool(e1ok and all(ok_e) and all(ok_i) and lc.get('verified'))
    rep['kernels'] = 'exact: sympy + FLINT (fmpq_poly); intervals: Arb (python-flint) + mpmath.iv'
    rep['label'] = 'cross-verified (exact: sympy + FLINT; intervals: Arb + mpmath.iv)' if rep['verified'] else 'failed'
    rep['atlas_level'] = 3 if rep['verified'] else None
    rep['statement'] = (f"The Gram matrix of the minimizer of Theorem 10 (N = {N}) is G_e of sphcap/data/gram_exact.json "
                        f"({name}), relabelled by 'relabelling'; its entries lie in Q(D*), with D* the root of "
                        f"D_minpoly_ascending in [{float(lo):.15f}, {float(hi):.15f}].")
    rep['D_isolating_interval'] = [str(Dlo), str(Dhi)]
    json.dump(rep, open(os.path.join(CERT, f'gramcert_{name}.json'), 'w'), indent=1)
    print(f"{name}: {'VERIFIED' if rep['verified'] else 'FAILED'} -> {rep['label']}", flush=True)
    return rep['verified']


if __name__ == '__main__':
    names = sys.argv[1:] or ['N5_best', 'N6_best']
    ok = all([run(n) for n in names])
    sys.exit(0 if ok else 1)
