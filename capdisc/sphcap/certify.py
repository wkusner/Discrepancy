"""
sphcap.certify
==============

Rigorous (ball-arithmetic) cell bounds for the spherical cap discrepancy, used to refine the
cells left over by the fast C certifier near minimizers.

Coordinates q (2N-3) as in bnb.py / capcert.c:
    x0 = (0,0,1),  x1 = (sin q0, 0, cos q0),  x_i = (sin th cos ph, sin th sin ph, cos th),
    th = q[2i-3], ph = q[2i-2].

For a cell B = c + [-w, w] and a cap C defined by points I (a pair: the smallest cap; a triple:
one of the two caps of its circle), the over-count satisfies, for all Y in B,

    f_C(Y) >= n_B/N - sigma(C(Y)) =: phi(Y),

where n_B counts the points that are in the closed cap C(Y) for every Y in B (points of I always;
others decided by an interval sign test).  By the mean-value theorem
    phi(Y) >= phi(c) + sum_j [ m_j v_j - r_j |v_j| ],   v = Y - c,  G_j = m_j +- r_j  ⊇  d phi/dq_j (B),
so phi(Y) >= a + m.v with a = phi(c) - sum_j r_j w_j.  For ANY weights lambda >= 0 summing to 1,

    D(Y) >= max_C phi_C(Y) >= sum_C lambda_C a_C - sum_j |sum_C lambda_C m_Cj| w_j     (*)

(the LP dual of  min_v max_C (a_C + m_C.v)).  We choose lambda by a floating-point LP and then
evaluate (*) in ball arithmetic, so the bound is rigorous whatever the LP solver does.
Near a sharp minimizer (*) is second-order accurate, which avoids the cluster problem.
"""
from __future__ import annotations

import itertools
import numpy as np
import flint
from flint import arb

ZERO = arb(0)
ONE = arb(1)


# ---------------------------------------------------------------- forward-mode AD over arb
class AD:
    __slots__ = ("v", "d")

    def __init__(self, v, d=None):
        self.v = v if isinstance(v, arb) else arb(v)
        self.d = d or {}

    def _lift(o):
        return o if isinstance(o, AD) else AD(o)

    def __add__(a, b):
        b = AD._lift(b); d = dict(a.d)
        for k, x in b.d.items(): d[k] = d[k] + x if k in d else x
        return AD(a.v + b.v, d)
    __radd__ = __add__

    def __neg__(a):
        return AD(-a.v, {k: -x for k, x in a.d.items()})

    def __sub__(a, b): return a + (-AD._lift(b))
    def __rsub__(a, b): return AD._lift(b) + (-a)

    def __mul__(a, b):
        b = AD._lift(b); d = {k: x * b.v for k, x in a.d.items()}
        for k, x in b.d.items(): d[k] = d[k] + a.v * x if k in d else a.v * x
        return AD(a.v * b.v, d)
    __rmul__ = __mul__

    def __truediv__(a, b):
        b = AD._lift(b); inv = 1 / b.v
        d = {k: x * inv for k, x in a.d.items()}
        q = a.v * inv
        for k, x in b.d.items():
            t = -q * x * inv
            d[k] = d[k] + t if k in d else t
        return AD(q, d)

    def sqrt(a):
        s = a.v.sqrt(); h = 1 / (2 * s)
        return AD(s, {k: x * h for k, x in a.d.items()})


def ad_sin_cos(x: AD):
    s, c = x.v.sin(), x.v.cos()
    return AD(s, {k: v * c for k, v in x.d.items()}), AD(c, {k: -v * s for k, v in x.d.items()})


def ball(lo, hi):
    return arb((lo + hi) / 2, (hi - lo) / 2)


def points_ad(qint, N):
    """qint: list of arb (point values or intervals).  Returns list of 3-lists of AD."""
    P = [[AD(0), AD(0), AD(1)]]
    q0 = AD(qint[0], {0: ONE})
    s0, c0 = ad_sin_cos(q0)
    P.append([s0, AD(0), c0])
    for i in range(2, N):
        th = AD(qint[2 * i - 3], {2 * i - 3: ONE}); ph = AD(qint[2 * i - 2], {2 * i - 2: ONE})
        st, ct = ad_sin_cos(th); sp, cp = ad_sin_cos(ph)
        P.append([st * cp, st * sp, ct])
    return P


def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


# ---------------------------------------------------------------- caps
class CapSpec:
    """A cap kind: (I, side).  |I|=2: smallest cap (side ignored); |I|=3: side = +1/-1."""
    __slots__ = ("I", "side")

    def __init__(self, I, side=1):
        self.I = tuple(I); self.side = side

    def __repr__(self): return f"Cap{self.I}{'+' if self.side > 0 else '-'}"


def cap_terms(P, cap):
    """Return (t, margin_fn) where t is the signed level (AD) and margin_fn(x) >= 0 iff x in cap."""
    I = cap.I
    if len(I) == 2:
        S = add(P[I[0]], P[I[1]]); SS = dot(S, S); nS = SS.sqrt()
        t = nS * 0.5
        return t, (lambda x: dot(S, x) * 2 - SS)
    a, b, c = (P[i] for i in I)
    n = cross(sub(b, a), sub(c, a)); nn = dot(n, n).sqrt(); tr = dot(n, a)
    s = cap.side
    t = tr * s / nn
    return t, (lambda x: (dot(n, x) - tr) * s)


def all_caps(N):
    caps = [CapSpec(I) for I in itertools.combinations(range(N), 2)]
    for I in itertools.combinations(range(N), 3):
        caps += [CapSpec(I, 1), CapSpec(I, -1)]
    return caps


def _mv_interval(center_val: arb, grad_box: dict, w):
    """mean-value enclosure over the box: center value + sum_j G_j * [-w_j, w_j]."""
    out = center_val
    for k, g in grad_box.items():
        out = out + g * arb(0, w[k])
    return out


def cap_data(qc, w, N, caps):
    """For each cap: (a_lower_center_value, mid_grad (np), rad_grad (np), robust_count) or None.
    phi = n/N - (1 - t)/2.  Uses center evaluation (point balls) for values and box evaluation
    for gradients."""
    nq = 2 * N - 3
    qcb = [arb(x) for x in qc]
    qbox = [arb(qc[k], w[k]) for k in range(nq)]
    Pc = points_ad(qcb, N); Pb = points_ad(qbox, N)
    out = []
    for cap in caps:
        try:
            tc, mc = cap_terms(Pc, cap)
            tb, mb = cap_terms(Pb, cap)
        except Exception:
            out.append(None); continue
        if not (tc.v.is_finite() and tb.v.is_finite()):
            out.append(None); continue
        # robust count: points of I always in; others need a certain sign over the box
        n = len(cap.I); n_up = n
        for j in range(N):
            if j in cap.I: continue
            mj = _mv_interval(mc(Pc[j]).v, mb(Pb[j]).d, w)
            if mj > 0: n += 1; n_up += 1
            elif mj < 0: pass
            else:        # undecided: outside for the lower bound, inside for the upper bound
                n_up += 1
        phi_c = arb(n) / N - (1 - tc.v) / 2
        mid = np.zeros(nq); rad = np.zeros(nq)
        for k, g in tb.d.items():
            gg = g / 2               # d phi / dq = + dt/dq / 2
            mid[k] = float(gg.mid()); rad[k] = (float(gg.rad()) + 2.3e-16 * abs(mid[k]) + 1e-300) * (1 + 1e-12)
        a = phi_c - arb(float(np.dot(rad, w)) * (1 + 1e-12) + 1e-300)
        # account for rounding of mid to float: absorb into rad (done above)
        # upper enclosure of phi over the box (count upper bound n_up), for 'below D*' checks
        phi_up = (arb(n_up) / N - (1 - tc.v) / 2) + arb(float(np.dot(np.abs(mid) + rad, w)) * (1 + 1e-12) + 1e-300)
        out.append((a, mid, rad, n, phi_up, n_up))
    return out


def cell_lower_bound(qc, w, N, T, caps=None, use_lp=True):
    """Rigorous lower bound for D over the cell (arb).  Returns (lb_float_lower, info)."""
    from scipy.optimize import linprog
    caps = caps or all_caps(N)
    data = cap_data(qc, w, N, caps)
    rows = [(i, d) for i, d in enumerate(data) if d is not None]
    best = arb(1) / N
    # single caps
    for i, (a, mid, rad, n, _pu, _nu) in rows:
        lb1 = a - arb(float(np.dot(np.abs(mid), w)) * (1 + 1e-12) + 1e-300)
        if lb1 > best: best = lb1
    if best >= T or not use_lp:
        return best, dict(method='single')
    # LP: min s s.t. s >= a_b + m_b.v, -w <= v <= w
    A = np.array([np.append(d[1], -1.0) for _, d in rows])
    av = np.array([float(d[0].mid()) for _, d in rows])
    nq = len(w)
    c = np.zeros(nq + 1); c[-1] = 1
    res = linprog(c, A_ub=A, b_ub=-av, bounds=[(-wk, wk) for wk in w] + [(None, None)], method='highs')
    if res.status != 0:
        return best, dict(method='lp-failed')
    lam = np.maximum(-res.ineqlin.marginals, 0.0)
    if lam.sum() <= 0:
        return best, dict(method='lp-zero')
    lam = lam / lam.sum()
    # rigorous evaluation of (*) with these lambda (exact rationals not needed: any lam>=0, renormalized in arb)
    L = [arb(float(x)) for x in lam]; S = sum(L, ZERO)
    acc = ZERO
    for (i, d), l in zip(rows, L):
        acc = acc + l * d[0]
    gsum = np.zeros(nq)
    for (i, d), l in zip(rows, lam):
        gsum += l * d[1]
    # bound |sum lam m| rigorously: the float sum has rounding error <= 1e-15 * sum|...|
    err = 1e-12 * sum(abs(l) * np.abs(d[1]) for (_, d), l in zip(rows, lam))
    pen = arb(float(np.dot(np.abs(gsum) + err, w)) * (1 + 1e-12))
    lb = (acc - pen) / S
    if lb > best: best = lb
    return best, dict(method='lp', lp_value=res.fun)


# ---------------------------------------------------------------- images in the fundamental domain
def gauge_coords(Y):
    """gauge-fix (x0 -> pole, x1 -> xz-plane, x>=0) and return (Yg, q) in capcert coordinates."""
    from . import highprec as H
    Yg = H.gauge_fix(Y)
    if Yg[1, 0] < 0:
        Yg = Yg * np.array([-1, -1, 1.0])          # rotate by pi about z
    q = np.array(H.to_q(Yg), float)
    return Yg, q


def in_fundamental_domain(q, N, tol=1e-9):
    """capcert's domain: (x0,x1) a closest pair, th_2<=th_3<=..., ph_2 in [0,pi]."""
    from . import bnb
    X = bnb.points(q[None], N)[0]
    G = np.clip(X @ X.T, -1, 1); A = np.arccos(G)
    th1 = q[0]
    for i, j in itertools.combinations(range(N), 2):
        if (i, j) != (0, 1) and A[i, j] < th1 - tol: return False
    for i in range(2, N - 1):
        if q[2 * i - 3] > q[2 * (i + 1) - 3] + tol: return False
    if N >= 3 and not (-tol <= q[2] <= np.pi + tol): return False
    return True


def domain_images(X, tol=1e-9):
    """All relabeled/reflected copies of X, gauge-fixed, that lie in capcert's fundamental domain."""
    N = len(X); out = []
    for perm in itertools.permutations(range(N)):
        for refl in (1, -1):
            Y = X[list(perm)] * np.array([1, refl, 1.0])
            Yg, q = gauge_coords(Y)
            for flip in (1, -1):                        # reflection y -> -y keeps x0, x1
                Yf = Yg * np.array([1, flip, 1.0]); qf = q.copy()
                for i in range(2, N): qf[2 * i - 2] = flip * q[2 * i - 2]
                qf[[2 * i - 2 for i in range(2, N)]] = (qf[[2 * i - 2 for i in range(2, N)]] + np.pi) % (2 * np.pi) - np.pi
                if in_fundamental_domain(qf, N, tol) and not any(np.allclose(qf, o[1], atol=1e-7) for o in out):
                    out.append((Yf, qf))
    return out
