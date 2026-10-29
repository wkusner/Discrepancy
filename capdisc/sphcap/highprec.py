"""
sphcap.highprec
===============

Arbitrary-precision polishing of a balanced configuration.

Given a configuration X with active branch list A (from ``balance_report``) we
gauge-fix (x_0 = north pole, x_1 in the xz-plane, so 2N-3 coordinates q) and solve

    f_b(q) - s = 0         (b in A)             |A| equations
    sum_b lam_b d f_b/dq_j = 0   (j = 1..2N-3)  balance (force) equations
    sum_b lam_b - 1 = 0

in unknowns (q, s, lam).  The system is square iff |A| = 2N-2 (generic for an
isolated sharp minimiser); then Newton converges quadratically and the
value s = D can be computed to hundreds of digits, after which an integer
relation search (``identify``) gives its minimal polynomial.
"""
from __future__ import annotations

import numpy as np
import mpmath as mp

from . import balance as B


# ---------------------------------------------------------------------------
def gauge_fix(X):
    """Rotate (and reflect if needed) so that x_0 = e_3 and x_1 lies in the xz-plane (y >= 0 ok)."""
    X = B.normalize(X)
    z = X[0]
    # rotation taking z to e3
    e3 = np.array([0, 0, 1.0])
    v = np.cross(z, e3)
    c = float(z @ e3)
    if np.linalg.norm(v) < 1e-14:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K / (1 + c)
    Y = X @ R.T
    phi = np.arctan2(Y[1, 1], Y[1, 0])
    Rz = np.array([[np.cos(-phi), -np.sin(-phi), 0], [np.sin(-phi), np.cos(-phi), 0], [0, 0, 1]])
    Y = Y @ Rz.T
    return Y


def to_q(Y):
    """Spherical coordinates of the gauge-fixed configuration (theta_1, then theta_i, phi_i)."""
    N = len(Y)
    q = [np.arccos(np.clip(Y[1, 2], -1, 1))]
    for i in range(2, N):
        q += [np.arccos(np.clip(Y[i, 2], -1, 1)), np.arctan2(Y[i, 1], Y[i, 0])]
    return q


def build_points(q, N):
    """x_i(q) and dx_i/dq_j as mp vectors; returns X (list of 3-lists) and dX[j][i] (3-lists)."""
    zero = mp.mpf(0)
    X = [[zero, zero, mp.mpf(1)]]
    dX = [[[zero] * 3 for _ in range(N)] for _ in range(len(q))]
    th1 = q[0]
    X.append([mp.sin(th1), zero, mp.cos(th1)])
    dX[0][1] = [mp.cos(th1), zero, -mp.sin(th1)]
    for i in range(2, N):
        th, ph = q[2 * i - 3], q[2 * i - 2]
        X.append([mp.sin(th) * mp.cos(ph), mp.sin(th) * mp.sin(ph), mp.cos(th)])
        dX[2 * i - 3][i] = [mp.cos(th) * mp.cos(ph), mp.cos(th) * mp.sin(ph), -mp.sin(th)]
        dX[2 * i - 2][i] = [-mp.sin(th) * mp.sin(ph), mp.sin(th) * mp.cos(ph), zero]
    return X, dX


def _sub(a, b): return [a[k] - b[k] for k in range(3)]
def _dot(a, b): return sum(a[k] * b[k] for k in range(3))
def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def _norm(a): return mp.sqrt(_dot(a, a))


def cap_geometry_mp(X, I):
    """(t, w, beta) in mp arithmetic; same conventions as balance.cap_geometry."""
    if len(I) == 2:
        s = [X[I[0]][k] + X[I[1]][k] for k in range(3)]
        ns = _norm(s)
        return ns / 2, [c / ns for c in s], [mp.mpf(1) / 2, mp.mpf(1) / 2]
    a, b, c = X[I[0]], X[I[1]], X[I[2]]
    u, v = _sub(b, a), _sub(c, a)
    n = _cross(u, v)
    nn = _norm(n)
    w = [x / nn for x in n]
    t = _dot(w, a)
    f = [t * x for x in w]
    r = _sub(f, a)
    M = mp.matrix([[_dot(u, u), _dot(u, v)], [_dot(u, v), _dot(v, v)]])
    rhs = mp.matrix([_dot(u, r), _dot(v, r)])
    g = mp.lu_solve(M, rhs)
    return t, w, [1 - g[0] - g[1], g[0], g[1]]


def make_system(N, spec):
    """spec = list of (I, kind, n_cl, n_op).  Returns F(*unknowns) for mp.findroot."""
    m = len(spec)
    nq = 2 * N - 3

    def F(*u):
        q = list(u[:nq])
        s = u[nq]
        lam = list(u[nq + 1:nq + 1 + m])
        X, dX = build_points(q, N)
        vals, grads = [], []
        for (I, kind, ncl, nop) in spec:
            t, w, beta = cap_geometry_mp(X, I)
            V = (1 - t) / 2
            sign = 1 if kind == "up" else -1
            vals.append(mp.mpf(ncl) / N - V if kind == "up" else V - mp.mpf(nop) / N)
            # d f / d q_j = sum_l (1/2) sign beta_l <w, dx_l/dq_j>
            gj = []
            for j in range(nq):
                gj.append(sum(sign * beta[k] * _dot(w, dX[j][l]) / 2 for k, l in enumerate(I)))
            grads.append(gj)
        eqs = [vals[b] - s for b in range(m)]
        eqs += [sum(lam[b] * grads[b][j] for b in range(m)) for j in range(nq)]
        eqs.append(sum(lam) - 1)
        return eqs

    return F


def polish(X, report=None, dps=60, act_tol=1e-6, maxsteps=60):
    """Newton polish of a balanced configuration with |A| = 2N-2; returns dict with mp results."""
    N = len(X)
    rep = report or B.balance_report(X, act_tol=act_tol)
    act = rep.active
    spec = [(b.I, b.kind, b.n_cl, b.n_op) for b in act]
    if len(spec) != 2 * N - 2:
        raise ValueError(f"system is not square: |A|={len(spec)} != 2N-2={2 * N - 2}")
    Y = gauge_fix(X)
    q0 = to_q(Y)
    mp.mp.dps = dps
    F = make_system(N, spec)
    x0 = [mp.mpf(float(v)) for v in q0] + [mp.mpf(rep.D)] + [mp.mpf(float(l)) for l in rep.lam]
    sol = mp.findroot(F, x0, tol=mp.mpf(10) ** (-dps + 8), maxsteps=maxsteps, verify=False)
    nq = 2 * N - 3
    q = [sol[i] for i in range(nq)]
    return dict(q=q, D=sol[nq], lam=[sol[nq + 1 + i] for i in range(len(spec))], spec=spec,
                residual=max(abs(e) for e in F(*sol)))


def identify(x, max_degree=24, dps=None, maxcoeff=10**8):
    """Search for a minimal polynomial of the mp number x (degree <= max_degree)."""
    for d in range(1, max_degree + 1):
        p = mp.findpoly(x, d, maxcoeff=maxcoeff, tol=mp.mpf(10) ** (-(mp.mp.dps - 12)), maxsteps=100000)
        if p:
            return d, p
    return None


# ---------------------------------------------------------------------------
# value-equation polish (no multipliers): the location of a sharp minimiser is
# fixed by the area conditions f_b = s alone; lambda is then a linear solve.
# ---------------------------------------------------------------------------
def polish_values(X, report=None, dps=60, act_tol=1e-6, maxsteps=60):
    """Solve f_b(q) = s for an independent subset of 2N-2 active branches.

    Returns dict(q, D, spec, used, extra_residual).  Redundant active branches
    (|A| > 2N-2, forced by symmetry) are checked, not used.
    """
    import scipy.linalg as sl
    N = len(X)
    rep = report or B.balance_report(X, act_tol=act_tol)
    act = rep.active
    spec_all = [(b.I, b.kind, b.n_cl, b.n_op) for b in act]
    Y = gauge_fix(X)
    q0 = to_q(Y)
    nq = 2 * N - 3

    def vals_mp(q, spec):
        Xm, _ = build_points(q, N)
        out = []
        for (I, kind, ncl, nop) in spec:
            t, w, beta = cap_geometry_mp(Xm, I)
            V = (1 - t) / 2
            out.append(mp.mpf(ncl) / N - V if kind == "up" else V - mp.mpf(nop) / N)
        return out

    mp.mp.dps = max(dps, 30)
    # choose independent rows via pivoted QR on the float Jacobian [df/dq, -1]
    h = 1e-7
    def fv(q):
        Xf = np.array([[float(c) for c in p] for p in build_points([mp.mpf(v) for v in q], N)[0]])
        return np.array([B.frozen_value(Xf, b) for b in act])
    J = np.array([(fv(np.array(q0) + h * e) - fv(np.array(q0) - h * e)) / (2 * h) for e in np.eye(nq)]).T
    J = np.hstack([J, -np.ones((len(act), 1))])
    _, _, piv = sl.qr(J.T, pivoting=True)
    used = sorted(piv[:nq + 1])
    spec = [spec_all[i] for i in used]
    rest = [spec_all[i] for i in range(len(act)) if i not in used]

    def F(*u):
        q = list(u[:nq]); s = u[nq]
        return [v - s for v in vals_mp(q, spec)]

    x0 = [mp.mpf(float(v)) for v in q0] + [mp.mpf(rep.D)]
    sol = mp.findroot(F, x0, tol=mp.mpf(10) ** (-dps + 8), maxsteps=maxsteps, verify=False)
    q = [sol[i] for i in range(nq)]
    s = sol[nq]
    extra = max([abs(v - s) for v in vals_mp(q, rest)] + [mp.mpf(0)]) if rest else mp.mpf(0)
    return dict(q=q, D=s, spec=spec, used=used, extra_residual=extra, spec_all=spec_all,
                residual=max(abs(e) for e in F(*sol)))


# ---------------------------------------------------------------------------
# integer relations via LLL (python-flint); much faster than mpmath.findpoly
# ---------------------------------------------------------------------------
def identify_lll(x, max_degree=40, digits=None, check_digits=None):
    """Minimal polynomial candidate of the mp real x by LLL on [1, x, ..., x^d].

    Tries d = 1..max_degree; uses `digits` (default: mp.dps - 20) for the lattice
    and accepts a relation only if it also holds to `check_digits` (default:
    mp.dps - 10) and the polynomial is irreducible.  Returns (d, coeffs ascending)
    or None.  A returned relation is a *numerical* identification.
    """
    import flint
    dps = mp.mp.dps
    digits = digits or dps - 20
    check_digits = check_digits or dps - 10
    scale = mp.mpf(10) ** digits
    pw = [mp.mpf(1)]
    for _ in range(max_degree):
        pw.append(pw[-1] * x)
    for d in range(1, max_degree + 1):
        rows = []
        for i in range(d + 1):
            r = [0] * (d + 2)
            r[i] = 1
            r[d + 1] = int(mp.nint(pw[i] * scale))
            rows.append(r)
        M = flint.fmpz_mat(rows).lll()
        for k in range(M.nrows()):
            c = [int(M[k, i]) for i in range(d + 1)]
            if c[d] == 0 or all(v == 0 for v in c):
                continue
            val = mp.fsum(mp.mpf(ci) * pw[i] for i, ci in enumerate(c))
            hmax = max(abs(v) for v in c)
            # accept only if the relation is far stronger than a generic LLL vector would give
            if (d + 1) * mp.log10(hmax + 1) < 0.7 * digits and abs(val) < mp.mpf(10) ** (-check_digits) * hmax:
                p = flint.fmpz_poly(c)
                fac = p.factor()[1]
                if len(fac) == 1 and fac[0][1] == 1 and p.degree() == d:
                    if c[-1] < 0:
                        c = [-v for v in c]
                    return d, c
        # no relation of degree d
    return None


def express_in_field(x, alpha, degree, digits=None, check_digits=None):
    """Find integers (a, b_0..b_{d-1}) with a*x = sum b_i alpha^i (x in Q(alpha)).

    Returns (a, [b_0..b_{d-1}]) or None.  Numerical identification (LLL).
    """
    import flint
    dps = mp.mp.dps
    digits = digits or dps - 20
    check_digits = check_digits or dps - 10
    scale = mp.mpf(10) ** digits
    vec = [x] + [alpha ** i for i in range(degree)]
    n = len(vec)
    rows = []
    for i in range(n):
        r = [0] * (n + 1)
        r[i] = 1
        r[n] = int(mp.nint(vec[i] * scale))
        rows.append(r)
    M = flint.fmpz_mat(rows).lll()
    for k in range(M.nrows()):
        c = [int(M[k, i]) for i in range(n)]
        if c[0] == 0:
            continue
        hmax = max(abs(v) for v in c)
        val = mp.fsum(mp.mpf(ci) * vi for ci, vi in zip(c, vec))
        if n * mp.log10(hmax + 1) < 0.7 * digits and abs(val) < mp.mpf(10) ** (-check_digits) * hmax:
            a, b = c[0], [-v for v in c[1:]]
            if a < 0:
                a, b = -a, [-v for v in b]
            return a, b
    return None
