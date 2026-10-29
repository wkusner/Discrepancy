"""
sphcap.balance
==============

Balance (KKT / Clarke) conditions for the spherical cap discrepancy on S^2.

Notation (2016 lecture notation, see the accompanying note)::

    X = (x_1,...,x_N) in (S^2)^N,   G = X^T X   (Gram matrix, rank <= 3, diag = 1)
    D(X) = sup_C | #(X cap C)/N - sigma(C) |,   sigma(C(w,t)) = (1-t)/2.

Branches.  For a boundary set I (|I| = 2 or 3) let P_I be the plane through the
points of I, w_I its unit normal and t_I = <w_I, x_i> (i in I) the *signed*
distance of P_I from the origin (for |I| = 2: the minimal cap, t_I = |x_a+x_b|/2).
With V_I = (1 - t_I)/2 and the closed / open counts

    n_cl = #{j : <x_j,w_I> >= t_I},     n_op = #{j : <x_j,w_I> > t_I}

the two *branch functions* are

    up_I  = n_cl/N - V_I          (closed cap)
    low_I = V_I - n_op/N          (open cap).

Heitsch--Henrion: D = max over all I of max(up_I, low_I) (plus the constant 1/N
coming from |I| = 1).  Moving the boundary points moves t_I smoothly:

    grad_{x_l} t_I = beta_l * w_I ,   l in I,

beta = barycentric coordinates of the foot t_I w_I of the origin on the plane
(for |I| independent this is t_I^2 G_I^{-1} 1, HH Lemma 4.5).  The *force* a
cap exerts on its boundary point x_l is therefore

    grad_{x_l} up_I = +(1/2) beta_l P_l w_I ,   grad_{x_l} low_I = -(1/2) beta_l P_l w_I ,

P_l = tangent projection at x_l.  Balance (Clarke stationarity / KKT of the
frozen-count minimax problem): there are lambda_b >= 0, sum lambda_b = 1 over the
active branches with  sum_b lambda_b grad_{x_l} f_b = 0  for every l.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import linprog, minimize


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class Branch:
    I: tuple
    kind: str            # "up", "low" or "const"
    value: float
    t: float
    w: Optional[np.ndarray]
    beta: Optional[np.ndarray]
    n_cl: int
    n_op: int
    extra: int           # boundary points of the cap that are NOT in I (incidences)

    @property
    def sign(self) -> int:
        return +1 if self.kind == "up" else -1

    def label(self) -> str:
        return f"{self.kind}{list(self.I)}"


def normalize(X):
    X = np.asarray(X, float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def cap_geometry(X, I, tol=1e-12):
    """(t, w, beta) for the plane through the points of I, or None if degenerate."""
    if len(I) == 2:
        a, b = X[I[0]], X[I[1]]
        s = a + b
        ns = np.linalg.norm(s)
        if ns < 1e-9:
            return None
        return ns / 2.0, s / ns, np.array([0.5, 0.5])
    if len(I) == 3:
        a, b, c = X[I[0]], X[I[1]], X[I[2]]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return None
        w = n / nn
        t = float(w @ a)
        f = t * w
        A = np.stack([b - a, c - a], axis=1)
        gam, *_ = np.linalg.lstsq(A, f - a, rcond=None)
        return t, w, np.array([1 - gam.sum(), gam[0], gam[1]])
    raise ValueError("|I| must be 2 or 3")


def branches(X, tol=1e-9) -> List[Branch]:
    """All branch functions (up/low for every |I| in {2,3}, plus the constant 1/N)."""
    X = normalize(X)
    N = len(X)
    out = [Branch((i,), "const", 1.0 / N, 1.0, None, None, 1, 1, 0) for i in range(min(N, 1))]
    for k in (2, 3):
        for I in itertools.combinations(range(N), k):
            g = cap_geometry(X, I)
            if g is None:
                continue
            t, w, beta = g
            proj = X @ w
            V = (1 - t) / 2
            n_cl = int(np.count_nonzero(proj >= t - tol))
            n_op = int(np.count_nonzero(proj > t + tol))
            bnd = int(np.count_nonzero(np.abs(proj - t) <= tol))
            extra = bnd - k
            out.append(Branch(I, "up", n_cl / N - V, t, w, beta, n_cl, n_op, extra))
            out.append(Branch(I, "low", V - n_op / N, t, w, beta, n_cl, n_op, extra))
    return out


def discrepancy(X, tol=1e-9) -> float:
    return max(b.value for b in branches(X, tol))


def active_set(X, tol=1e-9, act_tol=1e-7):
    bs = branches(X, tol)
    D = max(b.value for b in bs)
    return D, [b for b in bs if b.value >= D - act_tol]


# ---------------------------------------------------------------------------
# tangent coordinates, gradients, rotations
# ---------------------------------------------------------------------------
def tangent_frames(X):
    """Orthonormal tangent basis (N,2,3) at each point."""
    N = len(X)
    E = np.zeros((N, 2, 3))
    for i, x in enumerate(X):
        a = np.array([1.0, 0, 0]) if abs(x[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = a - (a @ x) * x
        e1 /= np.linalg.norm(e1)
        E[i, 0], E[i, 1] = e1, np.cross(x, e1)
    return E


def branch_gradient(X, b: Branch, E=None):
    """Gradient of branch function in tangent coordinates, shape (N,2)."""
    N = len(X)
    E = tangent_frames(X) if E is None else E
    g = np.zeros((N, 2))
    if b.kind == "const":
        return g
    for beta_l, l in zip(b.beta, b.I):
        F3 = 0.5 * b.sign * beta_l * b.w          # ambient force on x_l
        g[l] = E[l] @ F3                          # tangent projection in the frame
    return g


def gradient_matrix(X, bs):
    E = tangent_frames(X)
    return np.array([branch_gradient(X, b, E).ravel() for b in bs]), E


def rotation_generators(X, E=None):
    """Three infinitesimal rotations as vectors in tangent coordinates (3, 2N)."""
    E = tangent_frames(X) if E is None else E
    R = np.zeros((3, len(X), 2))
    for k in range(3):
        ek = np.zeros(3)
        ek[k] = 1
        for i, x in enumerate(X):
            R[k, i] = E[i] @ np.cross(ek, x)
    return R.reshape(3, -1)


# ---------------------------------------------------------------------------
# KKT / balance certificate
# ---------------------------------------------------------------------------
@dataclass
class BalanceReport:
    D: float
    active: List[Branch]
    lam: np.ndarray
    residual: float          # || sum lam_b grad_b ||
    rank: int                # rank of active gradients (<= 2N-3)
    dim_quot: int            # 2N-3
    min_lambda: float
    extra_incidences: int
    stationary: bool
    sharp: bool              # strict first-order (sharp) local minimum mod rotations
    G: np.ndarray = field(repr=False, default=None)

    def summary(self) -> str:
        return (f"D={self.D:.12f}  |A|={len(self.active)}  rank={self.rank}/{self.dim_quot}  "
                f"resid={self.residual:.2e}  min lambda={self.min_lambda:.3e}  "
                f"incidences={self.extra_incidences}  stationary={self.stationary}  sharp={self.sharp}")


def balance_report(X, tol=1e-9, act_tol=1e-7, res_tol=1e-8) -> BalanceReport:
    """Solve for the balance multipliers (min-norm point of the gradient hull).

    sharp := stationary, all multipliers > 0 *in some* balancing (we test the
    relative interior via the maximin LP), gradients span the (2N-3)-dim quotient
    and no extra incidences: then D(X') >= D(X) + c*dist(X', SO(3)X) nearby.
    """
    X = normalize(X)
    N = len(X)
    D, act = active_set(X, tol, act_tol)
    G, E = gradient_matrix(X, act)
    m = len(act)
    # min-norm point of conv{g_b}: min ||G^T lam||^2 on the simplex
    Q = G @ G.T
    res = minimize(lambda l: l @ Q @ l, np.ones(m) / m, jac=lambda l: 2 * Q @ l,
                   bounds=[(0, 1)] * m, constraints=[{"type": "eq", "fun": lambda l: l.sum() - 1}],
                   method="SLSQP", options={"ftol": 1e-16, "maxiter": 500})
    lam = res.x
    resid = float(np.linalg.norm(G.T @ lam))
    # maximin LP: maximise s.t. lam_b >= s, G^T lam = 0, sum = 1 -> strict positivity
    c = np.zeros(m + 1)
    c[-1] = -1
    A_eq = np.vstack([np.hstack([G.T, np.zeros((G.shape[1], 1))]),
                      np.hstack([np.ones((1, m)), np.zeros((1, 1))])])
    b_eq = np.concatenate([np.zeros(G.shape[1]), [1.0]])
    A_ub = np.hstack([-np.eye(m), np.ones((m, 1))])
    lp = linprog(c, A_ub=A_ub, b_ub=np.zeros(m), A_eq=A_eq, b_eq=b_eq,
                 bounds=[(0, None)] * m + [(None, None)], method="highs")
    if lp.status == 0:
        lam_star = lp.x[:m]
        min_l = float(lp.x[-1])
        stationary = True
        lam = lam_star
        resid = float(np.linalg.norm(G.T @ lam))
    else:
        min_l = 0.0
        stationary = resid < res_tol
    sv = np.linalg.svd(G, compute_uv=False)
    rank = int(np.sum(sv > 1e-7 * max(sv.max(), 1e-30)))
    extra = int(sum(1 for b in act if b.extra > 0 and b.kind != "const"))
    dimq = 2 * N - 3
    sharp = stationary and min_l > 1e-9 and rank == dimq and extra == 0
    return BalanceReport(D, act, lam, resid, rank, dimq, min_l, extra, stationary, sharp, G)


# ---------------------------------------------------------------------------
# smooth (frozen count) branch function and Lagrangian Hessian
# ---------------------------------------------------------------------------
def frozen_value(X, b: Branch):
    N = len(X)
    if b.kind == "const":
        return 1.0 / N
    t = cap_geometry(X, b.I)[0]
    V = (1 - t) / 2
    return b.n_cl / N - V if b.kind == "up" else V - b.n_op / N


def retract(X, E, v):
    Y = X + np.einsum("ik,ikj->ij", v.reshape(-1, 2), E)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def lagrangian_hessian(X, act, lam, h=1e-4):
    """Hessian (in tangent coordinates) of L(v) = sum lam_b f_b(retract(X,v)), FD."""
    X = normalize(X)
    E = tangent_frames(X)
    n = 2 * len(X)

    def L(v):
        Y = retract(X, E, v)
        return sum(l * frozen_value(Y, b) for l, b in zip(lam, act))

    H = np.zeros((n, n))
    I = np.eye(n) * h
    L0 = L(np.zeros(n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                H[i, i] = (L(I[i]) - 2 * L0 + L(-I[i])) / h**2
            else:
                H[i, j] = H[j, i] = (L(I[i] + I[j]) - L(I[i] - I[j]) - L(-I[i] + I[j]) + L(-I[i] - I[j])) / (4 * h * h)
    return H


# ---------------------------------------------------------------------------
# sequential LP descent with trust region (nonsmooth minimax)
# ---------------------------------------------------------------------------
def slp_descent(X, iters=400, delta0=0.2, margin=0.05, tol=1e-9, verbose=False):
    """Minimise D by sequential linear programming on the frozen-count envelope.

    Each step solves   min s   s.t.  f_b(X) + <g_b, v> <= s  (b near-active),  |v|_inf <= delta
    and accepts the step if the true D decreases; delta is adapted.  At a sharp
    (vertex) minimiser this converges quadratically.
    """
    X = normalize(X)
    delta = delta0
    D = discrepancy(X, tol)
    n = 2 * len(X)
    for it in range(iters):
        bs = branches(X, tol)
        near = [b for b in bs if b.value >= D - margin]
        G, E = gradient_matrix(X, near)
        fv = np.array([b.value for b in near])
        c = np.zeros(n + 1)
        c[-1] = 1
        A = np.hstack([G, -np.ones((len(near), 1))])
        res = linprog(c, A_ub=A, b_ub=-fv, bounds=[(-delta, delta)] * n + [(None, None)], method="highs")
        if res.status != 0:
            delta *= 0.5
            continue
        v = res.x[:n]
        pred = D - res.x[-1]
        Y = retract(X, E, v)
        Dn = discrepancy(Y, tol)
        if verbose:
            print(it, D, Dn, pred, delta)
        if Dn < D - 1e-15:
            ratio = (D - Dn) / max(pred, 1e-18)
            X, D = Y, Dn
            if ratio > 0.75:
                delta = min(delta * 2, 0.5)
            elif ratio < 0.25:
                delta *= 0.5
        else:
            delta *= 0.5
        if delta < 1e-13:
            break
    return X, D
