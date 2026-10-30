"""
sphcap.bnb
==========

Branch-and-bound over the gauge- and symmetry-reduced configuration space,
proving lower bounds  D(X) >= T  cell by cell (pass 1: floating point screening).

Coordinates (2N-3 of them):
    x_0 = north pole,  x_1 = (sin th_1, 0, cos th_1),
    x_i = (sin th_i cos ph_i, sin th_i sin ph_i, cos th_i)      (i >= 2).

Fundamental domain for O(3) x S_N (closed; every configuration has a representative):
    (x_0, x_1) is a closest pair,                     th_1 <= d(x_i, x_j) for all pairs,
    the others are ordered by distance to x_0,        th_2 <= th_3 <= ... ,
    reflection y -> -y                                ph_2 in [0, pi].

Cell lower bound (rigorous given exact arithmetic for d_i and eps_i).
For a cell with centre configuration c and per-point motion bounds eps_i
(every Y in the cell has d(y_i, c_i) <= eps_i), and ANY axis w:
    up  : the closed cap of angular radius u_(k) (k-th smallest of d(c_i,w)+eps_i)
          contains >= k points of Y          ->  D(Y) >= k/N - (1 - cos u_(k))/2
    low : the open cap of radius l_(k) (k-th smallest of max(d(c_i,w)-eps_i, 0))
          contains <= k-1 points of Y        ->  D(Y) >= (1 - cos l_(k))/2 - (k-1)/N
and D(Y) >= 1/N.  The axes used are the centre's 2- and 3-point cap axes (+-),
so at eps = 0 the bound equals D(c) exactly (Heitsch--Henrion).

Motion bound: moving along a latitude then a meridian,
    eps_i <= h_th,i + sin(th_c,i) * h_ph,i      (h = half-widths).
"""
from __future__ import annotations

import itertools
import time
import numpy as np


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def domain(N):
    """Initial box (centre, half-width) in q-space for the fundamental domain."""
    lo = [0.0]
    hi = [np.pi]
    for i in range(2, N):
        lo += [0.0, 0.0 if i == 2 else -np.pi]
        hi += [np.pi, np.pi]
    lo, hi = np.array(lo), np.array(hi)
    return (lo + hi) / 2, (hi - lo) / 2


def points(Q, N):
    """Q (B, 2N-3) -> X (B, N, 3)."""
    B = Q.shape[0]
    X = np.zeros((B, N, 3))
    X[:, 0, 2] = 1.0
    X[:, 1, 0] = np.sin(Q[:, 0])
    X[:, 1, 2] = np.cos(Q[:, 0])
    for i in range(2, N):
        th, ph = Q[:, 2 * i - 3], Q[:, 2 * i - 2]
        s = np.sin(th)
        X[:, i, 0] = s * np.cos(ph)
        X[:, i, 1] = s * np.sin(ph)
        X[:, i, 2] = np.cos(th)
    return X


def motion(Q, H, N):
    """Per-point motion bound eps (B, N)."""
    B = Q.shape[0]
    E = np.zeros((B, N))
    E[:, 1] = H[:, 0]
    for i in range(2, N):
        E[:, i] = H[:, 2 * i - 3] + np.abs(np.sin(Q[:, 2 * i - 3])) * H[:, 2 * i - 2]
    return np.minimum(E, np.pi)


class Axes:
    def __init__(self, N):
        self.N = N
        self.T = np.array(list(itertools.combinations(range(N), 3)))
        self.P = np.array(list(itertools.combinations(range(N), 2)))

    def __call__(self, X):
        a, b, c = X[:, self.T[:, 0]], X[:, self.T[:, 1]], X[:, self.T[:, 2]]
        n = np.cross(b - a, c - a)
        s = X[:, self.P[:, 0]] + X[:, self.P[:, 1]]
        A = np.concatenate([n, s], axis=1)
        nr = np.linalg.norm(A, axis=2, keepdims=True)
        bad = nr[..., 0] < 1e-12
        A = A / np.where(nr < 1e-12, 1.0, nr)
        A[bad] = np.array([0.0, 0.0, 1.0])
        return np.concatenate([A, -A], axis=1)          # (B, K, 3)


def lower_bound(X, E, axes):
    """Robust lower bound for D over each cell; (B,)."""
    N = X.shape[1]
    A = axes(X)
    cosd = np.einsum("bkd,bnd->bkn", A, X)
    d = np.arccos(np.clip(cosd, -1.0, 1.0))
    u = np.sort(np.minimum(d + E[:, None, :], np.pi), axis=2)
    l = np.sort(np.maximum(d - E[:, None, :], 0.0), axis=2)
    k = np.arange(1, N + 1)
    up = (k / N - (1 - np.cos(u)) / 2).max(axis=(1, 2))
    low = ((1 - np.cos(l)) / 2 - (k - 1) / N).max(axis=(1, 2))
    return np.maximum(np.maximum(up, low), 1.0 / N)


def discrepancy_q(Q, N, axes=None):
    """Exact D at the gauge coordinates (eps = 0 bound)."""
    axes = axes or Axes(N)
    X = points(np.atleast_2d(Q), N)
    return lower_bound(X, np.zeros(X.shape[:2]), axes)


# ---------------------------------------------------------------------------
# symmetry pruning
# ---------------------------------------------------------------------------
def outside_domain(Q, H, X, E, N):
    """True for cells containing no point of the fundamental domain."""
    out = np.zeros(Q.shape[0], bool)
    # theta ordering of x_2..x_{N-1}
    for i in range(2, N - 1):
        ti, tj = 2 * i - 3, 2 * (i + 1) - 3
        out |= (Q[:, ti] - H[:, ti]) > (Q[:, tj] + H[:, tj])
    # closest pair: some other pair is closer than th_1 everywhere in the cell
    th1_lo = Q[:, 0] - H[:, 0]
    for i, j in itertools.combinations(range(N), 2):
        if (i, j) == (0, 1):
            continue
        dij = np.arccos(np.clip(np.einsum("bd,bd->b", X[:, i], X[:, j]), -1, 1))
        out |= dij + E[:, i] + E[:, j] < th1_lo
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def split(Q, H, N):
    """Bisect each cell along the coordinate contributing most to the motion bound."""
    W = H.copy()
    for i in range(2, N):
        W[:, 2 * i - 2] = H[:, 2 * i - 2] * np.abs(np.sin(Q[:, 2 * i - 3])) + 1e-3 * H[:, 2 * i - 2]
    j = np.argmax(W, axis=1)
    r = np.arange(Q.shape[0])
    H2 = H.copy()
    H2[r, j] /= 2
    Qa, Qb = Q.copy(), Q.copy()
    Qa[r, j] -= H2[r, j]
    Qb[r, j] += H2[r, j]
    return np.concatenate([Qa, Qb]), np.concatenate([H2, H2])


def run(N, T, min_eps=1e-4, batch=200_000, max_cells=None, verbose=True, log_every=10.0):
    """Certify D >= T on the fundamental domain.

    Returns dict(stats, survivors_Q, survivors_H, survivors_lb): survivors are cells
    whose motion bound fell below min_eps without certification (the region D < T + O(min_eps)).
    """
    axes = Axes(N)
    q0, h0 = domain(N)
    stack_Q, stack_H = [q0[None]], [h0[None]]
    nQ = 1
    stats = dict(processed=0, bounded=0, pruned_symmetry=0, survivors=0, max_depth_eps=np.inf)
    surv_Q, surv_H, surv_lb = [], [], []
    t0 = tlast = time.time()
    while stack_Q:
        Q = np.concatenate(stack_Q)
        H = np.concatenate(stack_H)
        stack_Q, stack_H = [], []
        # process in batches; LIFO keeps memory bounded
        pending_Q, pending_H = [Q], [H]
        while pending_Q:
            Q, H = pending_Q.pop(), pending_H.pop()
            if Q.shape[0] > batch:
                pending_Q += [Q[batch:], Q[:batch]]
                pending_H += [H[batch:], H[:batch]]
                continue
            X = points(Q, N)
            E = motion(Q, H, N)
            stats["processed"] += Q.shape[0]
            out = outside_domain(Q, H, X, E, N)
            stats["pruned_symmetry"] += int(out.sum())
            keep = ~out
            Q, H, X, E = Q[keep], H[keep], X[keep], E[keep]
            lb = lower_bound(X, E, axes)
            ok = lb >= T
            stats["bounded"] += int(ok.sum())
            Q, H, E, lb = Q[~ok], H[~ok], E[~ok], lb[~ok]
            emax = E.max(axis=1) if E.size else np.zeros(0)
            small = emax < min_eps
            if small.any():
                surv_Q.append(Q[small]); surv_H.append(H[small]); surv_lb.append(lb[small])
                stats["survivors"] += int(small.sum())
            Q, H = Q[~small], H[~small]
            if Q.shape[0]:
                Qs, Hs = split(Q, H, N)
                pending_Q.append(Qs); pending_H.append(Hs)
            if max_cells and stats["processed"] > max_cells:
                stats["aborted"] = True
                stats["time"] = time.time() - t0
                return dict(stats=stats, open_cells=sum(p.shape[0] for p in pending_Q),
                            survivors_Q=_cat(surv_Q, 2 * N - 3), survivors_H=_cat(surv_H, 2 * N - 3),
                            survivors_lb=_cat(surv_lb, None))
            if verbose and time.time() - tlast > log_every:
                tlast = time.time()
                print(f"  [{tlast - t0:7.1f}s] processed {stats['processed']:,} bounded {stats['bounded']:,} "
                      f"sym {stats['pruned_symmetry']:,} surv {stats['survivors']:,} "
                      f"open {sum(p.shape[0] for p in pending_Q):,}", flush=True)
    stats["time"] = time.time() - t0
    return dict(stats=stats, survivors_Q=_cat(surv_Q, 2 * N - 3), survivors_H=_cat(surv_H, 2 * N - 3),
                survivors_lb=_cat(surv_lb, None))


def _cat(lst, w):
    if lst:
        return np.concatenate(lst)
    return np.zeros((0, w)) if w else np.zeros(0)
