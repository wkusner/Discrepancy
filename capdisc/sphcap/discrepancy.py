"""
sphcap.discrepancy
===================

Computing the spherical cap discrepancy

    D(X) = sup_{w in S^{n-1}, t in [-1,1]}  | mu_emp(w,t) - mu_cap(t) |,
    mu_emp(w,t) = (1/N) #{ i : <w, x_i> >= t }

for a configuration X = (x_1,...,x_N) in (S^{n-1})^N.

The key mathematical fact (Heitsch & Henrion, "An enumerative formula
for the spherical cap discrepancy", J. Comput. Appl. Math. 390 (2021);
and the cleaner "generic point sets" version in Unif. Distrib. Theory
20 (2025)) is that the supremum over the *continuum* of caps is
attained at one of finitely many explicit "boundary" caps: those whose
boundary hyperplane passes through some subset I of 1 to n points.
For a generic configuration (no n points linearly dependent), the
extremal cap through I is given in closed form:

    t_I = ( 1^T (X_I^T X_I)^{-1} 1 )^{-1/2}
    w_I = t_I  X_I (X_I^T X_I)^{-1} 1

and

    D(X) = max_{ I, 1<=|I|<=n } max( Delta(w_I, t_I), Delta(-w_I, -t_I) ).

This is exactly the rigorous version of the "pass the boundary through
2 or 3 points, and balance" ("cap capturing") idea in the accompanying
note. On S^2 (n=3), |I| in {1,2,3}, so the combinatorial cost is
O(N) + O(N^2) + O(N^3) subsets.

Four computation strategies are provided, trading exactness for speed:

  * "bruteforce"  : literal triple-nested loop over subsets, O(N^n).
                    Slow, but the most obviously-correct reference
                    implementation; good for unit tests on small N.
  * "exact"       : same O(N^n) subset family, but every subset's
                    (t_I, w_I, Delta) is evaluated with batched,
                    chunked NumPy linear algebra (optionally on a
                    GPU/multiprocessing backend, see sphcap.backend).
                    This is the recommended exact method.
  * "axis_sweep"  : a genuinely sub-cubic *lower bound*,
                        Delta_tilde(X) = max_i sup_t |mu_emp(x_i,t)-mu_cap(t)|,
                    i.e. we only look at caps centered at one of the N
                    data points, but for each such axis we find the
                    *exact* extremal t along that axis by sorting the
                    N-1 projections (O(N log N) per axis instead of a
                    naive O(N) scan repeated over O(N) threshold
                    candidates). Total cost O(N^2 log N): one power of
                    N cheaper than the full exact method, at the cost
                    of only being a lower bound in general (Heitsch &
                    Henrion observed it is numerically tight for
                    N gtrsim 500 on S^2 -- see their Fig. 3).
  * "monte_carlo" : statistical sampling. Draw M random directions w
                    and, for each, do the exact O(N log N) sorted
                    sweep over t (so the *t*-optimization is exact,
                    only the *direction* is randomized). Returns a
                    lower-bound estimate that increases monotonically
                    and (for large M) a rough confidence band from the
                    empirical distribution of per-direction maxima.

All methods accept an ``eps`` tolerance used (a) to decide which points
count as "on the boundary" of a candidate cap and (b) to guard against
near-singular subsets (nearly collinear / nearly antipodal points) in
the exact combinatorial method.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .utils import cap_measure
from .backend import Backend, get_default_backend


@dataclass
class DiscrepancyResult:
    value: float
    method: str
    witness_w: Optional[np.ndarray] = None
    witness_t: Optional[float] = None
    witness_I: Optional[tuple] = None
    is_lower_bound: bool = False
    extra: dict = field(default_factory=dict)

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        tag = " (lower bound)" if self.is_lower_bound else ""
        return f"DiscrepancyResult({self.value:.6f}{tag}, method={self.method!r})"


# ---------------------------------------------------------------------
# core local-discrepancy primitive
# ---------------------------------------------------------------------
def _local_discrepancy(proj: np.ndarray, t: float, n_points: int, n: int, eps: float) -> float:
    """Delta(w,t) given the precomputed projections <w, x_i> for all i."""
    emp = np.count_nonzero(proj >= t - eps) / n_points
    cap = float(cap_measure(np.array(t), n))
    return abs(emp - cap)


def _boundary_cap_for_subset(X: np.ndarray, I: tuple, eps: float):
    """Closed-form (t_I, w_I) for boundary index set I (Heitsch-Henrion).

    Returns None if the subset is (numerically) linearly dependent,
    i.e. not usable as a boundary-defining set for a generic point
    configuration (this is the ``eps``-tolerant genericity guard).
    """
    X_I = X[list(I), :].T  # n x |I|
    G = X_I.T @ X_I  # |I| x |I| Gram matrix
    # guard against near-singular Gram matrices (near-collinear /
    # near-antipodal points in I): condition number check.
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return None
    if np.linalg.cond(G) > 1.0 / max(eps, 1e-14):
        return None
    ones = np.ones(len(I))
    denom = ones @ Ginv @ ones
    if denom <= 0:
        return None
    t_I = denom ** -0.5
    w_I = t_I * (X_I @ Ginv @ ones)
    norm = np.linalg.norm(w_I)
    if norm < eps:
        return None
    w_I = w_I / norm
    return t_I, w_I


def _candidate_subsets(N: int, n: int, max_size: Optional[int] = None) -> Iterable[tuple]:
    kmax = max_size or n
    for k in range(1, kmax + 1):
        yield from itertools.combinations(range(N), k)


# ---------------------------------------------------------------------
# method: bruteforce
# ---------------------------------------------------------------------
def _discrepancy_bruteforce(X: np.ndarray, eps: float, max_size: Optional[int]) -> DiscrepancyResult:
    N, n = X.shape
    best = 0.0
    best_w = best_t = best_I = None
    for I in _candidate_subsets(N, n, max_size):
        sol = _boundary_cap_for_subset(X, I, eps)
        if sol is None:
            continue
        t_I, w_I = sol
        proj = X @ w_I
        for sign in (+1, -1):
            w, t = sign * w_I, sign * t_I
            d = _local_discrepancy(sign * proj, t, N, n, eps)
            if d > best:
                best, best_w, best_t, best_I = d, w, t, I
    return DiscrepancyResult(best, "bruteforce", best_w, best_t, best_I)


# ---------------------------------------------------------------------
# method: exact (vectorized / batched, chunked through a Backend)
# ---------------------------------------------------------------------
def _discrepancy_exact(X: np.ndarray, eps: float, max_size: Optional[int],
                        backend: Optional[Backend] = None,
                        chunk_size: int = 4096) -> DiscrepancyResult:
    N, n = X.shape
    backend = backend or get_default_backend(prefer_gpu=False)  # linalg-heavy, CPU vectorization wins here
    kmax = max_size or n

    best = 0.0
    best_w = best_t = best_I = None

    for k in range(1, kmax + 1):
        idx = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(N), k)),
                           dtype=np.int64)
        if idx.size == 0:
            continue
        idx = idx.reshape(-1, k)  # (M, k)
        M = idx.shape[0]

        for start in range(0, M, chunk_size):
            block = idx[start:start + chunk_size]  # (m, k)
            Xb = X[block]  # (m, k, n)
            G = np.einsum("mik,mjk->mij", Xb, Xb)  # (m,k,k) Gram matrices
            # batched solve; mask out (near-)singular ones
            ones = np.ones(k)
            try:
                Ginv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                # fall back to per-item pseudo-inverse for the whole block
                Ginv = np.linalg.pinv(G)
            denom = np.einsum("i,mij,j->m", ones, Ginv, ones)
            good = denom > 1e-12
            if not np.any(good):
                continue
            t_I = np.zeros(block.shape[0])
            t_I[good] = denom[good] ** -0.5
            w_dir = np.einsum("mkn,mk->mn", Xb[good], (Ginv[good] @ ones) * t_I[good, None])
            norms = np.linalg.norm(w_dir, axis=1)
            ok = norms > eps
            if not np.any(ok):
                continue
            w_I = w_dir[ok] / norms[ok, None]
            t_good = t_I[good][ok]
            block_good = block[good][ok]

            proj = w_I @ X.T  # (m', N)
            cap_pos = cap_measure(t_good, n)  # (m',)
            emp_pos = np.count_nonzero(proj >= (t_good[:, None] - eps), axis=1) / N
            d_pos = np.abs(emp_pos - cap_pos)

            cap_neg = cap_measure(-t_good, n)
            emp_neg = np.count_nonzero(-proj >= (-t_good[:, None] - eps), axis=1) / N
            d_neg = np.abs(emp_neg - cap_neg)

            local_best_pos = np.argmax(d_pos)
            local_best_neg = np.argmax(d_neg)
            if d_pos[local_best_pos] > best:
                best = float(d_pos[local_best_pos])
                best_w, best_t = w_I[local_best_pos], float(t_good[local_best_pos])
                best_I = tuple(block_good[local_best_pos])
            if d_neg[local_best_neg] > best:
                best = float(d_neg[local_best_neg])
                best_w = -w_I[local_best_neg]
                best_t = -float(t_good[local_best_neg])
                best_I = tuple(block_good[local_best_neg])

    return DiscrepancyResult(best, "exact", best_w, best_t, best_I)


# ---------------------------------------------------------------------
# method: axis_sweep  (sorted, sub-cubic *lower bound*)
# ---------------------------------------------------------------------
def _discrepancy_axis_sweep(X: np.ndarray, eps: float) -> DiscrepancyResult:
    """Delta_tilde(X) = max_i sup_t |mu_emp(x_i,t) - mu_cap(t)|.

    For each axis w = x_i, sort the N projections <x_i, x_j> once
    (O(N log N)) and sweep the N distinct threshold candidates in a
    single O(N) pass, so each axis costs O(N log N) and the whole
    method costs O(N^2 log N) -- one power of N cheaper than the full
    exact O(N^3) enumeration on S^2, at the price of only certifying a
    lower bound (only N of the infinitely many directions are tried).
    """
    N, n = X.shape
    G = X @ X.T  # (N,N), row i = projections of all points onto x_i
    best = 0.0
    best_w = best_t = None
    for i in range(N):
        proj = np.sort(G[i])[::-1]  # descending
        # candidate thresholds: just below/at each distinct projection
        k = np.arange(1, N + 1)  # after sorting descending, k points have proj >= proj[k-1]
        emp = k / N
        cap = cap_measure(proj, n)
        d = np.abs(emp - cap)
        j = np.argmax(d)
        if d[j] > best:
            best = float(d[j])
            best_w = X[i].copy()
            best_t = float(proj[j])
    return DiscrepancyResult(best, "axis_sweep", best_w, best_t, is_lower_bound=True)


# ---------------------------------------------------------------------
# method: monte_carlo (statistical sampling)
# ---------------------------------------------------------------------
def _discrepancy_monte_carlo(X: np.ndarray, n_samples: int, eps: float,
                              rng: Optional[np.random.Generator] = None) -> DiscrepancyResult:
    """Random directions, exact sorted sweep over t per direction.

    Returns a lower-bound estimate together with basic sampling
    diagnostics (mean / std of the per-direction maxima, and the
    running max as a function of sample count) so the caller can judge
    convergence.
    """
    rng = rng or np.random.default_rng()
    N, n = X.shape
    W = rng.normal(size=(n_samples, n))
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    proj_all = W @ X.T  # (n_samples, N)
    per_dir_max = np.empty(n_samples)
    for s in range(n_samples):
        proj = np.sort(proj_all[s])[::-1]
        k = np.arange(1, N + 1)
        d = np.abs(k / N - cap_measure(proj, n))
        per_dir_max[s] = d.max()

    running_max = np.maximum.accumulate(per_dir_max)
    best_idx = int(np.argmax(per_dir_max))
    return DiscrepancyResult(
        float(per_dir_max.max()), "monte_carlo",
        witness_w=W[best_idx], is_lower_bound=True,
        extra={
            "n_samples": n_samples,
            "mean": float(per_dir_max.mean()),
            "std": float(per_dir_max.std()),
            "running_max": running_max,
        },
    )


# ---------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------
def discrepancy(X: np.ndarray, method: str = "exact", eps: float = 1e-9,
                 max_size: Optional[int] = None, n_samples: int = 20000,
                 backend: Optional[Backend] = None,
                 rng: Optional[np.random.Generator] = None) -> DiscrepancyResult:
    """Spherical cap discrepancy of the configuration X (rows = points).

    Parameters
    ----------
    X : (N, n) array
        N points on S^{n-1} subset R^n. Not required to be exactly
        normalized; they are renormalized defensively.
    method : {"exact", "bruteforce", "axis_sweep", "monte_carlo"}
    eps : float
        Numerical tolerance for boundary ties and for the genericity
        guard on near-singular subsets (see module docstring).
    max_size : int, optional
        Cap the boundary-subset size searched (default: n, the
        mathematically necessary/sufficient bound for generic points).
        Smaller values give a (cheaper) lower bound.
    n_samples : int
        Number of random directions for method="monte_carlo".
    backend : sphcap.backend.Backend, optional
        Only consulted by method="exact"; defaults to an
        auto-detected backend (see sphcap.backend.get_default_backend).
    """
    X = np.asarray(X, dtype=float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    N, n = X.shape
    if N < 1:
        return DiscrepancyResult(0.0, method)

    if method == "bruteforce":
        return _discrepancy_bruteforce(X, eps, max_size)
    if method == "exact":
        return _discrepancy_exact(X, eps, max_size, backend=backend)
    if method == "axis_sweep":
        return _discrepancy_axis_sweep(X, eps)
    if method == "monte_carlo":
        return _discrepancy_monte_carlo(X, n_samples, eps, rng=rng)
    raise ValueError(f"unknown method {method!r}")
