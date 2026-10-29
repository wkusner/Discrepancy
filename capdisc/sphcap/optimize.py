"""
sphcap.optimize
================

Gradient-based minimization of the spherical cap discrepancy over
configurations X in (S^{n-1})^N.

Why this is a nonsmooth max-of-smooth-functions problem
--------------------------------------------------------
D(X) = max_I max(Delta(w_I,t_I;X), Delta(-w_I,-t_I;X)) is, for fixed
combinatorial structure, a max over finitely many locally smooth
"branches" Delta_I(X). By Danskin's theorem the (Clarke) subdifferential
at a point where several branches tie for the max is the convex hull
of the individual branches' gradients -- and the note's own
characterization of the minimizer (several boundary caps balancing at
exactly the same discrepancy value) says this is *exactly* where we
expect the optimum to sit. A plain subgradient method therefore tends
to zig-zag right at the optimum; we address this the standard way,
with a log-sum-exp (softmax) smoothing of the finitely many active
branches, annealing the temperature toward the true max as iterations
proceed.

Gradient of a single branch Delta_I
------------------------------------
For the active index set I (|I| <= n points on the boundary), write

    Delta_I(X) = | emp_I(X) - mu_cap(t_I(X_I)) |.

Near a generic configuration, emp_I (an integer count divided by N) is
locally constant in X (it only changes when some point crosses the
cap boundary, a measure-zero event) -- so to first order the only
X-dependence worth differentiating is through t_I(X_I), and *only*
through the coordinates of the points actually in I (moving a point
that is not on the boundary does not change which cap is extremal, to
first order). This is once again Danskin's / the envelope theorem,
applied to the *inner* maximization defining (w_I, t_I) itself.

We compute d(mu_cap(t_I))/dX_i for i in I by a short central-difference
stencil restricted to the (at most 3*n) coordinates of the points in
I -- deliberately not a hand-derived analytic Jacobian of the closed
form, to keep this code short and easy to check against
:func:`sphcap.discrepancy` directly (see tests). All points outside I
receive exactly zero gradient contribution from this branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .discrepancy import _boundary_cap_for_subset, _candidate_subsets, discrepancy
from .utils import cap_measure, normalize, random_points


@dataclass
class OptimizeResult:
    X: np.ndarray
    value: float
    history: list = field(default_factory=list)
    n_iter: int = 0


def _all_branch_values(X: np.ndarray, eps: float, max_size: Optional[int]):
    """All (I, sign, Delta_I) triples -- used by the smoothed objective.

    Returns lists of index-sets, signs (+1/-1), and discrepancy values.
    O(N^n); fine for the modest N this optimizer targets per exact step.
    """
    N, n = X.shape
    Is, signs, vals = [], [], []
    for I in _candidate_subsets(N, n, max_size):
        sol = _boundary_cap_for_subset(X, I, eps)
        if sol is None:
            continue
        t_I, w_I = sol
        proj = X @ w_I
        for sign in (+1, -1):
            emp = np.count_nonzero(sign * proj >= sign * t_I - eps) / N
            cap = float(cap_measure(np.array(sign * t_I), n))
            vals.append(abs(emp - cap))
            Is.append(I)
            signs.append(sign)
    return Is, signs, np.array(vals)


def _branch_grad_fd(X: np.ndarray, I: tuple, sign: int, eps: float, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of Delta_{I,sign} w.r.t. the
    coordinates of the points in I only (see module docstring); all
    other rows of the returned (N,n) array are exactly zero.
    """
    N, n = X.shape
    grad = np.zeros_like(X)
    for i in I:
        for k in range(n):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, k] += h
            Xm[i, k] -= h
            Xp[i] = normalize(Xp[i][None, :])[0]
            Xm[i] = normalize(Xm[i][None, :])[0]

            def val(Y):
                solp = _boundary_cap_for_subset(Y, I, eps)
                if solp is None:
                    return None
                t_I, w_I = solp
                proj = Y @ w_I
                emp = np.count_nonzero(sign * proj >= sign * t_I - eps) / N
                cap = float(cap_measure(np.array(sign * t_I), Y.shape[1]))
                return abs(emp - cap)

            vp, vm = val(Xp), val(Xm)
            if vp is None or vm is None:
                continue
            grad[i, k] = (vp - vm) / (2 * h)
    return grad


def _softmax_gradient_step(X: np.ndarray, eps: float, max_size: Optional[int],
                            beta: float, top_k: int) -> tuple[np.ndarray, float]:
    Is, signs, vals = _all_branch_values(X, eps, max_size)
    if len(vals) == 0:
        return np.zeros_like(X), 0.0
    order = np.argsort(vals)[::-1][:top_k]
    top_vals = vals[order]
    w = np.exp(beta * (top_vals - top_vals.max()))
    w /= w.sum()

    grad = np.zeros_like(X)
    for wj, j in zip(w, order):
        g = _branch_grad_fd(X, Is[j], signs[j], eps)
        grad += wj * g
    return grad, float(vals.max())


def minimize_discrepancy(X0: np.ndarray, n_iter: int = 200, lr: float = 0.05,
                          beta0: float = 20.0, beta1: float = 400.0,
                          top_k: int = 6, eps: float = 1e-9,
                          max_size: Optional[int] = None,
                          verbose: bool = False,
                          callback: Optional[Callable[[int, np.ndarray, float], None]] = None
                          ) -> OptimizeResult:
    """Projected (Riemannian) gradient descent on a softmax-smoothed
    spherical cap discrepancy objective, annealing the softmax
    temperature beta from beta0 (smooth, easy to move) to beta1 (close
    to the true nonsmooth max) over the run.

    Returns the *best* configuration seen (by exact discrepancy), not
    just the final iterate, since the smoothed objective is a proxy.
    """
    X = normalize(np.array(X0, dtype=float))
    history = []
    best_X, best_val = X.copy(), discrepancy(X, method="exact", max_size=max_size).value

    for it in range(n_iter):
        beta = beta0 * (beta1 / beta0) ** (it / max(n_iter - 1, 1))
        grad, smooth_max = _softmax_gradient_step(X, eps, max_size, beta, top_k)
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            # Riemannian: project gradient tangent to the sphere at each point, step, retract.
            tangent = grad - X * np.sum(grad * X, axis=1, keepdims=True)
            X = normalize(X - lr * tangent)
        true_val = discrepancy(X, method="exact", max_size=max_size).value
        history.append(true_val)
        if true_val < best_val:
            best_val, best_X = true_val, X.copy()
        if callback:
            callback(it, X, true_val)
        if verbose and it % max(n_iter // 10, 1) == 0:
            print(f"iter {it:4d}  beta={beta:8.1f}  D={true_val:.6f}  best={best_val:.6f}")

    return OptimizeResult(best_X, best_val, history, n_iter)


def minimize_discrepancy_multistart(N: int, n_starts: int = 8, n: int = 3,
                                     seed: Optional[int] = None, **kwargs) -> OptimizeResult:
    """Random-restart wrapper: independent runs from i.i.d. uniform
    starts, keeping the overall best. Discrepancy optimization on the
    sphere is highly non-convex (finite symmetry groups + relabeling
    act on the landscape), so multi-start is the practical default.
    """
    rng = np.random.default_rng(seed)
    best: Optional[OptimizeResult] = None
    for s in range(n_starts):
        X0 = random_points(N, n=n, rng=rng)
        res = minimize_discrepancy(X0, **kwargs)
        if best is None or res.value < best.value:
            best = res
    return best
