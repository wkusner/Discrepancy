"""
sphcap.utils
============

Low-level geometric utilities shared across the package:

* the normalized surface-measure of a spherical cap on S^{n-1},
* random rotations / uniform sampling on the sphere,
* small numerical-linear-algebra helpers used by the discrepancy
  formulas (Heitsch & Henrion, 2020 / 2025).

Everything here is pure NumPy + SciPy so it works with no optional
dependencies installed.
"""
from __future__ import annotations

import numpy as np
from scipy import special


def cap_measure(t: np.ndarray, n: int) -> np.ndarray:
    """Normalized area of the spherical cap {x in S^{n-1} : x_1 >= t}.

    Parameters
    ----------
    t : array_like
        Threshold(s) in [-1, 1].
    n : int
        Ambient dimension (points live in R^n, on S^{n-1}). n=3 is the
        ordinary 2-sphere.

    Returns
    -------
    Same shape as ``t``: mu_cap(t) in [0, 1], the fraction of the total
    surface area of S^{n-1} contained in the cap {x_1 >= t}.

    Notes
    -----
    Closed form via the regularized incomplete beta function:

        mu_cap(t) = 0.5 * I_{1-t^2}((n-1)/2, 1/2)          for t >= 0
                  = 1 - 0.5 * I_{1-t^2}((n-1)/2, 1/2)       for t <  0

    For n=3 this reduces to the elementary (1-t)/2 used throughout the
    hand computations in the accompanying note; we keep the general
    beta-function form so the same code path works for S^{d-1}, d>=2.
    """
    t = np.asarray(t, dtype=float)
    a = (n - 1) / 2.0
    x = np.clip(1.0 - t * t, 0.0, 1.0)
    ibeta = special.betainc(a, 0.5, x)
    pos = 0.5 * ibeta
    neg = 1.0 - 0.5 * ibeta
    return np.where(t >= 0, pos, neg)


def random_points(N: int, n: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
    """N i.i.d. points drawn uniformly from S^{n-1} (ambient dimension n)."""
    rng = rng or np.random.default_rng()
    x = rng.normal(size=(N, n))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def random_rotation(n: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
    """A Haar-random n x n rotation matrix (via QR of a Gaussian matrix)."""
    rng = rng or np.random.default_rng()
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def normalize(x: np.ndarray) -> np.ndarray:
    """Project rows of x back onto the unit sphere."""
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def pairwise_inner(X: np.ndarray) -> np.ndarray:
    """N x N Gram matrix <x_i, x_j> for points on the sphere."""
    return X @ X.T


def great_circle_distance(X: np.ndarray) -> np.ndarray:
    """N x N matrix of geodesic distances arccos(<x_i,x_j>) (clipped)."""
    G = np.clip(pairwise_inner(X), -1.0, 1.0)
    return np.arccos(G)
