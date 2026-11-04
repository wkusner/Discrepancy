"""A robust floating-point UPPER bound for the cap discrepancy, for testing lower bounds on degenerate cells.

Not part of the proof and not a proof of anything: a test oracle.  `sphcap.discrepancy` decides cap membership in
plain floating point and can underestimate D when points nearly coincide (e.g. 0.6332 for an N = 6 configuration
with four points within 1e-7, where D >= 4/6 - 1e-14; audit 2026-11-04, finding 9), which makes a correct lower
bound look wrong.  This oracle never underestimates (up to the conditioning model below):

  D = max over the capture caps of max(closed count/N - sigma, sigma - open count/N), together with 1/N, where the
  capture caps are the pair caps (axis +-(a + b)) and the triple caps (axis +-(b - a) x (c - a)), both orientations.

For each cap the defining points are counted in the closed cap and not in the open one, and every other point
within tau of the boundary is counted in the closed cap and not in the open one (inclusive counting).  tau bounds
the error of the computed axis direction: for a pair, w = a + b has absolute error ~eps, so the direction error is
~eps/|w|; for a triple, n = u x v (u = b - a, v = c - a, each with absolute error ~eps) has error ~eps(|u| + |v|),
so the direction error is ~eps(|u| + |v|)/|n|; tau is 64 times that, plus 1e-12.  So each term is at least the true value of its
cap, and the maximum is at least D.  On degenerate configurations the oracle may be far above D (it is then a weak
test, never a false alarm).
"""
import itertools
import numpy as np

EPS_ANGLE = 1e-12          # absolute slack on every angle
SAFE = 64 * np.finfo(float).eps


def _angles(w, X):
    c = np.cross(w, X)
    return np.arctan2(np.linalg.norm(c, axis=1), X @ w)


def discrepancy_upper(X):
    X = np.asarray(X, dtype=float)
    X = X / np.linalg.norm(X, axis=1)[:, None]
    N = len(X); best = 1.0 / N
    caps = []
    for i, j in itertools.combinations(range(N), 2):
        caps.append(((i, j), X[i] + X[j], 1.0))
    for i, j, k in itertools.combinations(range(N), 3):
        u, v = X[j] - X[i], X[k] - X[i]
        caps.append(((i, j, k), np.cross(u, v), np.linalg.norm(u) + np.linalg.norm(v)))
    for I, w, scale in caps:
        s = np.linalg.norm(w)
        if not s > 1e-300: continue                    # the axis is not defined (antipodal pair, coincident points)
        tau = EPS_ANGLE + SAFE * scale / s
        for sg in (1.0, -1.0):
            a = sg * w / s
            d = _angles(a, X)
            r_lo, r_hi = min(d[list(I)]), max(d[list(I)])  # the circle through the defining points (computed spread)
            other = np.ones(N, bool); other[list(I)] = False
            closed = len(I) + int(np.sum(other & (d <= r_hi + tau)))
            open_ = int(np.sum(other & (d < r_lo - tau)))
            s_lo, s_hi = (1 - np.cos(max(r_lo - tau, 0.0))) / 2, (1 - np.cos(min(r_hi + tau, np.pi))) / 2
            best = max(best, closed / N - s_lo + EPS_ANGLE, s_hi - open_ / N + EPS_ANGLE)
    return min(best, 1.0)


def points_float(q, N):
    """the capcert gauge: x0 = north pole, x1 = (sin q0, 0, cos q0), x_i = (theta, phi) = (q[2i-3], q[2i-2])"""
    X = [[0.0, 0.0, 1.0], [np.sin(q[0]), 0.0, np.cos(q[0])]]
    for i in range(2, N):
        th, ph = q[2 * i - 3], q[2 * i - 2]
        X.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return np.array(X)
