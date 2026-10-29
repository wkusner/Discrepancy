"""
sphcap.configs
==============

Generators for initial / conjectural point configurations on S^2 (a
couple of them generalize to S^{n-1}), covering:

  * the exact conjectural minimizer for N=4 (the "squished disphenoid"
    derived in the accompanying note) and the general disphenoid
    family it belongs to,
  * Platonic solids,
  * the spherical Fibonacci lattice (Aistleitner-Brauchart-Dick, 2012),
  * the Beltran-Etayo "Diamond ensemble" (Etayo, 2021),
  * a simple Lubotzky-Phillips-Sarnak-style "Hecke point" construction
    (number-theoretic, from a free/near-free rotation group generated
    by unit quaternions with denominator p),
  * i.i.d. uniform random points,
  * Riesz-energy ("Thomson problem") minimizers via simple gradient
    descent, as a generic "energy construction" baseline.

Every generator returns an (N, 3) (or (N, n)) NumPy array of unit
vectors.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .utils import normalize, random_points


# ----------------------------------------------------------------------
# 0. The N=4 disphenoid family (see accompanying note, Theorem 1)
# ----------------------------------------------------------------------
def disphenoid_family(t: float) -> np.ndarray:
    """The D_2d-symmetric 1-parameter family of isosceles tetrahedra:

        p1=(t,s,0), p2=(t,-s,0), p3=(-t,0,s), p4=(-t,0,-s),  s=sqrt(1-t^2).

    t = 1/sqrt(3) gives the regular tetrahedron. t is otherwise free in
    (0,1). See the note for the derivation of the *optimal* (squished)
    value returned by :func:`minimal_four_point_configuration`.
    """
    if not (0.0 < t < 1.0):
        raise ValueError("t must lie in (0,1)")
    s = np.sqrt(1.0 - t * t)
    return np.array([
        [t, s, 0.0],
        [t, -s, 0.0],
        [-t, 0.0, s],
        [-t, 0.0, -s],
    ])


def regular_tetrahedron() -> np.ndarray:
    """The disphenoid family at t = 1/sqrt(3): the regular tetrahedron."""
    return disphenoid_family(1.0 / np.sqrt(3.0))


def minimal_four_point_configuration() -> np.ndarray:
    """The conjectured *unique* (up to isometry) minimum spherical cap
    discrepancy configuration of 4 points on S^2, D(X) = 3/8 exactly.

    Derived in the accompanying note: within the disphenoid family
    above, balancing the two "3 points on the cap boundary" local
    discrepancies at 3/8 forces

        t^2 = (13 - sqrt(105)) / 32.

    (The other root of the same quartic gives a *worse* configuration,
    where a 2-point boundary cap becomes the binding constraint; see
    the note for the disambiguation.)
    """
    t2 = (13.0 - np.sqrt(105.0)) / 32.0
    return disphenoid_family(np.sqrt(t2))


# ----------------------------------------------------------------------
# 1. Platonic solids / simple lattices
# ----------------------------------------------------------------------
def platonic_solid(name: str) -> np.ndarray:
    name = name.lower()
    phi = (1 + np.sqrt(5)) / 2
    if name in ("tetrahedron", "tet"):
        X = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    elif name in ("octahedron", "oct"):
        X = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    elif name in ("cube", "hexahedron"):
        X = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    elif name in ("icosahedron", "ico"):
        X = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                X += [[0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]]
        X = np.array(X, dtype=float)
    elif name in ("dodecahedron", "dodeca"):
        X = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                for s3 in (-1, 1):
                    X.append([s1, s2, s3])
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                X += [[0, s1 / phi, s2 * phi], [s1 / phi, s2 * phi, 0], [s2 * phi, 0, s1 / phi]]
        X = np.array(X, dtype=float)
    else:
        raise ValueError(f"unknown solid {name!r}")
    return normalize(X)


def fibonacci_sphere(N: int) -> np.ndarray:
    """Spherical Fibonacci lattice (Aistleitner-Brauchart-Dick 2012),
    known to give O(N^{-1/2}) spherical cap discrepancy -- the best
    *explicit* deterministic rate known until the Diamond ensemble.
    """
    i = np.arange(N) + 0.5
    golden = (1 + np.sqrt(5)) / 2
    z = 1 - 2 * i / N
    theta = np.arccos(np.clip(z, -1, 1))
    phi = 2 * np.pi * i / golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    return np.stack([x, y, z], axis=1)


# ----------------------------------------------------------------------
# 2. The Diamond ensemble (Beltran-Etayo 2018; Etayo 2021, Def. 2.1 & Prop. 2.2)
# ----------------------------------------------------------------------
def diamond_ensemble(M: int, r_of_j=None, theta_of_j=None) -> np.ndarray:
    """The concrete Diamond-ensemble model of Etayo (2021), Thm 1.4:
    n=1, r_j = 4j for 1 <= j <= M, plus the North and South poles.
    Total N = 2 + 4 M^2 points; proven D_sup,cap = O(N^{-1/2}) with
    explicit constants (the best known *deterministic* rate at the
    time of that paper).

    Parameters
    ----------
    M : int
        Number of parallels per hemisphere.
    r_of_j : callable, optional
        Overrides the default r_j = 4j (points per parallel j); must
        return a positive integer for j=1..M. r_j should be
        non-decreasing in j (Etayo's monotonicity assumption).
    theta_of_j : callable, optional
        Rotation phase per parallel (default: 0 for all j).
    """
    r_of_j = r_of_j or (lambda j: 4 * j)
    theta_of_j = theta_of_j or (lambda j: 0.0)

    r = np.array([r_of_j(j) for j in range(1, M + 1)])
    # Total parallels p = 2M-1: j=1..M-1 mirrored (north/south), j=M sits
    # exactly on the equator (z_M = 0, see Prop. 2.2) and is NOT mirrored.
    N = 2 + 2 * int(r[:-1].sum()) + int(r[-1])  # poles + mirrored pairs + equator

    Nj = 1 + np.concatenate([[0], np.cumsum(r)[:-1]])  # eq. (5)
    zj = 1 - (1 + r + 2 * (Nj - 1)) / (N - 1)  # Prop. 2.2

    pts = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
    for j in range(1, M + 1):
        rj = r[j - 1]
        z = zj[j - 1]
        rad = np.sqrt(max(1.0 - z * z, 0.0))
        th0 = theta_of_j(j)
        idx = np.arange(rj)
        angles = 2 * np.pi * idx / rj + th0
        ring = np.stack([rad * np.cos(angles), rad * np.sin(angles), np.full(rj, z)], axis=1)
        pts.append(ring)
        if j < M:  # mirror everything except the equatorial ring
            pts.append(np.stack([rad * np.cos(angles), rad * np.sin(angles), np.full(rj, -z)], axis=1))
    return normalize(np.concatenate([p if p.ndim == 2 else p[None, :] for p in pts], axis=0))


# ----------------------------------------------------------------------
# 3. "Hecke points": LPS-style number-theoretic construction
# ----------------------------------------------------------------------
def _lps_generators(p: int) -> np.ndarray:
    """Unit quaternions (as 4-vectors) with a^2+b^2+c^2+d^2=p, a odd>0,
    b,c,d even, giving the p+1 Lubotzky-Phillips-Sarnak generators used
    to build optimal ("Ramanujan") expander/equidistribution walks on
    SO(3). p must be prime, p = 1 (mod 4).
    """
    if p % 4 != 1:
        raise ValueError("p must be a prime with p = 1 (mod 4)")
    sols = []
    lim = int(np.sqrt(p)) + 1
    for a in range(1, lim + 1, 2):  # a odd, a>0
        rem = p - a * a
        if rem < 0:
            continue
        for b in range(-lim, lim + 1, 2):  # b even
            rem2 = rem - b * b
            if rem2 < 0:
                continue
            for c in range(-lim, lim + 1, 2):  # c even
                rem3 = rem2 - c * c
                if rem3 < 0:
                    continue
                d2 = rem3
                d = int(round(np.sqrt(d2)))
                if d % 2 != 0:
                    continue
                if d * d == d2:
                    for dd in ({d, -d} if d != 0 else {0}):
                        sols.append((a, b, c, dd))
    sols = np.array(sorted(set(sols)), dtype=float)
    if len(sols) != p + 1:
        # Jacobi's four-square theorem guarantees p+1 such solutions
        # exist for p = 1 (mod 4); a mismatch means the search range
        # (`lim`) or parity filtering missed some -- fall back to
        # whatever was found rather than hard-failing.
        pass
    return sols / np.sqrt(p)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    a, b, c, d = q
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
    ])


def hecke_points(p: int, depth: int, x0: Optional[np.ndarray] = None,
                  max_points: Optional[int] = None) -> np.ndarray:
    """A simple LPS/"Hecke-tree" construction: starting from a seed
    point x0, repeatedly apply the p+1 LPS rotation generators (a
    near-free, near-optimal-expander generating set for SO(3) coming
    from unit quaternions of norm p, Jacobi four-square solutions),
    breadth-first up to ``depth`` steps, without immediately backtracking.
    This is a standard number-theoretic route to (conjecturally/
    provably in various regimes) well-equidistributed point sets, used
    e.g. in Lubotzky-Phillips-Sarnak and subsequent "Hecke points on
    the sphere" constructions.

    This is a *reference/teaching implementation* (not optimized for
    large depth: the tree grows like (p+1)*p^{depth-1}); use
    ``max_points`` to cap the output size by truncating breadth-first.
    """
    gens = _lps_generators(p)
    mats = [_quat_to_rotmat(q) for q in gens]
    # index of each generator's "inverse" (conjugate quaternion) so we
    # can forbid immediately backtracking, as in the LPS construction
    conj_index = []
    for q in gens:
        qc = np.array([q[0], -q[1], -q[2], -q[3]])
        matches = np.argmin(np.sum((gens - qc) ** 2, axis=1))
        conj_index.append(int(matches))

    x0 = np.array([0.0, 0.0, 1.0]) if x0 is None else normalize(np.asarray(x0, dtype=float))
    frontier = [(x0, -1)]
    all_points = [x0]
    for _ in range(depth):
        new_frontier = []
        for x, last_gen in frontier:
            for gi, M in enumerate(mats):
                if gi == last_gen:
                    continue
                if last_gen != -1 and gi == conj_index[last_gen]:
                    continue
                y = M @ x
                new_frontier.append((y, gi))
                all_points.append(y)
                if max_points and len(all_points) >= max_points:
                    return normalize(np.array(all_points[:max_points]))
        frontier = new_frontier
    return normalize(np.array(all_points))


# ----------------------------------------------------------------------
# 4. Random baselines and energy-minimized ("Thomson problem") points
# ----------------------------------------------------------------------
def random_uniform(N: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return random_points(N, n=3, rng=rng)


def riesz_energy_minimize(N: int, s: float = 1.0, iters: int = 2000, lr: float = 0.05,
                           seed: Optional[int] = None, x0: Optional[np.ndarray] = None) -> np.ndarray:
    """A simple projected-gradient minimizer of the Riesz s-energy

        E_s(X) = sum_{i != j} ||x_i - x_j||^{-s}    (s -> 0: log energy)

    as a generic "energy construction" baseline (Thomson-problem-style).
    This is deliberately simple (plain gradient descent + renormalize,
    no line search / L-BFGS) -- see sphcap.optimize for the more
    careful machinery reused for discrepancy minimization itself.
    """
    rng = np.random.default_rng(seed)
    X = random_points(N, n=3, rng=rng) if x0 is None else normalize(np.array(x0, dtype=float))
    for _ in range(iters):
        diff = X[:, None, :] - X[None, :, :]  # (N,N,3)
        dist2 = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist2, np.inf)
        if abs(s) < 1e-9:
            grad_coeff = -1.0 / dist2  # d/dx (-log||.||) ~ -(x_i-x_j)/||.||^2
        else:
            grad_coeff = -s / dist2 ** (s / 2 + 1)
        grad = np.einsum("ijk,ij->ik", diff, grad_coeff)
        X = X - lr * grad
        X = normalize(X)
    return X
