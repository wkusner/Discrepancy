"""Spherical t-designs on S^2: moment residuals, symmetric-stratum search, high-precision refinement.

A set X of N unit vectors is a spherical t-design iff
    (1/N) sum_i x_i^a y_i^b z_i^c = mean over the sphere of x^a y^b z^c
for all monomials of degree <= t.  Since x^2+y^2+z^2 = 1 on the sphere it is enough to take c in {0, 1}:
this gives (t+1)^2 - 1 equations (the constant term is trivial).  The sphere means are exact rationals
(a-1)!!(b-1)!!(c-1)!!/(a+b+c+1)!! when a, b, c are all even and 0 otherwise.

Everything here is floating point (numpy/scipy) except `refine_mp`, which works at the precision you ask for
with mpmath.  Nothing in this file proves anything: a small residual is a numerical (level 0) or
high-precision (level 1) observation in the atlas sense.

Search strategy.  A generic search over all N points (group C1) works when the number of unknowns 2N-3 is at
least the number of equations (t+1)^2-1.  Many of the best designs only exist with symmetry, so the main
search is over *strata*: a rotation group G, a set of special orbits (points on rotation axes, no free
parameters) and g generic orbits (two free parameters each) with  |G|*g + sum(sizes of special orbits) = N.
"""
import itertools, math, hashlib
from fractions import Fraction
import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------- moments
def dfact(n):
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def monomials(t):
    """exponent triples (a,b,c), c in {0,1}, a+b+c <= t, excluding the constant."""
    return [(a, b, c) for c in (0, 1) for a in range(t + 1) for b in range(t + 1)
            if a + b + c <= t and (a, b, c) != (0, 0, 0)]


def sphere_mean_exact(a, b, c):
    if a % 2 or b % 2 or c % 2:
        return Fraction(0)
    return Fraction(dfact(a - 1) * dfact(b - 1) * dfact(c - 1), dfact(a + b + c + 1))


class Moments:
    def __init__(self, t):
        self.t = t
        self.mons = monomials(t)
        self.E = np.array(self.mons, dtype=int).reshape(-1, 3)
        self.mu = np.array([float(sphere_mean_exact(*m)) for m in self.mons])

    def resid(self, X):
        if len(self.mons) == 0:
            return np.zeros(1)
        P = (X[:, None, :] ** self.E[None, :, :]).prod(axis=2)
        return P.mean(axis=0) - self.mu


# ---------------------------------------------------------------- rotation groups
def rot(axis, ang):
    a = np.array(axis, float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def closure(gens):
    G = [np.eye(3)]
    while True:
        new = []
        for g in G:
            for h in gens:
                m = g @ h
                if not any(np.allclose(m, x, atol=1e-9) for x in G + new):
                    new.append(m)
        if not new:
            return G
        G += new


PHI = (1 + 5 ** 0.5) / 2


def rotation_group(name):
    """C<n>, D<n> (n>=2), T (order 12), O (order 24), I (order 60): proper rotation groups."""
    if name == 'C1':
        return [np.eye(3)]
    if name[0] == 'C':
        n = int(name[1:])
        return closure([rot((0, 0, 1), 2 * math.pi / n)])
    if name[0] == 'D':
        n = int(name[1:])
        return closure([rot((0, 0, 1), 2 * math.pi / n), rot((1, 0, 0), math.pi)])
    if name == 'T':
        return closure([rot((1, 0, 0), math.pi), rot((0, 1, 0), math.pi), rot((1, 1, 1), 2 * math.pi / 3)])
    if name == 'O':
        return closure([rot((0, 0, 1), math.pi / 2), rot((1, 1, 1), 2 * math.pi / 3)])
    if name == 'I':
        return closure([rot((0, 1, PHI), 2 * math.pi / 5), rot((1, 1, 1), 2 * math.pi / 3)])
    raise ValueError(name)


def special_orbits(G):
    """orbits of points on rotation axes (they have non-trivial stabilizer, hence no free parameters)."""
    axes = []
    for g in G:
        if np.allclose(g, np.eye(3)):
            continue
        w, v = np.linalg.eig(g)
        a = np.real(v[:, np.argmin(abs(w - 1))])
        a = a / np.linalg.norm(a)
        axes += [a, -a]
    seen, orbs = set(), []
    for a in axes:
        pts = np.array([g @ a for g in G])
        rounded, idx = np.unique(np.round(pts, 7), axis=0, return_index=True)
        key = tuple(map(tuple, rounded))
        if key in seen:
            continue
        seen.add(key)
        orbs.append(pts[np.sort(idx)])
    return orbs


def strata(G, N, gmax=4):
    """(special orbit subset, number of generic orbits) pairs with total size N."""
    sp = special_orbits(G)
    out = []
    for r in range(len(sp) + 1):
        for S in itertools.combinations(range(len(sp)), r):
            s = sum(len(sp[i]) for i in S)
            if s <= N and (N - s) % len(G) == 0 and (N - s) // len(G) <= gmax:
                out.append((S, (N - s) // len(G)))
    return sp, out


# ---------------------------------------------------------------- search
def min_pair_dist(X):
    D = np.linalg.norm(X[:, None] - X[None], axis=2) + 9 * np.eye(len(X))
    return D.min()


def signature(X, digits=6):
    G = X @ X.T
    iu = np.triu_indices(len(X), 1)
    v = np.sort(np.round(G[iu], digits)) + 0.0
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]


def search_stratum(mom, G, fixed, g, nstart, rng, tol=1e-12):
    """random-start least squares over g generic orbit representatives; returns (best_residual, X) or None."""
    def build(v):
        reps = v.reshape(g, 3)
        reps = reps / np.linalg.norm(reps, axis=1)[:, None]
        pts = [np.array([R @ r for R in G]) for r in reps]
        if len(fixed):
            pts.append(fixed)
        return np.vstack(pts)
    if g == 0:
        X = build(np.zeros(0))
        r = np.abs(mom.resid(X)).max()
        return (r, X) if r < tol else None
    best = (9.0, None)
    for _ in range(nstart):
        v0 = rng.normal(size=3 * g)
        res = least_squares(lambda v: mom.resid(build(v)), v0, method='trf',
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=300)
        X = build(res.x)
        r = np.abs(mom.resid(X)).max()
        if r < best[0] and min_pair_dist(X) > 1e-3:
            best = (r, X)
        if best[0] < tol:
            break
    return best if best[1] is not None and best[0] < tol else None


GROUP_ORDER = ['I', 'O', 'T', 'D6', 'D5', 'D4', 'D3', 'D2', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']


def search_design(N, t, groups=GROUP_ORDER, nstart=25, seed=0, want=3, time_limit=None, verbose=False):
    """find up to `want` inequivalent N-point t-designs; returns list of dicts."""
    import time
    t0 = time.time()
    rng = np.random.default_rng(seed)
    mom = Moments(t)
    found, sigs = [], set()
    for gname in groups:
        G = rotation_group(gname)
        sp, sts = strata(G, N, gmax=N)
        for S, g in sts:
            if gname == 'C1' and g != N:
                continue
            if 3 * g > 60:
                continue
            if time_limit and time.time() - t0 > time_limit:
                return found
            fixed = np.vstack([sp[i] for i in S]) if S else np.zeros((0, 3))
            r = search_stratum(mom, G, fixed, g, nstart, rng)
            if r is None:
                continue
            res, X = r
            sg = signature(X)
            if sg in sigs:
                continue
            sigs.add(sg)
            found.append(dict(X=X, group=gname, order=len(G), special=[len(sp[i]) for i in S],
                              generic_orbits=g, residual=float(res), signature=sg, min_dist=float(min_pair_dist(X))))
            if verbose:
                print(f"  N={N} t={t}: {gname} special={[len(sp[i]) for i in S]} g={g} resid={res:.1e}", flush=True)
            if len(found) >= want:
                return found
    return found


# ---------------------------------------------------------------- high precision
def refine_mp(X, t, dps=50, iters=5):
    """Gauss-Newton on the moment equations plus |x_i|^2 = 1 at `dps` digits (ridge-regularised normal
    equations; the system is consistent so convergence is quadratic).  Returns (coordinates as strings,
    max residual as mpf)."""
    import mpmath as mp
    mp.mp.dps = dps
    mons = monomials(t)
    N = len(X)
    mu = [mp.mpf(sphere_mean_exact(*m).numerator) / sphere_mean_exact(*m).denominator for m in mons]
    x = [[mp.mpf(float(v)) for v in row] for row in X]
    M = len(mons)
    ridge = mp.mpf(10) ** (-(dps // 2))
    for it in range(iters):
        r = mp.matrix(M + N, 1)
        J = mp.matrix(M + N, 3 * N)
        for k, (a, b, c) in enumerate(mons):
            s = mp.mpf(0)
            for i in range(N):
                xi, yi, zi = x[i]
                s += xi ** a * yi ** b * zi ** c
                J[k, 3 * i + 0] = (a * xi ** (a - 1) * yi ** b * zi ** c if a else 0) / N
                J[k, 3 * i + 1] = (b * xi ** a * yi ** (b - 1) * zi ** c if b else 0) / N
                J[k, 3 * i + 2] = (c * xi ** a * yi ** b * zi ** (c - 1) if c else 0) / N
            r[k] = s / N - mu[k]
        for i in range(N):
            r[M + i] = x[i][0] ** 2 + x[i][1] ** 2 + x[i][2] ** 2 - 1
            for j in range(3):
                J[M + i, 3 * i + j] = 2 * x[i][j]
        res = max(abs(r[k]) for k in range(M + N))
        if it == iters - 1 or res < mp.mpf(10) ** (-(dps - 6)):
            break
        A = J.T * J
        for q in range(3 * N):
            A[q, q] += ridge
        delta = mp.lu_solve(A, -(J.T * r))
        for i in range(N):
            for j in range(3):
                x[i][j] += delta[3 * i + j]
    # final residual
    nrm = [mp.sqrt(sum(v ** 2 for v in row)) for row in x]
    x = [[v / nrm[i] for v in row] for i, row in enumerate(x)]
    R = 0
    for (a, b, c), m_ in zip(mons, mu):
        s = sum(xi ** a * yi ** b * zi ** c for xi, yi, zi in x) / N - m_
        R = max(R, abs(s))
    return [[mp.nstr(v, dps) for v in row] for row in x], mp.nstr(R, 6)
