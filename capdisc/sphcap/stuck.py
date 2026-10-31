"""sphcap.stuck -- classify the stuck points of the descent (NOT TRUSTED; numerical labels for the atlas).

A stuck point X is where sequential-LP descent stops: the frozen-count model admits no first-order descent. That
does not yet make X a local minimizer, because the frozen model fixes two things that a nearby configuration may
change:

  (a) assignments: a point on the circle of a worst cap that is not one of its defining points is counted (closed
      cap, branch "up") or not (open complement, branch "low"); moving it off the circle to the other side lowers
      that branch by 1/N, so the branch stops being worst.  A direction may therefore descend by *switching* such a
      point instead of decreasing the branch smoothly.  (The other cap of the same circle is a branch of its own,
      the "low" branch of the same defining set, and is already in the active set when it ties.)
  (b) second order: when the active gradients balance with zero multipliers or do not span the quotient (rank
      < 2N-3), there are flat directions, and the second-order behaviour of the Lagrangian sum lam_b f_b along them
      decides.

Classes:
  sharp minimum       balanced, all multipliers > 0, gradients span: D grows linearly in every direction
  strict minimum      no first-order exit, Lagrangian Hessian positive definite on the flat directions
  non-strict minimum  flat directions along which D stays constant to second order, and probing does not descend
                      (a curve or family of minimizers)
  saddle              a flat direction of negative curvature: D decreases at second order
  exit                a first-order descent direction that switches assignments (a)
  not stationary      a first-order descent direction in the frozen model (the descent was stopped early)
Every non-sharp verdict is also probed: small random perturbations followed by descent must not go lower.
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space
from . import balance as B

TOL = 1e-9          # incidence tolerance (a point is on a circle)
ACT = 1e-7          # active-set tolerance


def margin(Y, b, j):
    """signed distance-like margin of point j for the cap of branch b at configuration Y (>0: inside the open cap)."""
    g = B.cap_geometry(Y, b.I)
    if g is None: return np.nan
    t, w, _ = g
    return float(w @ Y[j] - t)


def margin_grad(X, E, b, j, h=1e-7):
    """gradient of margin(., b, j) in tangent coordinates (2N), by central differences."""
    n = 2 * len(X); g = np.zeros(n)
    for k in range(n):
        v = np.zeros(n); v[k] = h
        g[k] = (margin(B.retract(X, E, v), b, j) - margin(B.retract(X, E, -v), b, j)) / (2 * h)
    return g


def incidences(X, b):
    """points on the circle of branch b that are not defining points"""
    g = B.cap_geometry(X, b.I)
    if g is None: return []
    t, w, _ = g
    return [j for j in range(len(X)) if j not in b.I and abs(w @ X[j] - t) <= TOL]


def descent_lp(G, R, extra_rows=()):
    """max s s.t. G v <= -s, rows <= -s, R v = 0, |v|_inf <= 1; returns (s, v)."""
    n = G.shape[1]; A = np.vstack([G] + [r[None] for r in extra_rows]) if extra_rows else G
    res = linprog(np.append(np.zeros(n), -1.0), A_ub=np.hstack([A, np.ones((len(A), 1))]), b_ub=np.zeros(len(A)),
                  A_eq=np.hstack([R, np.zeros((len(R), 1))]), b_eq=np.zeros(len(R)),
                  bounds=[(-1, 1)] * n + [(None, None)], method='highs')
    return (-res.fun, res.x[:n]) if res.status == 0 else (0.0, None)


def aligned_distance(X, Y):
    """max point distance after the best rotation/reflection (labels fixed; for small moves)"""
    U, _, Vt = np.linalg.svd(Y.T @ X); Q = U @ Vt
    return float(np.linalg.norm(Y @ Q - X, axis=1).max())


def local_descent(Y, D0, cap, iters=300, margin=0.05, tol=1e-9):
    """sequential-LP descent from Y with the trust region capped at `cap`; stops at the first iterate with
    D < D0 - 1e-9 and returns it (or the last iterate)."""
    Y = B.normalize(Y); D = B.discrepancy(Y, tol); delta = cap; n = 2 * len(Y)
    for _ in range(iters):
        if D < D0 - 1e-9: break
        near = [b for b in B.branches(Y, tol) if b.value >= D - margin]
        G, E = B.gradient_matrix(Y, near); fv = np.array([b.value for b in near])
        res = linprog(np.append(np.zeros(n), 1.0), A_ub=np.hstack([G, -np.ones((len(near), 1))]), b_ub=-fv,
                      bounds=[(-delta, delta)] * n + [(None, None)], method='highs')
        if res.status != 0: delta *= 0.5; continue
        Z = B.retract(Y, E, res.x[:n]); Dn = B.discrepancy(Z, tol)
        if Dn < D - 1e-15: Y, D = Z, Dn; delta = min(2 * delta, cap)
        else: delta *= 0.5
        if delta < 1e-13: break
    return Y, D


def probe(X, D0, rng, trials=12):
    """lowest D found near X: from a perturbation of size s, a descent whose steps are at most 10 s, accepted only
    if it gets below D0 within distance 20 s of X.  (A plain descent from a perturbation may cross a low ridge into
    a neighbouring basin: from the N = 5 valley, perturbations of size 1e-4 start above D0 and descend to the
    optimum 0.031 away, which says nothing about points arbitrarily close to the valley.)"""
    best = D0
    for _ in range(trials):
        sc = rng.choice([1e-6, 1e-5, 1e-4])
        Y, Dy = local_descent(X + rng.normal(scale=sc, size=X.shape), D0, cap=10 * sc)
        if Dy < D0 - 1e-9 and aligned_distance(X, Y) < 20 * sc: best = min(best, Dy)
    return best


def classify(X, rng=None):
    rng = rng or np.random.default_rng(0)
    X = B.normalize(np.asarray(X, float)); N = len(X)
    D0 = B.discrepancy(X, TOL)
    # a first-order descent of the frozen model is only believed if the actual descent makes progress; otherwise
    # the active set was too narrow for the precision of X, and it is widened
    for act_tol in (ACT, 1e-6, 1e-5):
        rep = B.balance_report(X, tol=TOL, act_tol=act_tol)
        act = [b for b in rep.active if b.kind != 'const']
        G, E = B.gradient_matrix(X, act); R = B.rotation_generators(X, E)
        s, v = descent_lp(G, R)
        s_tol = max(1e-6, 10 * rep.residual)        # first-order rates below the precision of X are noise
        if s <= s_tol: break
        Yd, Dd = B.slp_descent(X, iters=400)
        if Dd < D0 - 1e-9:
            return dict(D=D0, n_active=len(act), rank=rep.rank, dim=rep.dim_quot, min_lambda=rep.min_lambda,
                        incidences=rep.extra_incidences, cls='not stationary', escape_rate=s, descends_to=Dd,
                        moved=aligned_distance(X, Yd))
    out = dict(D=rep.D, n_active=len(act), rank=rep.rank, dim=rep.dim_quot, min_lambda=rep.min_lambda,
               incidences=rep.extra_incidences, act_tol=act_tol)
    out['first_order_rate'] = s
    if s > s_tol:
        out.update(cls='stuck, frozen model inconclusive'); return out
    # first order with switches: an active branch may be escaped by moving one of its incident points off the circle
    # to the side that lowers it ("up": out of the closed cap; "low": into the open cap)
    esc = []
    for bi, b in enumerate(act):
        opts = []
        for j in incidences(X, b):
            gm = margin_grad(X, E, b, j)
            opts.append(-gm if b.kind == 'low' else gm)      # row r with r.v < 0 moves j to the escaping side
        if opts: esc.append((bi, opts))
    best = (0.0, None, None)
    if esc:
        choices = list(itertools.product(*[[None] + list(range(len(o))) for _, o in esc]))
        if len(choices) > 4096: choices = [tuple(rng.integers(-1, len(o)) for _, o in esc) for _ in range(4096)]
        for ch in choices:
            if all(c is None or c == -1 for c in ch): continue
            keep = np.ones(len(act), bool); rows = []
            for (bi, opts), c in zip(esc, ch):
                if c is None or c == -1: continue
                keep[bi] = False; rows.append(opts[c])
            s2, v2 = descent_lp(G[keep], R, rows)
            if s2 > best[0]: best = (s2, v2, ch)
    if best[0] > s_tol:
        out.update(cls='exit', escape_rate=best[0], switched=int(sum(c not in (None, -1) for c in best[2]))); return out
    if rep.sharp:
        out.update(cls='sharp minimum'); return out
    # second order on the flat directions: K = {v : G_+ v = 0 (positive multipliers), v orthogonal to rotations}
    lam = np.maximum(rep.lam[:len(rep.active)], 0)
    lam_act = np.array([l for l, b in zip(lam, rep.active) if b.kind != 'const'])
    pos = lam_act > 1e-9
    Z = null_space(np.vstack([G[pos], R]), rcond=1e-7) if pos.any() else null_space(R, rcond=1e-7)
    out['flat_dim'] = int(Z.shape[1])
    if Z.shape[1] == 0:            # the positive-weight caps alone span the quotient: D grows linearly
        cls = 'sharp minimum (zero weights on some worst caps)'
    else:
        H = B.lagrangian_hessian(X, act, lam_act)
        Hz = Z.T @ H @ Z; ev, U = np.linalg.eigh((Hz + Hz.T) / 2)
        scale = max(np.abs(np.linalg.eigvalsh((H + H.T) / 2)).max(), 1e-12)     # size of the whole Hessian
        out['hessian_flat_eigs'] = [float(e) for e in ev]
        zero_inactive = [b for b, p in zip(act, pos) if not p]
        Gz = G[~pos] if (~pos).any() else np.zeros((0, G.shape[1]))
        neg = [k for k in range(len(ev)) if ev[k] < -1e-4 * scale]
        feasible_neg = False
        for k in neg:
            d = Z @ U[:, k]
            if len(Gz) == 0 or np.all(Gz @ d <= 1e-9) or np.all(Gz @ (-d) <= 1e-9): feasible_neg = True
        if feasible_neg: cls = 'saddle'
        elif np.all(ev > 1e-4 * scale): cls = 'strict minimum'
        else: cls = 'non-strict minimum'
    Dp = probe(X, rep.D, rng)
    out['probe_min_D'] = Dp
    if Dp < rep.D - 1e-9 and cls in ('strict minimum', 'non-strict minimum'):
        cls += ' (probe descends: not a local minimum)'
    out['cls'] = cls
    return out




def exit_direction(X, rng=None):
    """a first-order descent direction at a stuck point X that switches incident points to the other side of worst
    caps' circles, or None. Returns (rate, v, E) with v in tangent coordinates."""
    rng = rng or np.random.default_rng(0)
    X = B.normalize(np.asarray(X, float))
    rep = B.balance_report(X, tol=TOL, act_tol=ACT)
    act = [b for b in rep.active if b.kind != 'const']
    G, E = B.gradient_matrix(X, act); R = B.rotation_generators(X, E)
    esc = []
    for bi, b in enumerate(act):
        opts = [(-margin_grad(X, E, b, j) if b.kind == 'low' else margin_grad(X, E, b, j)) for j in incidences(X, b)]
        if opts: esc.append((bi, opts))
    if not esc: return None
    choices = list(itertools.product(*[[None] + list(range(len(o))) for _, o in esc]))
    if len(choices) > 4096: choices = [tuple(rng.integers(-1, len(o)) for _, o in esc) for _ in range(4096)]
    best = (0.0, None)
    for ch in choices:
        if all(c is None or c == -1 for c in ch): continue
        keep = np.ones(len(act), bool); rows = []
        for (bi, opts), c in zip(esc, ch):
            if c is None or c == -1: continue
            keep[bi] = False; rows.append(opts[c])
        s, v = descent_lp(G[keep], R, rows)
        if s > best[0]: best = (s, v)
    s_tol = max(1e-6, 10 * rep.residual)
    return (best[0], best[1], E) if best[0] > s_tol else None


def follow_exits(X, D, max_exits=8):
    """from a stuck point, step along exit directions and descend again, while exits exist"""
    for _ in range(max_exits):
        ex = exit_direction(X)
        if ex is None: break
        s, v, E = ex
        moved = False
        for eps in (1e-4, 1e-3):
            Y = B.retract(X, E, eps * v)
            if B.discrepancy(Y) < D:
                Yd, Dd = B.slp_descent(Y)
                if Dd < D - 1e-12: X, D, moved = Yd, Dd, True; break
        if not moved: break
    return X, D
