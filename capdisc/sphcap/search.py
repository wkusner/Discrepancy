"""
sphcap.search
=============

Candidate search for discrepancy minimisers: structured starting distributions,
SLP descent on the frozen-count envelope (balance.slp_descent), basin-hopping kicks
to escape non-sharp (paired / flat) critical strata, and clustering of the
resulting critical configurations into classes.

Starting distributions (mixed; weights in ``DEFAULT_MIX``):
  riesz     random points relaxed toward a Riesz s-energy minimiser (s random), then jittered
  spiral    generalised spiral / Fibonacci points, random rotation, jittered
  rings     symmetric seeds: poles + rings of m_k equally spaced points (sum = N)
            at random heights and phases (C_nv / D_n / D_nd-type), jittered
  uniform   iid uniform on the sphere (control)
  hop       a kick (random rotation of a random subset + Gaussian noise) of one
            of the current best classes
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import balance as B


# ---------------------------------------------------------------------------
# starting distributions
# ---------------------------------------------------------------------------
def _unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def _rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def start_uniform(N, rng):
    return _unit(rng.normal(size=(N, 3)))


def start_riesz(N, rng, steps=200):
    s = rng.uniform(0.5, 4.0)
    X = start_uniform(N, rng)
    for _ in range(steps):
        D = X[:, None, :] - X[None, :, :]
        r = np.linalg.norm(D, axis=2) + np.eye(N)
        F = (D / r[..., None] ** (s + 2)).sum(1)
        F -= (F * X).sum(1, keepdims=True) * X
        X = _unit(X + 0.05 * F / (np.abs(F).max() + 1e-12))
    return _unit(X + rng.normal(scale=rng.choice([0.02, 0.08, 0.2]), size=X.shape))


def start_spiral(N, rng):
    k = np.arange(N) + rng.uniform(0.2, 0.8)
    z = 1 - 2 * k / N
    ga = np.pi * (3 - np.sqrt(5)) * rng.uniform(0.9, 1.1)
    ph = ga * np.arange(N)
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    X = np.stack([r * np.cos(ph), r * np.sin(ph), z], 1) @ _rand_rot(rng).T
    return _unit(X + rng.normal(scale=0.05, size=X.shape))


def _compositions(N):
    """(poles, ring sizes) with ring sizes >= 2, a few random choices."""
    out = []
    for poles in (0, 1, 2):
        rest = N - poles
        for parts in itertools.combinations_with_replacement(range(2, rest + 1), 1):
            pass
        # enumerate partitions of rest into parts >= 2 (small N, cheap)
        def parts(n, mx):
            if n == 0:
                yield []
                return
            for p in range(min(n, mx), 1, -1):
                for t in parts(n - p, p):
                    yield [p] + t
        for p in parts(rest, rest):
            out.append((poles, p))
    return out


def start_rings(N, rng):
    comps = _compositions(N)
    poles, rings = comps[rng.integers(len(comps))]
    rng.shuffle(rings)
    P = []
    if poles >= 1:
        P.append([0, 0, 1.0])
    if poles == 2:
        P.append([0, 0, -1.0])
    zs = np.sort(rng.uniform(-0.9, 0.9, size=len(rings)))[::-1]
    for m, z in zip(rings, zs):
        ph0 = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(1 - z * z)
        for j in range(m):
            ph = ph0 + 2 * np.pi * j / m
            P.append([r * np.cos(ph), r * np.sin(ph), z])
    X = np.array(P) @ _rand_rot(rng).T
    return _unit(X + rng.normal(scale=rng.choice([0.0, 0.01, 0.05]), size=X.shape))


def kick(X, rng):
    N = len(X)
    Y = X.copy()
    m = rng.integers(1, max(2, N // 2) + 1)
    idx = rng.choice(N, m, replace=False)
    ang = rng.choice([0.05, 0.15, 0.4])
    R = _rand_rot(rng)
    # small rotation: interpolate toward identity
    Y[idx] = _unit((1 - ang) * Y[idx] + ang * (Y[idx] @ R.T))
    return _unit(Y + rng.normal(scale=rng.choice([0.01, 0.03, 0.1]), size=Y.shape))


STARTS = dict(uniform=start_uniform, riesz=start_riesz, spiral=start_spiral, rings=start_rings)
DEFAULT_MIX = dict(riesz=0.3, rings=0.3, spiral=0.1, uniform=0.1, hop=0.2)


# ---------------------------------------------------------------------------
# invariants and clustering
# ---------------------------------------------------------------------------
def invariant(X, digits=6):
    """Labelling/rotation invariant: sorted multiset of Gram entries."""
    G = X @ X.T
    v = np.sort(G[np.triu_indices(len(X), 1)])
    return tuple(np.round(v, digits))


def gram_automorphisms(X, tol=1e-6):
    """Number of permutations p with G[p][:,p] = G (= symmetry group order, for full-rank configs)."""
    N = len(X)
    G = X @ X.T
    count = 0
    # backtracking over permutations with Gram consistency
    def bt(p):
        nonlocal count
        k = len(p)
        if k == N:
            count += 1
            return
        for j in range(N):
            if j in p:
                continue
            if all(abs(G[k, i] - G[j, p[i]]) < tol for i in range(k)):
                bt(p + [j])
    bt([])
    return count


@dataclass
class Klass:
    D: float
    X: np.ndarray
    hits: int = 1
    sources: dict = field(default_factory=dict)
    report: object = None


def descend(X, rng, max_kicks=6):
    """SLP descent; if the end point is not sharp, kick and retry, keeping the best."""
    Y, D = B.slp_descent(X)
    best = (D, Y)
    for _ in range(max_kicks):
        rep = B.balance_report(best[1])
        if rep.sharp:
            break
        Z, Dz = B.slp_descent(kick(best[1], rng))
        if Dz < best[0] - 1e-12:
            best = (Dz, Z)
    return best[1], best[0]


def search(N, runs, seed=0, mix=None, keep_top=12, verbose=True, time_limit=None, init=None):
    """init: optional list of configurations used as basin-hopping seeds from the start."""
    rng = np.random.default_rng(seed)
    mix = mix or DEFAULT_MIX
    names, w = list(mix), np.array(list(mix.values()), float)
    w /= w.sum()
    classes: dict = {}
    for X in (init or []):
        X = _unit(np.asarray(X, float))
        D = B.discrepancy(X)
        classes[(round(D, 8), invariant(X, 4))] = Klass(D, X, 0, {'init': 1})
    t0 = time.time()
    for r in range(runs):
        src = rng.choice(names, p=w)
        if src == "hop" and classes:
            top = sorted(classes.values(), key=lambda k: k.D)[:keep_top]
            X0 = kick(top[rng.integers(len(top))].X, rng)
        else:
            src = "riesz" if src == "hop" else src
            X0 = STARTS[src](N, rng)
        Y, D = descend(X0, rng)
        key = (round(D, 8), invariant(Y, 4))
        if key in classes:
            k = classes[key]
            k.hits += 1
            k.sources[src] = k.sources.get(src, 0) + 1
        else:
            classes[key] = Klass(D, Y, 1, {src: 1})
        if verbose and (r + 1) % 25 == 0:
            best = min(k.D for k in classes.values())
            print(f"  N={N} run {r+1}/{runs}  classes {len(classes)}  best {best:.10f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if time_limit and time.time() - t0 > time_limit:
            break
    out = sorted(classes.values(), key=lambda k: k.D)
    for k in out[:keep_top]:
        k.report = B.balance_report(k.X)
    return out


def summarize(classes, top=8):
    rows = []
    for k in classes[:top]:
        r = k.report or B.balance_report(k.X)
        rows.append(dict(D=k.D, hits=k.hits, active=len(r.active), rank=r.rank, dimq=r.dim_quot,
                         min_lambda=float(r.min_lambda), sharp=bool(r.sharp), incid=int(r.extra_incidences),
                         aut=gram_automorphisms(k.X), sources=k.sources))
    return rows
