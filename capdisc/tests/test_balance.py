"""Tests for the balance / Gram / high-precision modules."""
import itertools
import numpy as np
import pytest

from sphcap import balance as B, gram as GM
from sphcap.discrepancy import discrepancy as D_exact
from sphcap.data import load_critical


def rand_cfg(N, seed):
    return B.normalize(np.random.default_rng(seed).normal(size=(N, 3)))


@pytest.mark.parametrize("N", [3, 4, 5, 6, 7])
def test_branch_max_equals_exact(N):
    for s in range(3):
        X = rand_cfg(N, 100 * N + s)
        assert abs(B.discrepancy(X) - D_exact(X, "exact").value) < 1e-12


def test_gradient_matches_finite_differences():
    X = rand_cfg(5, 7)
    E = B.tangent_frames(X)
    for b in B.branches(X)[1:]:
        g = B.branch_gradient(X, b, E).ravel()
        fd = np.zeros(10)
        for i in range(10):
            v = np.zeros(10)
            v[i] = 1e-6
            fd[i] = (B.frozen_value(B.retract(X, E, v), b) - B.frozen_value(B.retract(X, E, -v), b)) / 2e-6
        assert np.abs(g - fd).max() < 1e-7


def test_gram_formulas():
    X = rand_cfg(6, 3)
    G = GM.gram(X)
    for I in itertools.combinations(range(6), 3):
        t, w, beta = B.cap_geometry(X, I)
        assert abs(t * t - GM.t_squared(G, I)) < 1e-12
        for j in set(range(6)) - set(I):
            small = X[j] @ w * np.sign(t) >= abs(t)
            assert (GM.membership(G, I, j) >= 0) == small


def test_rotation_generators_are_null_for_gradients():
    X = rand_cfg(5, 11)
    bs = B.branches(X)
    G, E = B.gradient_matrix(X, bs)
    R = B.rotation_generators(X, E)
    assert np.abs(G @ R.T).max() < 1e-12


def test_n3_family():
    for a, b in [(2.0, 2.0), (1.7, 2.5), (1.9, 1.9)]:
        assert abs(B.discrepancy(GM.n3_points(a, b)) - 0.5) < 1e-12
    assert B.discrepancy(GM.n3_points(1.60, 2.3)) > 0.5 + 1e-3


def test_n4_arc_and_cubic():
    for e in (GM.N4_ARC_END_SYMMETRIC, GM.N4_ARC_END_EDGE):
        assert abs(GM.n4_cubic(*e)) < 1e-12
        assert abs(B.discrepancy(GM.disphenoid_points(*e)) - 3 / 8) < 1e-12
    pts = GM.n4_cubic_curve_points(1500)
    for p in pts[::25]:
        d = B.discrepancy(GM.disphenoid_points(*p))
        assert d >= 3 / 8 - 1e-12
        assert (abs(d - 3 / 8) < 1e-9) == (p.max() <= 9 / 16 + 1e-9) or abs(p.max() - 9 / 16) < 2e-3


def test_n4_pairing_cancels():
    """Both up and low of every face are active: forces cancel pairwise (poster pairing)."""
    X = GM.disphenoid_points(*GM.N4_ARC_END_SYMMETRIC)
    rep = B.balance_report(X)
    kinds = sorted(b.kind for b in rep.active)
    assert kinds.count("up") == kinds.count("low") == 4
    assert rep.stationary and rep.min_lambda > 0 and not rep.sharp   # flat arc direction


def test_n5_best_is_sharp_and_matches_polynomial():
    c = load_critical()["N5_best"]
    X = np.array(c["points"])
    rep = B.balance_report(X, act_tol=1e-7)
    assert rep.sharp and len(rep.active) == 8 and rep.rank == 7
    import mpmath as mp
    mp.mp.dps = 60
    D = mp.mpf(c["D_60digits"])
    val = sum(mp.mpf(a) * D ** i for i, a in enumerate(GM.MINPOLY["N5_best"]))
    assert abs(val) < mp.mpf(10) ** -40


def test_n5_symmetric_ansatz_and_elimination():
    sp = pytest.importorskip("sympy")
    X = GM.n5_symmetric_points(0.056030061511, -0.514411246903)
    assert abs(B.discrepancy(X) - 0.32332565099805) < 1e-9
    fl = GM.n5_eliminate()
    D = sp.Symbol("D")
    P = sp.Poly(sum(a * D ** i for i, a in enumerate(GM.MINPOLY["N5_best"])), D)
    assert any(sp.Poly(f, D).degree() == 12 and sp.div(sp.Poly(f, D), P)[1].is_zero for f, m in fl)


def test_n6_best_is_sharp():
    c = load_critical()["N6_best"]
    rep = B.balance_report(np.array(c["points"]), act_tol=1e-7)
    assert rep.sharp and len(rep.active) == 11 and rep.rank == 9
    assert abs(rep.D - 0.27867881517624) < 1e-9


def test_slp_reaches_n5_best():
    best = min(B.slp_descent(rand_cfg(5, s), iters=200)[1] for s in range(12))
    assert abs(best - 0.323325650998) < 1e-8


def test_disphenoid_closed_form():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.dirichlet([1, 1, 1])
        assert abs(GM.disphenoid_discrepancy(*x) - B.discrepancy(GM.disphenoid_points(*x))) < 1e-9
