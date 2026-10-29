import numpy as np
import pytest

from sphcap import configs, discrepancy
from sphcap.utils import cap_measure, random_points


def test_cap_measure_matches_monte_carlo():
    rng = np.random.default_rng(0)
    X = random_points(200_000, n=3, rng=rng)
    for t in (-0.6, -0.2, 0.0, 0.3, 0.8):
        emp = np.mean(X[:, 0] >= t)
        formula = float(cap_measure(np.array(t), 3))
        assert abs(emp - formula) < 0.005, (t, emp, formula)


def test_cap_measure_endpoints():
    assert cap_measure(np.array(1.0), 3) == pytest.approx(0.0, abs=1e-9)
    assert cap_measure(np.array(-1.0), 3) == pytest.approx(1.0, abs=1e-9)
    assert cap_measure(np.array(0.0), 3) == pytest.approx(0.5, abs=1e-9)


def test_minimal_four_point_configuration_value():
    X = configs.minimal_four_point_configuration()
    assert X.shape == (4, 3)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    res_exact = discrepancy(X, method="exact")
    res_brute = discrepancy(X, method="bruteforce")
    assert res_exact.value == pytest.approx(3 / 8, abs=1e-9)
    assert res_brute.value == pytest.approx(3 / 8, abs=1e-9)


def test_regular_tetrahedron_is_worse_than_squished():
    reg = configs.regular_tetrahedron()
    opt = configs.minimal_four_point_configuration()
    d_reg = discrepancy(reg, method="exact").value
    d_opt = discrepancy(opt, method="exact").value
    assert d_opt == pytest.approx(3 / 8, abs=1e-9)
    assert d_reg > d_opt + 1e-6  # regular tetrahedron strictly worse


def test_disphenoid_family_minimum_at_optimal_t():
    """Sweep t across the disphenoid family; the minimum of the exact
    discrepancy along the family should sit at the derived optimal t,
    matching the value 3/8 to high precision, and no other t should
    beat it."""
    t_star = np.sqrt((13 - np.sqrt(105)) / 32)
    ts = np.linspace(0.05, 0.95, 181)
    vals = []
    for t in ts:
        X = configs.disphenoid_family(t)
        vals.append(discrepancy(X, method="exact").value)
    vals = np.array(vals)
    i_min = np.argmin(vals)
    assert abs(ts[i_min] - t_star) < (ts[1] - ts[0]) + 1e-9
    assert vals.min() == pytest.approx(3 / 8, abs=2e-3)


def test_bruteforce_and_exact_agree_random_configs():
    rng = np.random.default_rng(1)
    for N in (3, 4, 5, 6):
        X = random_points(N, n=3, rng=rng)
        d_bf = discrepancy(X, method="bruteforce").value
        d_ex = discrepancy(X, method="exact").value
        assert d_bf == pytest.approx(d_ex, abs=1e-8), (N, d_bf, d_ex)


def test_axis_sweep_and_monte_carlo_are_lower_bounds():
    rng = np.random.default_rng(2)
    X = random_points(40, n=3, rng=rng)
    d_exact = discrepancy(X, method="exact").value
    d_axis = discrepancy(X, method="axis_sweep").value
    d_mc = discrepancy(X, method="monte_carlo", n_samples=5000, rng=rng).value
    assert d_axis <= d_exact + 1e-9
    assert d_mc <= d_exact + 1e-9
    # axis_sweep should be a reasonably tight lower bound in practice
    assert d_axis > 0.5 * d_exact


def test_platonic_solids_are_unit_norm_and_nontrivial():
    for name in ("tetrahedron", "octahedron", "cube", "icosahedron", "dodecahedron"):
        X = configs.platonic_solid(name)
        assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
        d = discrepancy(X, method="exact").value
        assert 0.0 < d < 1.0


def test_diamond_ensemble_point_count_and_shape():
    M = 5
    X = configs.diamond_ensemble(M)
    expected_N = 2 + 4 * M ** 2  # Etayo Thm 1.4 model: N = 2 + 4M^2
    assert X.shape == (expected_N, 3)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def test_diamond_ensemble_discrepancy_reasonable_scaling():
    # D_sup,cap should shrink roughly like N^{-1/2} (Etayo Thm 1.4);
    # just sanity check monotonic-ish decrease with N, not the constant.
    vals = []
    for M in (2, 4, 8):
        X = configs.diamond_ensemble(M)
        vals.append(discrepancy(X, method="axis_sweep").value)
    assert vals[-1] < vals[0]


def test_fibonacci_sphere_shape():
    X = configs.fibonacci_sphere(500)
    assert X.shape == (500, 3)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def test_hecke_points_generate_unit_vectors():
    X = configs.hecke_points(p=5, depth=3, max_points=100)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-8)
    assert X.shape[0] <= 100


def test_riesz_energy_minimize_improves_energy():
    from sphcap.configs import riesz_energy_minimize, random_uniform

    def energy(X, s=1.0):
        diff = X[:, None, :] - X[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        return np.sum(d2 ** (-s / 2))

    X0 = random_uniform(20, seed=3)
    X1 = riesz_energy_minimize(20, s=1.0, iters=300, lr=0.1, seed=3, x0=X0)
    assert energy(X1) < energy(X0)


def test_optimizer_recovers_known_n4_minimum():
    from sphcap.optimize import minimize_discrepancy_multistart
    res = minimize_discrepancy_multistart(N=4, n_starts=4, n_iter=120, lr=0.08,
                                           seed=42, top_k=8)
    assert res.value == pytest.approx(3 / 8, abs=1e-4)


def test_random_configs_never_beat_the_proven_minimum():
    rng = np.random.default_rng(4)
    opt = discrepancy(configs.minimal_four_point_configuration(), method="exact").value
    for _ in range(200):
        X = random_points(4, n=3, rng=rng)
        d = discrepancy(X, method="exact").value
        assert d >= opt - 1e-9
