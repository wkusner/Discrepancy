"""Three points on a great circle: the cap through them is a hemisphere (singular Gram matrix).
N = 3 by hand: every great-circle triple has D >= 1/2 (hemispheres); D = 1/2 exactly when every arc is at least
g* = 2 arccos(2/3), where the two-point caps start to contribute 2/3 - (1 - cos(arc/2))/2."""
import numpy as np
import pytest

from sphcap.discrepancy import discrepancy
from sphcap.gram import G_STAR, n3_points

DEG = np.pi / 180


@pytest.mark.parametrize("method", ["exact", "bruteforce"])
@pytest.mark.parametrize("a,b", [(120, 120), (110, 110), (100, 120), (G_STAR / DEG + 0.5, G_STAR / DEG + 0.5)])
def test_n3_family_is_one_half(method, a, b):
    assert discrepancy(n3_points(a * DEG, b * DEG), method=method).value == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("method", ["exact", "bruteforce"])
def test_n3_below_gstar_two_point_cap(method):
    # arcs 90, 135, 135: the 90-degree pair's smallest cap gives 2/3 - (1 - cos 45)/2
    want = 2 / 3 - (1 - np.cos(45 * DEG)) / 2
    assert discrepancy(n3_points(90 * DEG, 135 * DEG), method=method).value == pytest.approx(want, abs=1e-12)


def test_octahedron_hemisphere():
    # four points on the equator: the closed hemisphere holds 5 of 6, area 1/2 -> deviation 1/3
    X = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    assert discrepancy(X).value >= 1 / 3 - 1e-12
