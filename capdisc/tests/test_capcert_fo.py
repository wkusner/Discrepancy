"""Tests of the interval bounds in the C mesh (csrc/rig.h zeroth order, csrc/fo.h first order). Tests are not
proofs; they catch errors that a proof about the program would miss.

For random cells near the N = 5, 6 optima the bounds printed by capcert_rig (CAPCERT_EVAL mode) must not exceed
the discrepancy at sampled configurations of the cell (floating-point oracle, with a 1e-12 tolerance), and the
first-order bound must be the stronger one on small cells (otherwise it would be dead code).

    python -m pytest tests/test_capcert_fo.py -q
"""
import os, subprocess, sys, tempfile
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sphcap import discrepancy as exact_discrepancy          # noqa: E402  (test oracle only: floating point)
from tests.test_proof_core import points_float, local_q      # noqa: E402

BIN = os.path.join(tempfile.gettempdir(), 'capcert_rig_test')
rng = np.random.default_rng(2026)


@pytest.fixture(scope='module')
def binary():
    subprocess.run(['gcc', '-O2', '-Wall', '-Wno-unused-function', '-DRIGOROUS', '-ffp-contract=off', '-fno-fast-math',
                    '-o', BIN, os.path.join(ROOT, 'csrc', 'capcert.c'), '-lm'], check=True, capture_output=True)
    return BIN


def evaluate(binary, N, cells):
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        for c, w in cells:
            np.concatenate([c, w, [0.0]]).astype(np.float64).tofile(f)
    out = subprocess.run([binary, str(N), '1', '1', '/dev/null'], env=dict(os.environ, CAPCERT_EVAL=f.name),
                         check=True, capture_output=True, text=True).stdout.split('\n')
    os.unlink(f.name)
    return [tuple(float(x) for x in line.split()) for line in out if line.strip()]


@pytest.mark.parametrize('N,name', [(5, 'N5_best'), (6, 'N6_best')])
def test_bounds_below_discrepancy(binary, N, name):
    qs = local_q(name); nq = 2 * N - 3; cells = []
    for _ in range(60):
        c = qs + rng.normal(0, 10 ** rng.uniform(-3, -0.7), nq)
        w = np.full(nq, 10 ** rng.uniform(-5, -1.5)) * rng.uniform(0.5, 1.5, nq)
        cells.append((c, w))
    res = evaluate(binary, N, cells)
    assert len(res) == len(cells)
    stronger = 0
    for (c, w), (lb0, sym, lb1) in zip(cells, res):
        Dmin = min(exact_discrepancy(points_float(c + rng.uniform(-1, 1, nq) * w, N)).value for _ in range(8))
        Dmin = min(Dmin, exact_discrepancy(points_float(c, N)).value)
        assert lb0 <= Dmin + 1e-12, f'zeroth-order bound {lb0} above D = {Dmin}'
        assert lb1 <= Dmin + 1e-12, f'first-order bound {lb1} above D = {Dmin}'
        if w.max() < 1e-3 and lb1 > lb0: stronger += 1
    small = sum(w.max() < 1e-3 for _, w in cells)
    assert small == 0 or stronger >= small // 2, f'first-order bound stronger on only {stronger} of {small} small cells'


def test_bound_at_the_optimum_is_tight(binary):
    """on a tiny cell at the N = 5 optimum the first-order bound is within O(w) of D, the zeroth-order one is not better"""
    N = 5; qs = local_q('N5_best'); nq = 2 * N - 3; D = exact_discrepancy(points_float(qs, N)).value
    (lb0, _, lb1), = evaluate(binary, N, [(qs, np.full(nq, 1e-6))])
    assert lb1 <= D + 1e-12 and D - lb1 < 1e-4 and lb0 <= lb1 + 1e-12
