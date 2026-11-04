"""Tests of the interval bounds in the C mesh (csrc/rig.h zeroth order, csrc/fo.h first order). Tests are not
proofs; they catch errors that a proof about the program would miss.

The binary is the one build.sh produces (the flags of the proof run). The bounds printed in CAPCERT_EVAL mode must
not exceed the discrepancy at sampled configurations of the cell, measured by a robust floating-point upper bound
(tests/oracle.py; sphcap.discrepancy can be wrong on nearly coincident points), and a cell flagged "outside the
fundamental domain" must not contain a sampled point of it. Besides random cells near the N = 5, 6 optima, the
cells are deliberately degenerate (the kinds of the audit of 2026-11-04): anywhere in the root box with large
widths, all points near a pole, all points within 1e-7 of each other, straddling theta = 0 and phi = +-pi, widths
exactly 0, the root box and its halves; and cells containing coincident points or an exactly collinear triple
(three points on a great circle), where the first-order bound's gradient enclosures overflow (finding 7).

    python -m pytest tests/test_capcert_fo.py -q
"""
import os, shutil, subprocess, sys, tempfile
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.oracle import discrepancy_upper, points_float      # noqa: E402  (test oracle only: floating point)
from tests.test_proof_core import local_q                     # noqa: E402
from tests.buildbin import binary as build_binary, evaluate   # noqa: E402

rng = np.random.default_rng(2026)
PI = np.pi


@pytest.fixture(scope='module')
def binary():
    return build_binary('capcert_rig')


def D_up(y, N):
    return discrepancy_upper(points_float(y, N))


@pytest.mark.parametrize('N,name', [(5, 'N5_best'), (6, 'N6_best')])
def test_bounds_below_discrepancy(binary, N, name):
    qs = local_q(name); nq = 2 * N - 3; cells = []
    for _ in range(60):
        c = qs + rng.normal(0, 10 ** rng.uniform(-3, -0.7), nq)
        w = np.full(nq, 10 ** rng.uniform(-5, -1.5)) * rng.uniform(0.5, 1.5, nq)
        cells.append((c, w))
    res, _ = evaluate(binary, N, cells)
    assert len(res) == len(cells)
    stronger = 0
    for (c, w), (lb0, sym, lb1) in zip(cells, res):
        Dmin = min(D_up(c + rng.uniform(-1, 1, nq) * w, N) for _ in range(8))
        Dmin = min(Dmin, D_up(c, N))
        assert lb0 <= Dmin + 1e-12, f'zeroth-order bound {lb0} above D = {Dmin}'
        assert lb1 <= Dmin + 1e-12, f'first-order bound {lb1} above D = {Dmin}'
        if w.max() < 1e-3 and lb1 > lb0: stronger += 1
    small = sum(w.max() < 1e-3 for _, w in cells)
    assert small == 0 or stronger >= small // 2, f'first-order bound stronger on only {stronger} of {small} small cells'


def test_bound_at_the_optimum_is_tight(binary):
    """on a tiny cell at the N = 5 optimum the first-order bound is within O(w) of D, the zeroth-order one is not better"""
    N = 5; qs = local_q('N5_best'); nq = 2 * N - 3; D = D_up(qs, N)
    ((lb0, _, lb1),), _ = evaluate(binary, N, [(qs, np.full(nq, 1e-6))])
    assert lb1 <= D + 1e-12 and D - lb1 < 1e-4 and lb0 <= lb1 + 1e-12


# ---------------------------------------------------------------- degenerate cells
ROOT_C = lambda N: np.array([PI / 2] + sum(([PI / 2, PI / 2] if i == 2 else [PI / 2, 0.0] for i in range(2, N)), []))
ROOT_W = lambda N: np.array([PI / 2] + sum(([PI / 2, PI / 2] if i == 2 else [PI / 2, PI] for i in range(2, N)), []))


def in_F(y, N):
    """y is in the fundamental domain (capcert coordinates): x0, x1 a closest pair, theta_2 <= ... <= theta_{N-1},
    phi_2 in [0, pi], all theta in [0, pi], all phi in [-pi, pi]"""
    th = [y[0]] + [y[2 * i - 3] for i in range(2, N)]; ph = [y[2 * i - 2] for i in range(2, N)]
    if not all(0 <= t <= PI for t in th) or not all(-PI <= p <= PI for p in ph) or not 0 <= ph[0] <= PI: return False
    if any(th[i] > th[i + 1] for i in range(1, len(th) - 1)): return False
    X = points_float(y, N); G = np.clip(X @ X.T, -1, 1); d = np.arccos(G)
    return all(d[0, 1] <= d[i, j] for i in range(N) for j in range(i + 1, N) if (i, j) != (0, 1))


def degenerate_cells(N, k):
    nq = 2 * N - 3; rc, rw = ROOT_C(N), ROOT_W(N); out = []
    for _ in range(k):          # anywhere in the root box, widths up to 1
        out.append((rc + rng.uniform(-1, 1, nq) * rw, rng.uniform(0, 1, nq) * 10 ** rng.uniform(-6, 0)))
    for _ in range(k):          # every theta within 1e-3 of a pole, phi-widths up to 3.2
        c = rng.uniform(-PI, PI, nq); w = rng.uniform(0, 1e-3, nq)
        pole = rng.choice([0.0, PI])
        c[0] = pole + rng.uniform(-1e-3, 1e-3)
        for i in range(2, N): c[2 * i - 3] = pole + rng.uniform(-1e-3, 1e-3); w[2 * i - 2] = rng.uniform(0, 3.2)
        out.append((c, w))
    for _ in range(k):          # all free points within 1e-7 of each other (and of x1)
        base = rng.uniform(0.2, PI - 0.2, 2); c = np.zeros(nq); c[0] = base[0]
        for i in range(2, N): c[2 * i - 3] = base[0] + rng.normal(0, 1e-7); c[2 * i - 2] = rng.normal(0, 1e-7)
        out.append((c, np.full(nq, 10 ** rng.uniform(-12, -7))))
    for _ in range(k):          # straddling theta = 0 and phi = +-pi
        c = rng.uniform(-PI, PI, nq); w = rng.uniform(1e-4, 0.3, nq)
        for i in range(2, N): c[2 * i - 3] = rng.uniform(-0.05, 0.05); c[2 * i - 2] = PI * rng.choice([-1, 1]) + rng.uniform(-0.05, 0.05)
        out.append((c, w))
    for _ in range(k):          # coordinates of width exactly 0
        c = rc + rng.uniform(-1, 1, nq) * rw; w = rng.uniform(0, 0.1, nq); w[rng.uniform(size=nq) < 0.5] = 0.0
        out.append((c, w))
    out.append((rc, rw))        # the whole root box and its halves
    for j in range(nq):
        for sgn in (-1, 1):
            c, w = rc.copy(), rw.copy(); w[j] /= 2; c[j] += sgn * w[j]; out.append((c, w))
    return out


def samples(c, w, k):
    nq = len(c)
    ys = [c] + [c + rng.uniform(-1, 1, nq) * w for _ in range(k)]
    ys += [c + rng.choice([-1, 1], nq) * w for _ in range(2)]          # corners
    return ys


@pytest.mark.parametrize('N', [4, 5, 6, 7])
def test_degenerate_cells(binary, N):
    cells = degenerate_cells(N, 12)
    res, _ = evaluate(binary, N, cells)
    assert len(res) == len(cells)
    flagged = 0
    for (c, w), (lb0, sym, lb1) in zip(cells, res):
        assert not np.isnan(lb0) and not np.isnan(lb1)
        ys = samples(c, w, 5)
        Dmin = min(D_up(y, N) for y in ys)
        assert lb0 <= Dmin + 1e-12, f'N={N}: zeroth-order bound {lb0} above D <= {Dmin} on cell {c.tolist()} +- {w.tolist()}'
        assert lb1 <= Dmin + 1e-12, f'N={N}: first-order bound {lb1} above D <= {Dmin} on cell {c.tolist()} +- {w.tolist()}'
        if sym:
            flagged += 1
            assert not any(in_F(y, N) for y in ys), f'N={N}: cell flagged outside F contains a point of F'
    assert flagged > 0


def collinear_cells(N, rng_):
    """cells whose configurations contain coincident points or an exactly collinear triple (three points on a great
    circle): x2 at the pole (= x0), x2 = x1, x2 within 1e-160 / 1e-300 / 5e-324 of the pole, x2 and x3 at the pole,
    x0, x1, x2 on the great circle y = 0 (phi_2 = 0 or pi)"""
    nq = 2 * N - 3; out = []
    for _ in range(4):
        q = ROOT_C(N) + rng_.uniform(-1, 1, nq) * ROOT_W(N) * 0.9
        for th2, ph2 in ((0.0, None), (q[0], 0.0), (1e-160, None), (1e-300, None), (5e-324, None), (q[1], 0.0), (q[1], PI)):
            y = q.copy(); y[1] = th2
            if ph2 is not None: y[2] = ph2
            for w in (0.0, 1e-300, 1e-12, 1e-6, 1e-3):
                out.append((y.copy(), np.full(nq, w)))
        y = q.copy(); y[1] = 0.0; y[3] = 0.0                     # x2 = x3 = x0
        out.append((y, np.zeros(nq))); out.append((y.copy(), np.full(nq, 1e-9)))
    return out


@pytest.mark.parametrize('N', [5, 7])
def test_first_order_bound_on_collinear_and_coincident_triples(binary, N):
    """finding 7: the first-order bound is finite or the single-cap value, never above D, and deterministic, on cells
    where the triple-cap normals vanish or the gradient enclosures overflow"""
    cells = collinear_cells(N, np.random.default_rng(7 + N))
    res1, _ = evaluate(binary, N, cells)
    res2, _ = evaluate(binary, N, cells)
    assert res1 == res2                                   # no dependence on uninitialised memory
    for (c, w), (lb0, sym, lb1) in zip(cells, res1):
        assert not np.isnan(lb1) and lb1 >= -1.0
        Dmin = min(D_up(y, N) for y in samples(c, w, 3))
        assert lb0 <= Dmin + 1e-12 and lb1 <= Dmin + 1e-12, (c.tolist(), w.tolist(), lb0, lb1, Dmin)


@pytest.mark.skipif(shutil.which('valgrind') is None, reason='valgrind not installed')
def test_first_order_bound_reads_no_uninitialised_memory(tmp_path):
    """finding 7 under valgrind (memcheck): no uninitialised value reaches a branch or the output. The binary is built
    with the proof's -D and fp flags but -O1 -g, without -march=native (valgrind does not know every instruction)."""
    exe = str(tmp_path / 'capcert_rig_vg')
    subprocess.run(['gcc', '-O1', '-g', '-Wall', '-Wno-unused-function', '-DRIGOROUS', '-ffp-contract=off', '-fno-fast-math',
                    '-o', exe, os.path.join(ROOT, 'csrc', 'capcert.c'), '-lm'], check=True, capture_output=True)
    cells = collinear_cells(5, np.random.default_rng(12))[:40]
    res, out = evaluate(exe, 5, cells, prefix=('valgrind', '-q', '--error-exitcode=9', '--track-origins=yes'))
    assert len(res) == len(cells)
    assert 'uninitialised' not in out.stderr, out.stderr[:3000]
