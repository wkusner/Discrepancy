"""Tests of the trusted core proof/ (REFACTOR_PLAN.md, Tests). Tests are not proofs; they catch the errors that
proofs about programs miss. Mutation tests check that each verification FAILS when its input is wrong.

    python -m pytest tests/test_proof_core.py -q
"""
import copy, json, os, subprocess, sys
import numpy as np
import pytest
import flint
from flint import arb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from proof import arith, cellbound, domain, certio                     # noqa: E402
from sphcap import discrepancy as exact_discrepancy          # noqa: E402  (test oracle only: floating point)

flint.ctx.prec = 53
rng = np.random.default_rng(12345)


def points_float(q, N):
    X = [[0.0, 0.0, 1.0], [np.sin(q[0]), 0.0, np.cos(q[0])]]
    for i in range(2, N):
        th, ph = q[2 * i - 3], q[2 * i - 2]
        X.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return np.array(X)


def local_q(name):
    lc = json.load(open(os.path.join(ROOT, 'certificates', f'localcert_{name}.json')))
    return np.array(lc['images'][0]['q_center'])


@pytest.mark.parametrize('N,name', [(5, 'N5_best'), (6, 'N6_best')])
def test_cell_bound_below_discrepancy(N, name):
    """the rigorous lower bound on a cell never exceeds the discrepancy of configurations inside the cell"""
    qs = local_q(name); nq = 2 * N - 3; caps = arith.all_caps(N)
    for _ in range(40):
        c = qs + rng.normal(0, 10 ** rng.uniform(-3, -0.5), nq)
        w = np.full(nq, 10 ** rng.uniform(-4, -1.5))
        lb, _ = cellbound.cell_lower_bound(c, w, N, 1.0, caps)
        for _ in range(5):
            y = c + rng.uniform(-1, 1, nq) * w
            D = exact_discrepancy(points_float(y, N)).value
            assert float(lb.lower()) <= D + 1e-12


def test_bisection_covers_parent():
    """the children's half-widths cover the parent despite the rounding of the new centres (proof/refine.py)"""
    for _ in range(5000):
        c = rng.uniform(-3.2, 3.2); w = 10 ** rng.uniform(-12, 0)
        flint.ctx.prec = 53
        h = w / 2; a = c - h; b = c + h; hj = arb(h); cj = arb(c)
        err = max(abs(arb(a) - (cj - hj)).abs_upper(), abs(arb(b) - (cj + hj)).abs_upper())
        r = certio.float_up((hj + arb(err)).upper())
        flint.ctx.prec = 400
        A, B, C, W, R = (arb(x) for x in (a, b, c, w, r))
        assert A - R <= C - W and A + R >= B - R and B + R >= C + W
    flint.ctx.prec = 53


def test_outward_rounding():
    for _ in range(2000):
        x = arb(rng.uniform(-2, 2)) / 3
        assert arb(certio.float_up(x.upper())) >= x.upper()
        assert arb(certio.float_down(x.lower())) <= x.lower()


def test_misses_domain_never_discards_a_domain_point():
    """a cell containing a point of the fundamental domain is never discarded"""
    N = 5; nq = 7; qs = local_q('N5_best'); tested = 0
    for _ in range(600):
        y = qs + rng.normal(0, 0.05, nq)
        X = points_float(y, N); G = X @ X.T
        d01 = np.arccos(np.clip(G[0, 1], -1, 1))
        dmin = min(np.arccos(np.clip(G[i, j], -1, 1)) for i in range(N) for j in range(i + 1, N) if (i, j) != (0, 1))
        in_F = d01 <= dmin and y[1] <= y[3] <= y[5] and 0 <= y[2] <= np.pi
        if not in_F: continue
        w = np.full(nq, 10 ** rng.uniform(-6, -2)); c = y + rng.uniform(-1, 1, nq) * w
        assert not domain.misses_domain(c, w, N); tested += 1
    assert tested >= 50, tested


def run(*args, cwd=ROOT):
    return subprocess.run([sys.executable, '-m', *args], cwd=cwd, capture_output=True, text=True)


def test_local_verification_fails_on_a_wrong_count(tmp_path):
    """mutation: one wrong worst-cap count must make the local verification fail"""
    hints = json.load(open(os.path.join(ROOT, 'proof/hints/N5_best.json')))
    bad = copy.deepcopy(hints); bad['name'] = 'MUT_N5'; bad['images'] = bad['images'][:1]
    bad['images'][0]['counts'][0] += 1
    path = os.path.join(ROOT, 'proof/hints/MUT_N5.json')
    json.dump(bad, open(path, 'w'))
    try:
        r = run('proof.local', 'MUT_N5', '3e-4', '--out', str(tmp_path))
        assert 'NOT ALL VERIFIED' in r.stdout, r.stdout + r.stderr
    finally:
        os.remove(path)


def test_local_verification_fails_without_a_worst_cap(tmp_path):
    """mutation: replacing a worst cap by another cap must make it fail"""
    hints = json.load(open(os.path.join(ROOT, 'proof/hints/N5_best.json')))
    bad = copy.deepcopy(hints); bad['name'] = 'MUT2_N5'; bad['images'] = bad['images'][:1]
    caps = bad['images'][0]['caps']
    caps[0] = caps[0][:-1] + ('-' if caps[0].endswith('+') else '+')         # the other side of the circle
    path = os.path.join(ROOT, 'proof/hints/MUT2_N5.json')
    json.dump(bad, open(path, 'w'))
    try:
        r = run('proof.local', 'MUT2_N5', '3e-4', '--out', str(tmp_path))
        assert 'ALL KRAWCZYK-VERIFIED' not in r.stdout, r.stdout + r.stderr
    finally:
        os.remove(path)
