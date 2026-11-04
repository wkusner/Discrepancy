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
from tests.oracle import discrepancy_upper                   # noqa: E402  (test oracle only: floating point, robust)

flint.ctx.prec = 53
rng = np.random.default_rng(12345)


from tests.oracle import points_float                        # noqa: E402,F401  (re-exported for other tests)


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
            D = discrepancy_upper(points_float(y, N))
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
        r = run('proof.local', 'MUT_N5', '3e-4', '--out', str(tmp_path), '--level', 'full')
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
        r = run('proof.local', 'MUT2_N5', '3e-4', '--out', str(tmp_path), '--level', 'full')
        assert 'ALL KRAWCZYK-VERIFIED' not in r.stdout, r.stdout + r.stderr
    finally:
        os.remove(path)


# ---------------------------------------------------------------- redundant worst caps (audit 2026-11-04, finding 2)
def _local(name, level, out, hints=None):
    path = None
    if hints is not None:
        path = os.path.join(ROOT, 'proof', 'hints', f'{name}.json'); json.dump(hints, open(path, 'w'))
    try:
        r = run('proof.local', name, '1e-4', '--out', str(out), '--level', level)
        assert r.returncode == 0, r.stdout + r.stderr
        return json.load(open(os.path.join(str(out), f'localcert_{name}.json')))
    finally:
        if path: os.remove(path)


def test_extra_caps_are_refused_at_the_verified_level(tmp_path):
    """the N = 7 hints with an arbitrary cap added to extra_caps: verified=False at the level "verified" (before the
    fix the list was only echoed), and the honest D(q*) enclosure contains the D* enclosure"""
    hints = json.load(open(os.path.join(ROOT, 'proof/hints/N7_best.json')))
    bad = copy.deepcopy(hints); bad['name'] = 'MUT3_N7'; bad['images'] = bad['images'][:1]
    img = bad['images'][0]
    allc = [repr(c) for c in arith.all_caps(7)]
    new = next(c for c in allc if c not in img['caps'])
    img['extra_caps'] = [new]; img['extra_counts'] = [img['counts'][0]]
    lc = _local('MUT3_N7', 'verified', tmp_path, bad)
    im = lc['images'][0]
    assert lc['level'] == 'verified' and not lc['verified'] and not im['verified']
    assert any('extra caps' in w for w in im['not_verified_because'])
    lo, hi = im['D_qstar_enclosure']; dlo, dhi = im['D_enclosure']
    assert lo >= dlo and hi >= dhi


def test_verified_level_without_extra_caps_and_full_level(tmp_path):
    """N = 7 (no extra caps): verified at the level "verified", D(q*) enclosure = D* enclosure; N = 6 (one extra cap per
    image): refused at "verified", accepted at "full" (Step 5 closes the extra caps there)"""
    lc = _local('N7_best', 'verified', tmp_path / 'a')
    assert lc['verified'] and all(i['D_qstar_enclosure'] == i['D_enclosure'] for i in lc['images'])
    lc6 = _local('N6_best', 'verified', tmp_path / 'b')
    assert not lc6['verified'] and all(i['extra_caps'] and not i['verified'] for i in lc6['images'])
    lc6f = _local('N6_best', 'full', tmp_path / 'c')
    assert lc6f['verified'] and lc6f['level'] == 'full'
