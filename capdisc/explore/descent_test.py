"""explore/descent_test.py -- PROTOTYPE of the no-minimizer test, measured against the current refinement.

Test on a cell B = c +- w (PROOF.md, proposed lemma): let LB be a rigorous lower bound for D on B and S the caps
that can be worst somewhere on B (upper value, with the upper count, >= LB). If some direction v has
grad(t_C) . v < 0 throughout B for every C in S (and LB > 1/N, and no cap of S is degenerate on B), then no
configuration in B is a local minimizer of D: at each y the worst caps strictly decrease along v, the others stay
below for small t, and closed-cap counts can only drop. The direction is a CHOICE (floating-point LP: maximize
the worst decrease); only the inequalities are decided in Arb.

usage: python explore/descent_test.py N SURVIVOR_DIR [SAMPLE]
"""
import glob, json, sys, time
import numpy as np
sys.path.insert(0, '.')
import flint
from flint import arb
from scipy.optimize import linprog
from proof.arith import all_caps
from proof.cellbound import cap_data, cell_lower_bound
from proof.domain import misses_domain

flint.ctx.prec = 53


def no_minimizer(c, w, N, caps):
    data = cap_data(c, w, N, caps)
    wb = [arb(x) for x in w]
    LB = arb(1) / N
    for d in data:
        if d is None: continue
        lb1 = d.a - sum((abs(m) * x for m, x in zip(d.m, wb)), arb(0))
        if lb1 > LB: LB = lb1
    if not (LB > arb(1) / N): return False, 'LB<=1/N'
    S = []
    for d in data:
        if d is None: return False, 'degenerate cap'          # conservative: cannot exclude it from S
        if not (d.phi_up < LB): S.append(d)
    if not S: return False, 'empty S'
    G = np.array([d.m_float for d in S]); nq = len(w)
    # direction: maximize s subject to G v <= -s, |v|_inf <= 1
    res = linprog(np.append(np.zeros(nq), -1.0), A_ub=np.hstack([G, np.ones((len(S), 1))]), b_ub=np.zeros(len(S)),
                  bounds=[(-1, 1)] * nq + [(None, None)], method='highs')
    if res.status != 0 or -res.fun <= 0: return False, 'balanced (no common descent direction)'
    v = [arb(float(x)) for x in res.x[:nq]]
    for d in S:
        if not (sum((g * x for g, x in zip(d.g, v)), arb(0)) < 0): return False, 'not uniform on the cell'
    return True, f'{len(S)} caps'


def process(c0, w0, N, T, caps, use_descent, minw=1e-7):
    stack = [(c0, w0)]; n = 0; stats = dict(bound=0, descent=0, sym=0, fail=0)
    while stack:
        c, w = stack.pop(); n += 1
        if misses_domain(c, w, N): stats['sym'] += 1; continue
        lb, _ = cell_lower_bound(c, w, N, T, caps)
        if lb >= T: stats['bound'] += 1; continue
        if use_descent and no_minimizer(c, w, N, caps)[0]: stats['descent'] += 1; continue
        if w.max() < minw: stats['fail'] += 1; continue
        s = w.copy()
        for i in range(2, N): s[2 * i - 2] = w[2 * i - 2] * (abs(np.sin(c[2 * i - 3])) + 1e-3)
        j = int(np.argmax(s)); h = w.copy(); h[j] = w[j] / 2
        a = c.copy(); a[j] -= h[j]; b = c.copy(); b[j] += h[j]; h[j] = np.nextafter(h[j] * (1 + 1e-15), np.inf)
        stack += [(a, h), (b, h)]
    return n, stats


if __name__ == '__main__':
    N, DIR = int(sys.argv[1]), sys.argv[2]; SAMPLE = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    nq = 2 * N - 3; caps = all_caps(N)
    stamp = dict(kv.split('=', 1) for kv in open(f'{DIR}/mesh.done').read().split()); T = float.fromhex(stamp['T'])
    files = sorted(f for f in glob.glob(f'{DIR}/mesh.surv.*') if f.rsplit('.', 1)[1].isdigit())
    arr = np.concatenate([np.fromfile(f, dtype=np.float64) for f in files]).reshape(-1, 2 * nq + 1)
    idx = np.random.default_rng(0).choice(len(arr), SAMPLE, replace=False)
    direct = 0; reasons = {}
    for k in idx:
        ok, why = no_minimizer(arr[k, :nq], arr[k, nq:2 * nq], N, caps)
        direct += ok; reasons[why if not ok else 'closed'] = reasons.get(why if not ok else 'closed', 0) + 1
    print(f'N={N}: {SAMPLE} survivor cells; closed directly by the descent test: {direct} ({100*direct/SAMPLE:.0f}%)')
    print('  reasons:', {k: v for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])})
    for use in (False, True):
        t0 = time.time(); tot = 0; agg = dict(bound=0, descent=0, sym=0, fail=0)
        for k in idx[:min(SAMPLE, 120)]:
            n, st = process(arr[k, :nq].copy(), arr[k, nq:2 * nq].copy(), N, T, caps, use)
            tot += n; agg = {x: agg[x] + st[x] for x in agg}
        print(f"  refinement {'with' if use else 'without'} the descent test: {tot} cells for {min(SAMPLE,120)} survivors, {agg}, {time.time()-t0:.0f}s")
