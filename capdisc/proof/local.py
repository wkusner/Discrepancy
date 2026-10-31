"""proof.local -- local verification at each image of the minimizer (TRUSTED CORE; PROOF.md Step 4).

For each image k (from proof/hints/<NAME>.json: the parameter point, the square system of 2N-2 worst caps with
their counts, and redundant worst caps):
 1. Krawczyk test: the square system phi_b(q) - D = 0 (b in A, counts frozen) has exactly one solution
    (q*, D*) in a box Z of radius 1e-12; this encloses D*.
 2. On a larger box B (half-width r) around q*:
      - the counts of all worst caps are constant on B (interval sign tests), so D >= phi_b on B;
      - for every orthant s of directions v = Y - q* there are weights mu >= 0 with
        s_j (sum_b mu_b G^s_bj) > 0 for all j, where G^s_bj is the end of the interval gradient of phi_b
        over B that minimizes G.v on that orthant.  Then max_b phi_b(Y) > D* for Y in B, Y != q*.
 3. Every other cap of the capture list is below D* on Z, so D(q*) = D*.
usage: python -m proof.local NAME [r] [--out DIR]      writes DIR/localcert_<NAME>.json (default results/)
"""
import sys, os, json, itertools, time
import numpy as np
import flint
from flint import arb
from scipy.optimize import linprog
from .arith import points_ad, cap_terms, all_caps, parse_cap
from .cellbound import cap_data
from . import certio

flint.ctx.prec = 53          # double-precision balls (the default, set explicitly)
ROOT = certio.REPO
args = [a for a in sys.argv[1:] if not a.startswith('--')]
OUT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(ROOT, 'results')
if '--out' in sys.argv: args.remove(OUT)
name = args[0]; r_big = float(args[1]) if len(args) > 1 else 1e-4
hints_file = os.path.join(ROOT, 'proof', 'hints', f'{name}.json')
hints = json.load(open(hints_file)); N = hints['N']; nq = 2 * N - 3
t0 = time.time()


def phi_list(qballs, caps, counts):
    """values (AD) of phi_b = n_b/N - (1 - t_b)/2 for given caps and fixed counts."""
    P = points_ad(qballs, N); out = []
    for cap, n in zip(caps, counts):
        t, _ = cap_terms(P, cap)
        out.append(t * 0.5 + (arb(n) / N - arb(1) / 2))
    return out


def certify_image(q0, caps, counts, r_big, extra=(), extra_counts=()):
    report = {}
    # ---------- 1. Krawczyk on (q, D)
    m = np.append(q0, 0.0)
    vals = phi_list([arb(x) for x in q0], caps, counts)
    m[-1] = float(np.mean([float(v.v.mid()) for v in vals]))
    # Newton polish in floating point
    for _ in range(8):
        vals = phi_list([arb(x) for x in m[:-1]], caps, counts)
        F = np.array([float(v.v.mid()) - m[-1] for v in vals])
        J = np.zeros((len(caps), nq + 1))
        for i, v in enumerate(vals):
            for k, g in v.d.items(): J[i, k] = float(g.mid())
            J[i, -1] = -1
        m = m - np.linalg.solve(J, F)
    Yinv = np.linalg.inv(J)
    for rk in (1e-12, 1e-11, 1e-10, 1e-9):
        Xb = [arb(m[k], rk) for k in range(nq + 1)]
        vals_m = phi_list([arb(x) for x in m[:-1]], caps, counts)
        Fm = [v.v - arb(m[-1]) for v in vals_m]
        valsX = phi_list(Xb[:-1], caps, counts)
        JX = [[(v.d.get(k, arb(0))) for k in range(nq)] + [arb(-1)] for v in valsX]
        n = nq + 1; ok = True; K = []
        for i in range(n):
            acc = arb(m[i])
            for j in range(n): acc = acc - arb(Yinv[i, j]) * Fm[j]
            for j in range(n):
                cij = arb(1 if i == j else 0)
                for l in range(n): cij = cij - arb(Yinv[i, l]) * JX[l][j]
                acc = acc + cij * (Xb[j] - arb(m[j]))
            K.append(acc)
            # K(Z) inside the open box m +- rk: a rigorous upper bound of |acc - m_i| below rk (the ball Xb[i]
            # has exact midpoint m_i and radius >= rk, so this is conservative)
            if not ((acc - arb(m[i])).abs_upper() < arb(rk)):
                ok = False
        if ok: break
    report['krawczyk'] = ok
    if not ok: return report
    report['krawczyk_radius'] = rk
    Dstar = K[-1]
    report['D_enclosure'] = [float(Dstar.lower()), float(Dstar.upper())]
    report['q_center'] = m[:-1].tolist()
    # other caps below D* on the Krawczyk box (so D(q*) = D*)
    allc = all_caps(N); kb = [arb(m[k], rk) for k in range(nq)]
    datK = cap_data(m[:-1], np.full(nq, rk), N, allc)
    worst_other = -1.0
    act = {(c.I, c.side if len(c.I) == 3 else 0) for c in list(caps) + list(extra)}
    for c, dd in zip(allc, datK):
        if (c.I, c.side if len(c.I) == 3 else 0) in act: continue
        if dd is None: worst_other = float('inf'); continue     # cannot be bounded: verification fails
        worst_other = max(worst_other, float(dd[4].upper()))
    report['max_other_caps_on_K'] = worst_other
    report['others_below'] = bool(worst_other < float(Dstar.lower()) and float(arb(1) / N) < float(Dstar.lower()))
    if extra:   # redundant worst caps (forced by symmetry): report their enclosure on K
        datE = cap_data(m[:-1], np.full(nq, rk), N, list(extra))
        report['redundant_caps_on_K'] = [[float((dd[0]).lower()), float(dd[4].upper())] for dd in datE]
    # ---------- 2. sharpness on the big box
    w = np.full(nq, r_big)
    # only the square system enters the sharpness test: for these caps phi_b(q*) = D* exactly
    # (Krawczyk); redundant caps are not needed and their value at q* is not used.
    caps_all = list(caps); counts_all = list(counts)
    datB = cap_data(m[:-1], w, N, caps_all)
    # lower AND upper counts must equal the frozen count (no point may cross a worst-cap boundary)
    counts_ok = all(dd is not None and dd[3] == n and dd[5] == n for dd, n in zip(datB, counts_all))
    report['counts_frozen_on_box'] = counts_ok
    if not counts_ok: return report
    # gradient enclosures [mid - rad, mid + rad], rounded outward by one ulp (float subtraction)
    lo = np.nextafter(np.array([dd[1] - dd[2] for dd in datB]), -np.inf)
    hi = np.nextafter(np.array([dd[1] + dd[2] for dd in datB]), np.inf)
    nfail = 0
    for s in itertools.product((1, -1), repeat=nq):
        s = np.array(s)
        Gs = np.where(s > 0, lo, hi)                 # minimizes g.v on the orthant
        # find mu >= 0, sum mu = 1, maximizing min_j s_j (mu^T Gs)_j
        A = -(Gs * s).T                                   # constraints: -s_j (mu^T Gs)_j + z <= 0
        res = linprog(np.append(np.zeros(len(caps_all)), -1.0),
                      A_ub=np.hstack([A, np.ones((nq, 1))]), b_ub=np.zeros(nq),
                      A_eq=np.append(np.ones(len(caps_all)), 0.0)[None], b_eq=[1.0],
                      bounds=[(0, None)] * len(caps_all) + [(None, None)], method='highs')
        if res.status != 0 or -res.fun <= 0: nfail += 1; continue
        mu = np.maximum(res.x[:-1], 0.0)                   # the argument needs mu >= 0
        if mu.sum() <= 0: nfail += 1; continue
        # rigorous check in ball arithmetic (Gs are outward-rounded endpoints)
        for j in range(nq):
            acc = arb(0)
            for b in range(len(caps_all)): acc = acc + arb(float(mu[b])) * arb(float(Gs[b, j]))
            if not (acc * int(s[j]) > 0): nfail += 1; break
    report['orthants_failed'] = nfail
    report['box_radius'] = r_big
    report['verified'] = (nfail == 0) and report['others_below']
    return report



results = []
print(f"{name}: N={N}, {len(hints['images'])} image(s) in the fundamental domain")
for h in hints['images']:
    q0 = np.array([float.fromhex(x) for x in h['q']])
    caps = [parse_cap(c) for c in h['caps']]; counts = h['counts']
    extra = [parse_cap(c) for c in h['extra_caps']]; extra_counts = h['extra_counts']
    out = certify_image(q0, caps, counts, r_big, extra, extra_counts)
    out['image'] = h['image']; out['caps'] = h['caps']; out['counts'] = counts
    out['extra_caps'] = h['extra_caps']; out['extra_counts'] = extra_counts
    print(f"  image {h['image']}: " + ", ".join(f"{key}={out.get(key)}" for key in ('krawczyk', 'D_enclosure', 'others_below', 'counts_frozen_on_box', 'orthants_failed', 'verified')))
    results.append(out)
os.makedirs(OUT, exist_ok=True)
certio.write_json(os.path.join(OUT, f'localcert_{name}.json'),
    dict(name=name, N=N, box_radius=r_big, gauge='capcert coordinates (x0 pole, x1 in xz-plane x>=0)',
         verified=all(r.get('verified') for r in results), kernel='arb (python-flint)', method='Krawczyk test + orthant test',
         label='Krawczyk-verified (Arb)', images=results,
         stamp=certio.stamp('PROOF.md Step 4 (local verification)', inputs=[hints_file], t0=t0, prec=flint.ctx.prec)))
print("ALL KRAWCZYK-VERIFIED (Arb)" if all(r.get('verified') for r in results) else "NOT ALL VERIFIED")
