"""Local certificate around a sharp minimizer (rigorous, ball arithmetic).

For a configuration in critical_configurations.json with |A| = 2N-2 worst caps:
 1. Krawczyk test: the square system phi_b(q) - D = 0 (b in A, counts frozen) has exactly one
    solution (q*, D*) in a small box K; this encloses D*.
 2. On a larger box B (half-width r) around q*:
      - the counts of all worst caps are constant on B (interval sign tests), so D >= phi_b on B;
      - for every orthant s of directions v = Y - q*, there are weights mu >= 0 with
        s_j (sum_b mu_b G^s_bj) > 0 for all j, where G^s_bj is the end of the interval
        gradient of phi_b over B that minimizes G.v on that orthant.  Then
        max_b phi_b(Y) >= D* + (sum mu G^s).v > D* for Y in B, Y != q*.
    Hence q* is the unique minimizer of D on B and D(q*) = D* (the other caps are checked to be
    below D* on K).
Output: results/localcert_<NAME>.json with the boxes (in gauge coordinates) and the D* enclosure.

usage: python jobs/local_cert.py NAME [r]
"""
import sys, os, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
from flint import arb
from scipy.optimize import linprog
from sphcap import balance as B, highprec as H, certify as Cf

name = sys.argv[1]; r_big = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
d = {c['name']: c for c in json.load(open(os.path.join(HERE, '..', 'sphcap/data/critical_configurations.json')))['classes']}
X = np.array(d[name]['points']); N = len(X); nq = 2 * N - 3


def phi_list(qballs, caps, counts):
    """values (AD) of phi_b = n_b/N - (1 - t_b)/2 for given caps and fixed counts."""
    P = Cf.points_ad(qballs, N); out = []
    for cap, n in zip(caps, counts):
        t, _ = Cf.cap_terms(P, cap)
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
            if not (acc.lower() > Xb[i].lower() and acc.upper() < Xb[i].upper()):
                ok = False
        if ok: break
    report['krawczyk'] = ok
    if not ok: return report
    report['krawczyk_radius'] = rk
    Dstar = K[-1]
    report['D_enclosure'] = [float(Dstar.lower()), float(Dstar.upper())]
    report['q_center'] = m[:-1].tolist()
    # other caps below D* on the Krawczyk box (so D(q*) = D*)
    allc = Cf.all_caps(N); kb = [arb(m[k], rk) for k in range(nq)]
    datK = Cf.cap_data(m[:-1], np.full(nq, rk), N, allc)
    worst_other = -1.0
    act = {(c.I, c.side if len(c.I) == 3 else 0) for c in list(caps) + list(extra)}
    for c, dd in zip(allc, datK):
        if (c.I, c.side if len(c.I) == 3 else 0) in act: continue
        if dd is None: continue
        worst_other = max(worst_other, float(dd[4].upper()))
    report['max_other_caps_on_K'] = worst_other
    report['others_below'] = bool(worst_other < float(Dstar.lower()) and float(arb(1) / N) < float(Dstar.lower()))
    if extra:   # redundant worst caps (forced by symmetry): report their enclosure on K
        datE = Cf.cap_data(m[:-1], np.full(nq, rk), N, list(extra))
        report['redundant_caps_on_K'] = [[float((dd[0]).lower()), float(dd[4].upper())] for dd in datE]
    # ---------- 2. sharpness on the big box
    w = np.full(nq, r_big)
    caps_all = list(caps) + list(extra); counts_all = list(counts) + list(extra_counts)
    datB = Cf.cap_data(m[:-1], w, N, caps_all)
    counts_ok = all(dd is not None and dd[3] == n for dd, n in zip(datB, counts_all))
    report['counts_frozen_on_box'] = counts_ok
    if not counts_ok: return report
    lo = np.array([dd[1] - dd[2] for dd in datB]); hi = np.array([dd[1] + dd[2] for dd in datB])
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
        mu = res.x[:-1]
        # rigorous check in ball arithmetic (Gs already outward-rounded endpoints)
        for j in range(nq):
            acc = arb(0)
            for b in range(len(caps_all)): acc = acc + arb(float(mu[b])) * arb(float(Gs[b, j]))
            if not (acc * int(s[j]) > 0): nfail += 1; break
    report['orthants_failed'] = nfail
    report['box_radius'] = r_big
    report['certified'] = (nfail == 0) and report['others_below']
    return report


def worst_caps(Y):
    q0 = np.array(H.to_q(Y), float); rep = B.balance_report(Y); caps, counts = [], []
    for br in rep.active:
        I = tuple(sorted(br.I)); nC = br.n_cl if br.kind == 'up' else N - br.n_op
        if len(I) == 2:
            caps.append(Cf.CapSpec(I)); counts.append(nC); continue
        best = None
        for side in (1, -1):
            cs = Cf.CapSpec(I, side)
            dat = Cf.cap_data(q0, np.full(nq, 1e-13), N, [cs])[0]
            if dat is None or dat[3] != nC: continue
            val = float(dat[0].mid())
            if best is None or abs(val - rep.D) < abs(best[1] - rep.D): best = (cs, val)
        assert best is not None and abs(best[1] - rep.D) < 1e-9, (I, best)
        caps.append(best[0]); counts.append(nC)
    return q0, caps, counts


images = Cf.domain_images(X)
print(f"{name}: N={N}, {len(images)} image(s) in the fundamental domain")
results = []
for k, (Yi, qi) in enumerate(images):
    q0, caps, counts = worst_caps(Yi); extra, extra_counts = [], []
    assert np.allclose(q0, qi, atol=1e-9), (q0, qi)
    if len(caps) != 2 * N - 2:
        # symmetric configurations have redundant worst caps: keep an independent square subset
        J = []
        for cs in caps:
            dat = Cf.cap_data(q0, np.full(nq, 1e-13), N, [cs])[0]
            J.append(np.append(dat[1], -1.0))
        J = np.array(J)
        import scipy.linalg as sl
        _, _, piv = sl.qr(J.T, pivoting=True)
        keep = sorted(piv[:2 * N - 2])
        extra = [caps[i] for i in range(len(caps)) if i not in keep]
        extra_counts = [counts[i] for i in range(len(caps)) if i not in keep]
        caps = [caps[i] for i in keep]; counts = [counts[i] for i in keep]
        print(f"  image {k}: {len(caps) + len(extra)} worst caps, using an independent subset of {len(caps)}")
    out = certify_image(q0, caps, counts, r_big, extra, extra_counts)
    out['image'] = k; out['caps'] = [repr(c) for c in caps]; out['counts'] = counts
    print(f"  image {k}: " + ", ".join(f"{key}={out.get(key)}" for key in ('krawczyk', 'D_enclosure', 'others_below', 'counts_frozen_on_box', 'orthants_failed', 'certified')))
    results.append(out)
os.makedirs('results', exist_ok=True)
json.dump(dict(name=name, N=N, box_radius=r_big, gauge='capcert coordinates (x0 pole, x1 in xz-plane x>=0)',
               certified=all(r.get('certified') for r in results), images=results),
          open(f'results/localcert_{name}.json', 'w'), indent=1)
print("ALL CERTIFIED" if all(r.get('certified') for r in results) else "NOT ALL CERTIFIED")
