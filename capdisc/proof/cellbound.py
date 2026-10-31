"""proof.cellbound -- rigorous lower bounds for the discrepancy on a cell (TRUSTED CORE; PROOF.md Step 3).

For a cell B = c + [-w, w] and a cap C defined by points I, the over-count satisfies, for all Y in B,

    f_C(Y) >= n_B/N - sigma(C(Y)) =: phi(Y),

where n_B counts the points that are in the closed cap C(Y) for every Y in B (points of I always;
others decided by an interval sign test).  By the mean-value theorem
    phi(Y) >= phi(c) + sum_j [ m_j v_j - r_j |v_j| ],   v = Y - c,  G_j = m_j +- r_j  contains  d phi/dq_j (B),
so phi(Y) >= a + m.v with a = phi(c) - sum_j r_j w_j.  For ANY weights lambda >= 0 summing to 1,

    D(Y) >= max_C phi_C(Y) >= sum_C lambda_C a_C - sum_j |sum_C lambda_C m_Cj| w_j     (*)

(Exp. Math. paper, eq. lpbound).  lambda is chosen by a floating-point LP; (*) is evaluated in ball arithmetic.
"""
import numpy as np
from flint import arb
from .arith import ZERO, ONE, points_ad, cap_terms, all_caps


def _mv_interval(center_val: arb, grad_box: dict, w):
    """mean-value enclosure over the box: center value + sum_j G_j * [-w_j, w_j]."""
    out = center_val
    for k, g in grad_box.items():
        out = out + g * arb(0, w[k])
    return out


def cap_data(qc, w, N, caps):
    """For each cap: (a_lower_center_value, mid_grad (np), rad_grad (np), robust_count) or None.
    phi = n/N - (1 - t)/2.  Uses center evaluation (point balls) for values and box evaluation
    for gradients."""
    nq = 2 * N - 3
    qcb = [arb(x) for x in qc]
    qbox = [arb(qc[k], w[k]) for k in range(nq)]
    Pc = points_ad(qcb, N); Pb = points_ad(qbox, N)
    out = []
    for cap in caps:
        try:
            tc, mc = cap_terms(Pc, cap)
            tb, mb = cap_terms(Pb, cap)
        except Exception:
            out.append(None); continue
        if not (tc.v.is_finite() and tb.v.is_finite()):
            out.append(None); continue
        # robust count: points of I always in; others need a certain sign over the box
        n = len(cap.I); n_up = n
        for j in range(N):
            if j in cap.I: continue
            mj = _mv_interval(mc(Pc[j]).v, mb(Pb[j]).d, w)
            if mj > 0: n += 1; n_up += 1
            elif mj < 0: pass
            else:        # undecided: outside for the lower bound, inside for the upper bound
                n_up += 1
        phi_c = arb(n) / N - (1 - tc.v) / 2
        mid = np.zeros(nq); rad = np.zeros(nq)
        for k, g in tb.d.items():
            gg = g / 2               # d phi / dq = + dt/dq / 2
            mid[k] = float(gg.mid()); rad[k] = (float(gg.rad()) + 2.3e-16 * abs(mid[k]) + 1e-300) * (1 + 1e-12)
        a = phi_c - arb(float(np.dot(rad, w)) * (1 + 1e-12) + 1e-300)
        # account for rounding of mid to float: absorb into rad (done above)
        # upper enclosure of phi over the box (count upper bound n_up), for 'below D*' checks
        phi_up = (arb(n_up) / N - (1 - tc.v) / 2) + arb(float(np.dot(np.abs(mid) + rad, w)) * (1 + 1e-12) + 1e-300)
        out.append((a, mid, rad, n, phi_up, n_up))
    return out


def cell_lower_bound(qc, w, N, T, caps=None, use_lp=True):
    """Rigorous lower bound for D over the cell (arb).  Returns (lb_float_lower, info)."""
    from scipy.optimize import linprog
    caps = caps or all_caps(N)
    data = cap_data(qc, w, N, caps)
    rows = [(i, d) for i, d in enumerate(data) if d is not None]
    best = arb(1) / N
    # single caps
    for i, (a, mid, rad, n, _pu, _nu) in rows:
        lb1 = a - arb(float(np.dot(np.abs(mid), w)) * (1 + 1e-12) + 1e-300)
        if lb1 > best: best = lb1
    if best >= T or not use_lp:
        return best, dict(method='single')
    # LP: min s s.t. s >= a_b + m_b.v, -w <= v <= w
    A = np.array([np.append(d[1], -1.0) for _, d in rows])
    av = np.array([float(d[0].mid()) for _, d in rows])
    nq = len(w)
    c = np.zeros(nq + 1); c[-1] = 1
    res = linprog(c, A_ub=A, b_ub=-av, bounds=[(-wk, wk) for wk in w] + [(None, None)], method='highs')
    if res.status != 0:
        return best, dict(method='lp-failed')
    lam = np.maximum(-res.ineqlin.marginals, 0.0)
    if lam.sum() <= 0:
        return best, dict(method='lp-zero')
    lam = lam / lam.sum()
    # rigorous evaluation of (*) with these lambda (exact rationals not needed: any lam>=0, renormalized in arb)
    L = [arb(float(x)) for x in lam]; S = sum(L, ZERO)
    acc = ZERO
    for (i, d), l in zip(rows, L):
        acc = acc + l * d[0]
    gsum = np.zeros(nq)
    for (i, d), l in zip(rows, lam):
        gsum += l * d[1]
    # bound |sum lam m| rigorously: the float sum has rounding error <= 1e-15 * sum|...|
    err = 1e-12 * sum(abs(l) * np.abs(d[1]) for (_, d), l in zip(rows, lam))
    pen = arb(float(np.dot(np.abs(gsum) + err, w)) * (1 + 1e-12))
    lb = (acc - pen) / S
    if lb > best: best = lb
    return best, dict(method='lp', lp_value=res.fun)


