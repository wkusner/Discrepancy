"""proof.cellbound -- rigorous lower bounds for the discrepancy on a cell (TRUSTED CORE; PROOF.md Step 3).

For a cell B = c + [-w, w] and a cap C defined by points I, the over-count satisfies, for all Y in B,

    f_C(Y) >= n_B/N - sigma(C(Y)) =: phi(Y),

where n_B counts the points that are in the closed cap C(Y) for every Y in B (points of I always;
others decided by an interval sign test).  By the mean-value theorem
    phi(Y) >= phi(c) + sum_j [ m_j v_j - r_j |v_j| ],   v = Y - c,  G_j = m_j +- r_j  contains  d phi/dq_j (B),
so phi(Y) >= a + m.v with a = phi(c) - sum_j r_j w_j.  For ANY weights lambda >= 0 summing to 1,

    D(Y) >= max_C phi_C(Y) >= sum_C lambda_C a_C - sum_j |sum_C lambda_C m_Cj| w_j     (*)

(Exp. Math. paper, eq. lpbound).  lambda is chosen by a floating-point LP; (*) is evaluated in ball arithmetic.
No inequality in this file is decided in floating point: every value that enters a bound is an Arb ball.
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


class CapData:
    """Enclosures for one cap over a cell c + [-w, w], all in exact ball arithmetic:
    a      lower bound for phi(c) - sum_j r_j w_j, so that phi(c + v) >= a + m.v on the cell;
    m, r   the gradient of phi over the cell lies in [m_j - r_j, m_j + r_j] (m_j, r_j exact Arb values);
    g      the gradient balls themselves (lower and upper ends are exact: g_j.lower(), g_j.upper());
    n      robust count (points in the closed cap throughout the cell), n_up upper count;
    phi_up upper bound for phi over the cell with the upper count;
    m_float the midpoints as doubles, for the LP that chooses weights (selection only)."""
    __slots__ = ('a', 'm', 'r', 'g', 'n', 'n_up', 'phi_up', 'm_float')


def cap_data(qc, w, N, caps):
    """For each cap a CapData, or None if the cap is degenerate somewhere on the cell.
    phi = n/N - (1 - t)/2.  Values at the center (point balls), gradients over the cell (box balls).
    qc, w: sequences of doubles (exact as balls)."""
    nq = 2 * N - 3
    qcb = [arb(x) for x in qc]
    qbox = [arb(qc[k], w[k]) for k in range(nq)]
    wb = [arb(x) for x in w]
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
        d = CapData()
        d.n, d.n_up = n, n_up
        d.g = [tb.d.get(k, ZERO) / 2 for k in range(nq)]         # d phi / dq = + dt/dq / 2, over the cell
        d.m = [arb(gk.mid()) for gk in d.g]                      # exact midpoints
        d.r = [arb(gk.rad()) for gk in d.g]                      # exact radii
        d.m_float = np.array([float(gk.mid()) for gk in d.g])
        phi_c = arb(n) / N - (1 - tc.v) / 2
        d.a = phi_c - sum((rk * wk for rk, wk in zip(d.r, wb)), ZERO)
        phi_c_up = arb(n_up) / N - (1 - tc.v) / 2
        d.phi_up = phi_c_up + sum(((abs(mk) + rk) * wk for mk, rk, wk in zip(d.m, d.r, wb)), ZERO)
        out.append(d)
    return out


def cell_lower_bound(qc, w, N, T, caps=None, use_lp=True):
    """Rigorous lower bound (an Arb ball whose lower end is the bound) for D over the cell; (bound, info)."""
    from scipy.optimize import linprog
    caps = caps or all_caps(N)
    data = cap_data(qc, w, N, caps)
    rows = [d for d in data if d is not None]
    wb = [arb(x) for x in w]
    best = ONE / N
    # single caps: phi(c + v) >= a + m.v >= a - sum_j |m_j| w_j
    for d in rows:
        lb1 = d.a - sum((abs(mk) * wk for mk, wk in zip(d.m, wb)), ZERO)
        if lb1 > best: best = lb1
    if best >= T or not use_lp:
        return best, dict(method='single')
    # weights from an LP (selection only): min s s.t. s >= a_b + m_b.v, -w <= v <= w
    A = np.array([np.append(d.m_float, -1.0) for d in rows])
    av = np.array([float(d.a.mid()) for d in rows])
    nq = len(w)
    c = np.zeros(nq + 1); c[-1] = 1
    res = linprog(c, A_ub=A, b_ub=-av, bounds=[(-wk, wk) for wk in w] + [(None, None)], method='highs')
    if res.status != 0:
        return best, dict(method='lp-failed')
    lam = np.maximum(-res.ineqlin.marginals, 0.0)
    if lam.sum() <= 0:
        return best, dict(method='lp-zero')
    # rigorous evaluation of (*) with weights L_C / S, L_C = lam_C as exact balls, S = sum L_C
    L = [arb(float(x)) for x in lam]; S = sum(L, ZERO)
    acc = sum((l * d.a for l, d in zip(L, rows)), ZERO)
    pen = ZERO
    for j in range(nq):
        gj = sum((l * d.m[j] for l, d in zip(L, rows)), ZERO)       # sum_C L_C m_Cj (a ball)
        pen = pen + arb(gj.abs_upper()) * wb[j]
    lb = (acc - pen) / S
    if lb > best: best = lb
    return best, dict(method='lp', lp_value=res.fun)
