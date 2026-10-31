"""proof.domain -- the fundamental domain F of Lemma lem:domain (TRUSTED CORE; PROOF.md Step 1).

F: x0 = north pole, x1 in the half-plane y = 0, x >= 0, (x0, x1) a closest pair, th_2 <= th_3 <= ...,
0 <= ph_2 <= pi (0-based labels; the papers use 1-based labels).  The initial box Q is
[0, pi] x [0, pi] x [0, pi] x ([0, pi] x [-pi, pi])^(N-3), widened outward (see csrc/rig.h).
A cell may be discarded when it provably misses F; this file decides that.
"""
import itertools
from flint import arb
from .arith import points_ad


def misses_domain(qc, w, N):
    """Rigorous: True if the cell qc +- w contains no point of capcert's fundamental domain
    (theta_i > theta_{i+1} throughout, or some pair closer than (x0, x1) throughout)."""
    nq = 2 * N - 3
    for i in range(2, N - 1):
        a, b = 2 * i - 3, 2 * (i + 1) - 3
        if arb(qc[a]) - arb(w[a]) > arb(qc[b]) + arb(w[b]):
            return True
    qbox = [arb(qc[k], w[k]) for k in range(nq)]
    P = points_ad(qbox, N)
    pts = [[x.v for x in p] for p in P]
    c01 = qbox[0].cos()                      # <y0, y1> = cos q0 = cos d(y0, y1)
    for i, j in itertools.combinations(range(N), 2):
        if (i, j) == (0, 1): continue
        g = pts[i][0] * pts[j][0] + pts[i][1] * pts[j][1] + pts[i][2] * pts[j][2]
        if g > c01:                          # <y_i,y_j> > <y_0,y_1> throughout: d(y_i,y_j) < d(y_0,y_1)
            return True
    return False
