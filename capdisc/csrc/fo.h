/* fo.h -- first-order (LP-weighted) lower bound for D on a cell, in the IEEE interval arithmetic of ia.h
 * (rigorous mode only; PROOF.md Step 3, Exp. Math. eq. lpbound).  Included by capcert.c after rig.h.
 *
 * For a cap C with defining set I (a pair: the smallest cap; a triple: one side of its circle), level t_C(q), and a
 * robust count n_C (points in the closed cap C(Y) for every Y in the cell), the over-count satisfies on the cell
 *     f_C(Y) >= phi_C(Y) = n_C/N - (1 - t_C(Y))/2,
 * and by the mean-value theorem, with an interval enclosure [m_k - r_k, m_k + r_k] of d phi_C / d q_k over the cell,
 *     phi_C(c + v) >= a_C + m_C . v,   a_C = phi_C(c) - sum_k r_k h_k,   |v_k| <= h_k.
 * For any weights lambda_C >= 0 (not all 0):   D(Y) >= (sum lambda_C a_C - sum_k |sum lambda_C m_Ck| h_k) / sum lambda_C.
 * The weights are a CHOICE (exponentiated-gradient ascent in floating point); the bound is evaluated in intervals.
 * A cap whose level is not well defined somewhere on the cell (|S| or |n| may vanish) is skipped: any subset of
 * caps gives a valid bound.
 *
 * Gradients.  Pair cap: t = |S|/2, S = x_a + x_b, dt/dx_a = dt/dx_b = S/(2|S|).
 * Triple cap (side s): n = (b-a)x(c-a), det = a.(b x c) = n.a, t = s det/|n|, and
 *     dt/da = s[(b x c)/|n| - det ((b - c) x n)/|n|^3],   dt/db = s[(c x a)/|n| - det ((c - a) x n)/|n|^3],
 *     dt/dc = s[(a x b)/|n| - det ((a - b) x n)/|n|^3].
 * Point derivatives: x = (st cp, st sp, ct): dx/dth = (ct cp, ct sp, -st), dx/dph = (-st sp, st cp, 0);
 * x_1 = (sin q0, 0, cos q0): dx_1/dq0 = (cos q0, 0, -sin q0).  Over the cell, sin and cos of each coordinate are
 * enclosed by sin(q + d) = sin q cos d + cos q sin d with cos d in [cos h, 1] and sin d in [-sin h, sin h].
 */
typedef struct { I v[3]; } V3;
static inline V3 v3(I a, I b, I c) { V3 r = {{a, b, c}}; return r; }
static inline V3 vadd(V3 a, V3 b) { return v3(iadd(a.v[0], b.v[0]), iadd(a.v[1], b.v[1]), iadd(a.v[2], b.v[2])); }
static inline V3 vsub(V3 a, V3 b) { return v3(isub(a.v[0], b.v[0]), isub(a.v[1], b.v[1]), isub(a.v[2], b.v[2])); }
static inline V3 vscale(I s, V3 a) { return v3(imul(s, a.v[0]), imul(s, a.v[1]), imul(s, a.v[2])); }
static inline I vdot(V3 a, V3 b) { return iadd(iadd(imul(a.v[0], b.v[0]), imul(a.v[1], b.v[1])), imul(a.v[2], b.v[2])); }
static inline V3 vcross(V3 a, V3 b) {
    return v3(isub(imul(a.v[1], b.v[2]), imul(a.v[2], b.v[1])), isub(imul(a.v[2], b.v[0]), imul(a.v[0], b.v[2])),
              isub(imul(a.v[0], b.v[1]), imul(a.v[1], b.v[0])));
}
static inline I ihull(I a, I b) { I r = {fmin(a.lo, b.lo), fmax(a.hi, b.hi)}; return r; }
static inline I ipm(double h) { I r = {-h, h}; return r; }

typedef struct {
    V3 Xc[NMAX];                 /* centre points (enclosures) */
    V3 XB[NMAX];                 /* points over the cell */
    V3 DT[NMAX], DP[NMAX];       /* d x_i / d theta_i, d x_i / d phi_i over the cell (point 1: DT = d/dq0) */
} FOData;

/* sin, cos over [q - h, q + h] from the centre enclosures s, c */
static void trig_cell(I s, I c, double h, I *sB, I *cB) {
    /* |d| <= h: cos d in [cos h, 1] needs h <= pi, |sin d| <= sin h needs h <= pi/2 (else use the trivial bounds) */
    I sh, ch; isincos(h, &sh, &ch);
    double sm = h < 1.5 ? fmin(fmax(fabs(sh.lo), fabs(sh.hi)), 1.0) : 1.0;
    I cd = {h < 3.1 ? ch.lo : -1.0, 1.0}, sd = {-sm, sm};
    *sB = iadd(imul(s, cd), imul(c, sd));
    *cB = isub(imul(c, cd), imul(s, sd));
    if (sB->lo < -1) sB->lo = -1;
    if (sB->hi > 1) sB->hi = 1;
    if (cB->lo < -1) cB->lo = -1;
    if (cB->hi > 1) cB->hi = 1;
}

static void fo_prep(const Cell *c, const CellR *R, FOData *F) {
    for (int i = 0; i < N; i++) F->Xc[i] = v3(R->X[i][0], R->X[i][1], R->X[i][2]);
    I z = ipt(0), one = ipt(1);
    F->XB[0] = v3(z, z, one); F->DT[0] = v3(z, z, z); F->DP[0] = v3(z, z, z);
    I s0, c0; trig_cell(c->sq[0], c->cq[0], c->h[0], &s0, &c0);
    F->XB[1] = v3(s0, z, c0); F->DT[1] = v3(c0, z, ineg(s0)); F->DP[1] = v3(z, z, z);
    for (int i = 2; i < N; i++) {
        int kt = 2 * i - 3, kp = 2 * i - 2;
        I st, ct, sp, cp;
        trig_cell(c->sq[kt], c->cq[kt], c->h[kt], &st, &ct);
        trig_cell(c->sq[kp], c->cq[kp], c->h[kp], &sp, &cp);
        F->XB[i] = v3(imul(st, cp), imul(st, sp), ct);
        F->DT[i] = v3(imul(ct, cp), imul(ct, sp), ineg(st));
        F->DP[i] = v3(ineg(imul(st, sp)), imul(st, cp), z);
    }
}

/* level of a cap from its defining points; returns 0 if not well defined (|S| or |n| may vanish) */
static int cap_level(const V3 *P, const int *I3, int k, int side, I *t) {
    if (k == 2) {
        V3 S = vadd(P[I3[0]], P[I3[1]]); I SS = vdot(S, S);
        if (!(SS.lo > 0)) return 0;
        *t = ismul(0.5, isqrt_(SS)); return 1;
    }
    V3 a = P[I3[0]], b = P[I3[1]], cc = P[I3[2]];
    V3 n = vcross(vsub(b, a), vsub(cc, a)); I nn = vdot(n, n);
    if (!(nn.lo > 0)) return 0;
    I t0 = idiv(vdot(n, a), isqrt_(nn));
    *t = side > 0 ? t0 : ineg(t0); return 1;
}

/* robust count: points certainly in the closed cap throughout the cell (by interval sign of the margin) */
static int cap_count(const V3 *P, const int *I3, int k, int side) {
    int n = k;
    if (k == 2) {
        V3 S = vadd(P[I3[0]], P[I3[1]]); I SS = vdot(S, S);
        for (int j = 0; j < N; j++) {
            if (j == I3[0] || j == I3[1]) continue;
            I m = isub(ismul(2.0, vdot(S, P[j])), SS);          /* 2 S.y - |S|^2 >= 0  iff  y in the cap */
            if (m.lo >= 0) n++;
        }
        return n;
    }
    V3 a = P[I3[0]], b = P[I3[1]], cc = P[I3[2]];
    V3 nv = vcross(vsub(b, a), vsub(cc, a)); I tr = vdot(nv, a);
    for (int j = 0; j < N; j++) {
        if (j == I3[0] || j == I3[1] || j == I3[2]) continue;
        I m = isub(vdot(nv, P[j]), tr); if (side < 0) m = ineg(m);
        if (m.lo >= 0) n++;
    }
    return n;
}

/* gradient of t over the cell w.r.t. the points of I (enclosures), into g[0..k-1]; 0 if not well defined */
static int cap_grad(const V3 *P, const int *I3, int k, int side, V3 *g) {
    if (k == 2) {
        V3 S = vadd(P[I3[0]], P[I3[1]]); I SS = vdot(S, S);
        if (!(SS.lo > 0)) return 0;
        I inv = idiv(ipt(0.5), isqrt_(SS));
        g[0] = g[1] = vscale(inv, S); return 1;
    }
    V3 a = P[I3[0]], b = P[I3[1]], cc = P[I3[2]];
    V3 n = vcross(vsub(b, a), vsub(cc, a)); I nn = vdot(n, n);
    if (!(nn.lo > 0)) return 0;
    I nrm = isqrt_(nn), inv = idiv(ipt(1.0), nrm), inv3 = idiv(ipt(1.0), imul(nn, nrm));
    I det = vdot(n, a), dk = imul(det, inv3);
    g[0] = vsub(vscale(inv, vcross(b, cc)), vscale(dk, vcross(vsub(b, cc), n)));
    g[1] = vsub(vscale(inv, vcross(cc, a)), vscale(dk, vcross(vsub(cc, a), n)));
    g[2] = vsub(vscale(inv, vcross(a, b)), vscale(dk, vcross(vsub(a, b), n)));
    if (side < 0) for (int m = 0; m < 3; m++) g[m] = vscale(ipt(-1.0), g[m]);
    return 1;
}

#define FO_MAXC KMAX
static double SMEAR_MIX = 0.1;   /* weight of the width term in the smear (heuristic) */
static double FO_SMEAR[QMAX];          /* per-call: (|m_k| + r_k) h_k of the best single cap (for splitting) */
#ifdef _OPENMP
#pragma omp threadprivate(FO_SMEAR)
#endif

/* first-order lower bound for D on the cell (lower end of an interval enclosure); also fills FO_SMEAR */
static double first_order_bound_r(const Cell *c, const CellR *R) {
    FOData F; fo_prep(c, R, &F);
    static double A[FO_MAXC], M[FO_MAXC][QMAX], RR[FO_MAXC][QMAX];
#ifdef _OPENMP
#pragma omp threadprivate(A, M, RR)
#endif
    int nc = 0; double best = ndn(1.0 / N); int bestc = -1;
    for (int t = 0; t < NT3 + NP2; t++) {
        int k = t < NT3 ? 3 : 2; const int *I3 = t < NT3 ? T3[t] : P2[t - NT3];
        for (int side = 1; side >= (k == 3 ? -1 : 1); side -= 2) {
            if (nc >= FO_MAXC) break;
            I tc, dummy;
            if (!cap_level(F.Xc, I3, k, side, &tc) || !cap_level(F.XB, I3, k, side, &dummy)) continue;
            V3 g[3]; if (!cap_grad(F.XB, I3, k, side, g)) continue;
            int n = cap_count(F.XB, I3, k, side);
            /* d phi / d q_j = (1/2) sum_{i in I} dt/dx_i . dx_i/dq_j */
            I G[QMAX]; for (int j = 0; j < NQ; j++) G[j] = ipt(0);
            for (int m = 0; m < k; m++) {
                int i = I3[m];
                if (i == 0) continue;
                if (i == 1) { G[0] = iadd(G[0], vdot(g[m], F.DT[1])); continue; }
                G[2 * i - 3] = iadd(G[2 * i - 3], vdot(g[m], F.DT[i]));
                G[2 * i - 2] = iadd(G[2 * i - 2], vdot(g[m], F.DP[i]));
            }
            /* phi(c) = n/N - (1 - t_c)/2, enclosure; a = phi(c) - sum r_j h_j (lower end) */
            I phic = isub(idiv(ipt((double)n), ipt((double)N)), ismul(0.5, isub(ipt(1.0), tc)));
            double a = phic.lo, single;
            for (int j = 0; j < NQ; j++) {
                I gj = ismul(0.5, G[j]);
                double mj = 0.5 * (gj.lo + gj.hi);
                double rj = fmax(nup(gj.hi - mj), nup(mj - gj.lo));     /* [mj - rj, mj + rj] contains gj */
                M[nc][j] = mj; RR[nc][j] = rj;
                a = ndn(a - nup(rj * c->h[j]));
            }
            single = a;
            for (int j = 0; j < NQ; j++) single = ndn(single - nup(fabs(M[nc][j]) * c->h[j]));
            if (single > best) { best = single; bestc = nc; }
            A[nc++] = a;
        }
    }
    /* smear of the best single cap, plus a small multiple of the (sin-weighted) width, so that coordinates that
       the best cap does not see are still split eventually (otherwise the cell never shrinks in them) */
    for (int j = 0; j < NQ; j++) {
        double wk = c->h[j];
        if (j >= 2 && (j % 2 == 0)) wk *= fmax(fabs(c->sq[j - 1].lo), fabs(c->sq[j - 1].hi)) + 1e-3;
        FO_SMEAR[j] = (bestc >= 0 ? (fabs(M[bestc][j]) + RR[bestc][j]) * c->h[j] : 0.0) + SMEAR_MIX * wk;
    }
    if (nc < 2) return best;
    /* weights: exponentiated-gradient ascent on g(lam) = sum lam a - sum_j |sum lam m_j| h_j over the simplex,
       restricted to the caps with a close to the largest (floating point: a choice only) */
    double amax = -1e300; for (int i = 0; i < nc; i++) if (A[i] > amax) amax = A[i];
    int idx[FO_MAXC], m = 0; double hsum = 0; for (int j = 0; j < NQ; j++) hsum += c->h[j];
    for (int i = 0; i < nc; i++) if (A[i] >= amax - 2.0 * hsum) idx[m++] = i;
    if (m < 2) return best;
    double lam[FO_MAXC], gsum[QMAX];
    for (int i = 0; i < m; i++) lam[i] = 1.0 / m;
    double eta = 50.0 / (hsum + 1e-300), bestg = -1e300, bestlam[FO_MAXC];
    for (int it = 0; it < 60; it++) {
        double gv = 0;
        for (int j = 0; j < NQ; j++) { gsum[j] = 0; for (int i = 0; i < m; i++) gsum[j] += lam[i] * M[idx[i]][j]; }
        for (int i = 0; i < m; i++) gv += lam[i] * A[idx[i]];
        for (int j = 0; j < NQ; j++) gv -= fabs(gsum[j]) * c->h[j];
        if (gv > bestg) { bestg = gv; for (int i = 0; i < m; i++) bestlam[i] = lam[i]; }
        double z = 0, mx = -1e300, sg[QMAX];
        for (int j = 0; j < NQ; j++) sg[j] = (gsum[j] > 0) - (gsum[j] < 0);
        double gr[FO_MAXC];
        for (int i = 0; i < m; i++) {
            gr[i] = A[idx[i]];
            for (int j = 0; j < NQ; j++) gr[i] -= sg[j] * M[idx[i]][j] * c->h[j];
            if (gr[i] > mx) mx = gr[i];
        }
        for (int i = 0; i < m; i++) { lam[i] *= exp(eta * (gr[i] - mx)); z += lam[i]; }
        if (!(z > 0)) break;
        for (int i = 0; i < m; i++) lam[i] /= z;
    }
    /* rigorous evaluation with the chosen weights (exact doubles) */
    I acc = ipt(0), S = ipt(0);
    for (int i = 0; i < m; i++) { acc = iadd(acc, ismul(bestlam[i], ipt(A[idx[i]]))); S = iadd(S, ipt(bestlam[i])); }
    I pen = ipt(0);
    for (int j = 0; j < NQ; j++) {
        I gj = ipt(0);
        for (int i = 0; i < m; i++) gj = iadd(gj, ismul(bestlam[i], ipt(M[idx[i]][j])));
        double ab = fmax(fabs(gj.lo), fabs(gj.hi));
        pen = iadd(pen, ipt(nup(ab * c->h[j])));
    }
    if (!(S.lo > 0)) return best;
    I lb = idiv(isub(acc, pen), S);
    return lb.lo > best ? lb.lo : best;
}
