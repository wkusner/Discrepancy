/* ---------------- per-cell data
 * Cell (rigorous mode) caches enclosures of sin/cos of every centre coordinate and of every
 * motion bound E_i; a split changes one coordinate, so only that coordinate and the motion
 * bound of its point are recomputed.  */
#define EBIG 3.0                    /* E_i above this: no information from that point */
typedef struct {
    I X[NMAX][3];                   /* enclosure of the centre points */
    double Xm[NMAX][3];             /* midpoints (for choosing axes only) */
} CellR;

static inline int coord_point(int k) { return k == 0 ? 1 : (k + 3) / 2; }

static void update_E(Cell *c, int i) {
    if (i == 0) c->E[0] = 0;
    else if (i == 1) c->E[1] = c->h[0];
    else {
        I st = c->sq[2 * i - 3];
        double sth = fmax(fabs(st.lo), fabs(st.hi));
        c->E[i] = nup(c->h[2 * i - 3] + nup(sth * c->h[2 * i - 2]));
    }
    if (c->E[i] < EBIG) isincos(c->E[i], &c->sE[i], &c->cE[i]);
}

static void init_cache(Cell *c) {
    for (int k = 0; k < NQ; k++) isincos(c->q[k], &c->sq[k], &c->cq[k]);
    for (int i = 0; i < N; i++) update_E(c, i);
}

static void prep_r(const Cell *c, CellR *R) {
    for (int k = 0; k < 3; k++) R->X[0][k] = ipt(k == 2 ? 1.0 : 0.0);
    R->X[1][0] = c->sq[0]; R->X[1][1] = ipt(0); R->X[1][2] = c->cq[0];
    for (int i = 2; i < N; i++) {
        I st = c->sq[2 * i - 3], ct = c->cq[2 * i - 3], sp = c->sq[2 * i - 2], cp = c->cq[2 * i - 2];
        R->X[i][0] = imul(st, cp); R->X[i][1] = imul(st, sp); R->X[i][2] = ct;
    }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++) R->Xm[i][k] = 0.5 * (R->X[i][k].lo + R->X[i][k].hi);
}
/* cos of the angle between the exact double vector w and the true point i, and its sine */
static inline void sort_desc(double *a, int n) {
    for (int i = 1; i < n; i++) { double v = a[i]; int j = i - 1; while (j >= 0 && a[j] < v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
}

/* upper bound of sqrt(1 - x^2) for x in the interval cd (|x| <= 1) */
static inline double sin_hi(I cd) {
    double m = cd.lo > 0 ? ndn(cd.lo * cd.lo) : (cd.hi < 0 ? ndn(cd.hi * cd.hi) : 0.0);
    double q = nup(1.0 - m);
    return nup(sqrt(q));
}

/* bound from one axis w (any nonzero double vector) */
static double axis_bound_r(const double *w, const Cell *c, const CellR *R) {
    I nw = iadd(iadd(isqr(ipt(w[0])), isqr(ipt(w[1]))), isqr(ipt(w[2])));
    if (!(nw.lo > 0)) return -1;
    I inv = idiv(ipt(1.0), isqrt_(nw));                  /* 1/|w| */
    double cl[NMAX], Cu[NMAX];
    for (int i = 0; i < N; i++) {
        const I *x = R->X[i];
        I cd = imul(iadd(iadd(ismul(w[0], x[0]), ismul(w[1], x[1])), ismul(w[2], x[2])), inv);  /* cos d_i */
        if (cd.lo < -1) cd.lo = -1;
        if (cd.hi > 1) cd.hi = 1;
        double E = c->E[i];
        if (E >= EBIG) { cl[i] = -1; Cu[i] = 1; continue; }
        if (E == 0) { cl[i] = cd.lo; Cu[i] = cd.hi; continue; }
        I cE = c->cE[i];
        double ss = nup(sin_hi(cd) * fmax(c->sE[i].hi, 0.0));   /* >= sin d_i sin E_i */
        I p = imul(cd, cE);                                        /* cos d_i cos E_i */
        cl[i] = (ndn(cd.lo + cE.lo) >= 0) ? fmax(ndn(p.lo - ss), -1.0) : -1.0;   /* <= cos(d+E) */
        Cu[i] = (cd.hi <= cE.lo) ? fmin(nup(p.hi + ss), 1.0) : 1.0;              /* >= cos(d-E) */
    }
    sort_desc(cl, N); sort_desc(Cu, N);
    double best = -1;
    for (int k = 1; k <= N; k++) {
        double kN = ndn((double)k / N), k1N = nup((double)(k - 1) / N);
        double up = ndn(kN - nup(1.0 - cl[k - 1]) * 0.5);      /* k/N - (1 - c_(k))/2 */
        double lo = ndn(ndn(1.0 - Cu[k - 1]) * 0.5 - k1N);     /* (1 - C_(k))/2 - (k-1)/N */
        if (up > best) best = up;
        if (lo > best) best = lo;
    }
    return best;
}

/* rigorous lower bound for D on the cell (stops early once >= stop) */
static double cell_bound_r(const Cell *c, const CellR *R, double stop) {
    double best = ndn(1.0 / N), w[3];
    for (int t = 0; t < NT3 + NP2; t++) {
        if (t < NT3) {
            const double *a = R->Xm[T3[t][0]], *b = R->Xm[T3[t][1]], *e = R->Xm[T3[t][2]];
            double u0 = b[0] - a[0], u1 = b[1] - a[1], u2 = b[2] - a[2];
            double v0 = e[0] - a[0], v1 = e[1] - a[1], v2 = e[2] - a[2];
            w[0] = u1 * v2 - u2 * v1; w[1] = u2 * v0 - u0 * v2; w[2] = u0 * v1 - u1 * v0;
        } else {
            const double *a = R->Xm[P2[t - NT3][0]], *b = R->Xm[P2[t - NT3][1]];
            w[0] = a[0] + b[0]; w[1] = a[1] + b[1]; w[2] = a[2] + b[2];
        }
        double n = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (n < 1e-150) continue;
        w[0] /= n; w[1] /= n; w[2] /= n;           /* any double vector is a valid axis */
        for (int s = 0; s < 2; s++) {
            double m[3] = {s ? -w[0] : w[0], s ? -w[1] : w[1], s ? -w[2] : w[2]};
            double v = axis_bound_r(m, c, R);
            if (v > best) { best = v; if (best >= stop) return best; }
        }
    }
    return best;
}

/* 1 if the cell contains no point of the fundamental domain */
static int outside_domain_r(const Cell *c, const CellR *R) {
    for (int i = 2; i < N - 1; i++) {
        int a = 2 * i - 3, b = 2 * (i + 1) - 3;
        if (ndn(c->q[a] - c->h[a]) > nup(c->q[b] + c->h[b])) return 1;   /* th_i > th_{i+1} throughout */
    }
    double th1 = ndn(c->q[0] - c->h[0]);              /* d(y0,y1) >= th1 on the cell */
    if (!(th1 > 0)) return 0;
    I sT, cT; isincos(th1, &sT, &cT);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (i == 0 && j == 1) continue;
            if (nup(c->E[i] + c->E[j]) >= EBIG) continue;
            /* d(y_i,y_j) <= d(x_i,x_j) + E_i + E_j =: d + e ; we need d + e < th1 */
            I cd = iadd(iadd(imul(R->X[i][0], R->X[j][0]), imul(R->X[i][1], R->X[j][1])), imul(R->X[i][2], R->X[j][2]));
            if (cd.lo < -1) cd.lo = -1;
            if (cd.hi > 1) cd.hi = 1;
            I ce, se;                                 /* cos, sin of e = E_i + E_j (exact sum) */
            if (c->E[i] == 0) { ce = c->cE[j]; se = c->sE[j]; }
            else {
                ce = isub(imul(c->cE[i], c->cE[j]), imul(c->sE[i], c->sE[j]));
                se = iadd(imul(c->sE[i], c->cE[j]), imul(c->cE[i], c->sE[j]));
            }
            if (!(ndn(cd.lo + ce.lo) >= 0)) continue;  /* need d + e <= pi */
            double v = ndn(imul(cd, ce).lo - nup(sin_hi(cd) * fmax(se.hi, 0.0)));   /* <= cos(d + e) */
            if (v > cT.hi) return 1;                   /* cos(d+e) > cos(th1)  =>  d + e < th1 */
        }
    return 0;
}

/* split along the widest coordinate; children cover the parent exactly (see header) */
static void split_r(const Cell *c, Cell *a, Cell *b) {
    int j = 0; double wmax = -1;
    for (int k = 0; k < NQ; k++) {
        double wk = c->h[k];
        if (k >= 2 && (k % 2 == 0)) {             /* phi coordinate: weight by |sin theta| (heuristic only) */
            I st = c->sq[k - 1]; wk = c->h[k] * (fmax(fabs(st.lo), fabs(st.hi)) + 1e-3);
        }
        if (wk > wmax) { wmax = wk; j = k; }
    }
    *a = *c; *b = *c;
    double hh = nup(c->h[j] * 0.5 + nup((fabs(c->q[j]) + c->h[j]) * 0x1p-52));
    a->h[j] = b->h[j] = hh;
    a->q[j] = c->q[j] - c->h[j] * 0.5;
    b->q[j] = c->q[j] + c->h[j] * 0.5;
    isincos(a->q[j], &a->sq[j], &a->cq[j]);
    isincos(b->q[j], &b->sq[j], &b->cq[j]);
    int p = coord_point(j);
    update_E(a, p); update_E(b, p);
}

static void root_r(Cell *root) {
    double P = M_PI / 2;                       /* exact halving of the double M_PI */
    root->q[0] = P; root->h[0] = nup(P);       /* [P - h, P + h] contains [0, pi] */
    for (int i = 2; i < N; i++) {
        root->q[2 * i - 3] = P; root->h[2 * i - 3] = nup(P);
        if (i == 2) { root->q[2] = P; root->h[2] = nup(P); }
        else { root->q[2 * i - 2] = 0; root->h[2 * i - 2] = nup(M_PI); }   /* contains [-pi, pi] */
    }
    init_cache(root);
}
