/*
 * capcert_arb.c -- second-kernel mesh stage (Arb ball arithmetic, FLINT), see VERIFICATION.md.
 *
 * Independent of csrc/ia.h, csrc/rig.h and capcert.c's arithmetic.  For a cell (box of
 * parameters q, centre c, half-widths h) every point y_i is enclosed directly as a ball vector
 * by evaluating x(theta, phi) with Arb on the parameter balls (Arb's rigorous sin/cos of balls).
 * For an axis w (any double vector, chosen from the caps of the centre configuration) the balls
 * of <y_i, w>/|w| give numbers c_i <= <y_i,w^> <= C_i for every configuration in the cell, and
 *     D >= k/N - (1 - c_(k))/2,    D >= (1 - C_(k))/2 - (k-1)/N,    D >= 1/N
 * (c_(k), C_(k) the k-th largest), the cell lemma of the paper (Section 10).
 * Pruning (no point of the fundamental domain in the cell): theta_i > theta_{i+1} throughout,
 * or <y_i,y_j> > <y_0,y_1> = cos q0 throughout for some pair (i,j) != (0,1).
 * Covering: children of a cell are computed exactly in Arb (c +- h/2); if the new centre is not
 * a double, the half-width is enlarged by the rounding error (computed in Arb, rounded up).
 * The root box contains [0, pi] (theta, q0, phi_2) and [-pi, pi] (other phi), checked with
 * arb_const_pi.
 * Output: OUT.surv.<tid> (same binary format as capcert: q, h, lb as doubles) and OUT.done
 * (mode=arb).  No resume: a run either completes or is repeated.
 *
 * Build: gcc -O2 -fopenmp -o capcert_arb csrc/capcert_arb.c -lflint -lgmp -lm
 * Usage: capcert_arb N T MIN_EPS OUT [--threads K]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <flint/arb.h>

#define NMAX 10
#define QMAX (2 * NMAX - 3)
#define PREC 64

static int N, NQ;
static double TGT, MIN_EPS;
static int T3[120][3], P2[45][2], NT3, NP2;
typedef struct { double q[QMAX], h[QMAX]; } Cell;
typedef struct { long processed, bounded, sym, surv; } Stats;

/* arithmetic with an exact double operand */
static void arb_mul_dbl(arb_t z, const arb_t x, double d, slong prec) { arb_t t; arb_init(t); arb_set_d(t, d); arb_mul(z, x, t, prec); arb_clear(t); }
static void arb_add_dbl(arb_t z, const arb_t x, double d, slong prec) { arb_t t; arb_init(t); arb_set_d(t, d); arb_add(z, x, t, prec); arb_clear(t); }
static void arb_sub_dbl(arb_t z, const arb_t x, double d, slong prec) { arb_t t; arb_init(t); arb_set_d(t, d); arb_sub(z, x, t, prec); arb_clear(t); }

/* ball for the parameter interval [c - h, c + h] (exact: c, h are doubles) */
static void param_ball(arb_t b, double c, double h) {
    arb_set_d(b, c);
    mag_set_d(arb_radref(b), h);           /* rounds up */
}

/* enclosures of all points over the cell (centre = 0 width when h = 0) */
static void points_ball(arb_t Y[][3], const double *q, const double *h, int centre_only) {
    arb_t th, ph, s1, c1, s2, c2;
    arb_init(th); arb_init(ph); arb_init(s1); arb_init(c1); arb_init(s2); arb_init(c2);
    arb_zero(Y[0][0]); arb_zero(Y[0][1]); arb_one(Y[0][2]);
    param_ball(th, q[0], centre_only ? 0 : h[0]);
    arb_sin_cos(s1, c1, th, PREC);
    arb_set(Y[1][0], s1); arb_zero(Y[1][1]); arb_set(Y[1][2], c1);
    for (int i = 2; i < N; i++) {
        param_ball(th, q[2 * i - 3], centre_only ? 0 : h[2 * i - 3]);
        param_ball(ph, q[2 * i - 2], centre_only ? 0 : h[2 * i - 2]);
        arb_sin_cos(s1, c1, th, PREC); arb_sin_cos(s2, c2, ph, PREC);
        arb_mul(Y[i][0], s1, c2, PREC); arb_mul(Y[i][1], s1, s2, PREC); arb_set(Y[i][2], c1);
    }
    arb_clear(th); arb_clear(ph); arb_clear(s1); arb_clear(c1); arb_clear(s2); arb_clear(c2);
}

static int cmp_desc(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x < y) - (x > y);
}

/* lower bound for D on the cell from one axis w (double vector, |w| ~ 1) */
static double axis_bound(const double *w, arb_t Y[][3], arb_t inv_nw, arb_t tmp, arb_t acc) {
    double cl[NMAX], Cu[NMAX];
    for (int i = 0; i < N; i++) {
        arb_zero(acc);
        for (int k = 0; k < 3; k++) { arb_mul_dbl(tmp, Y[i][k], w[k], PREC); arb_add(acc, acc, tmp, PREC); }
        arb_mul(acc, acc, inv_nw, PREC);
        arf_t lo, hi; arf_init(lo); arf_init(hi);
        arb_get_lbound_arf(lo, acc, PREC); arb_get_ubound_arf(hi, acc, PREC);
        double l = arf_get_d(lo, ARF_RND_FLOOR), u = arf_get_d(hi, ARF_RND_CEIL);
        arf_clear(lo); arf_clear(hi);
        cl[i] = l < -1 ? -1 : l; Cu[i] = u > 1 ? 1 : u;
    }
    qsort(cl, N, sizeof(double), cmp_desc); qsort(Cu, N, sizeof(double), cmp_desc);
    double best = -1;
    arb_t v; arb_init(v);
    for (int k = 1; k <= N; k++) {
        /* k/N - (1 - c_(k))/2 */
        arb_set_si(v, k); arb_div_si(v, v, N, PREC);
        arb_set_d(tmp, cl[k - 1]); arb_sub_si(tmp, tmp, 1, PREC); arb_mul_2exp_si(tmp, tmp, -1);
        arb_add(v, v, tmp, PREC);
        arf_t lo; arf_init(lo); arb_get_lbound_arf(lo, v, PREC); double a = arf_get_d(lo, ARF_RND_FLOOR);
        /* (1 - C_(k))/2 - (k-1)/N */
        arb_set_d(tmp, Cu[k - 1]); arb_neg(tmp, tmp); arb_add_si(tmp, tmp, 1, PREC); arb_mul_2exp_si(tmp, tmp, -1);
        arb_set_si(v, k - 1); arb_div_si(v, v, N, PREC); arb_sub(v, tmp, v, PREC);
        arb_get_lbound_arf(lo, v, PREC); double b = arf_get_d(lo, ARF_RND_FLOOR); arf_clear(lo);
        if (a > best) best = a;
        if (b > best) best = b;
    }
    arb_clear(v);
    return best;
}

static double cell_bound(const Cell *c, double stop) {
    arb_t Y[NMAX][3], Xc[NMAX][3];
    for (int i = 0; i < N; i++) for (int k = 0; k < 3; k++) { arb_init(Y[i][k]); arb_init(Xc[i][k]); }
    points_ball(Y, c->q, c->h, 0);
    points_ball(Xc, c->q, c->h, 1);
    double X[NMAX][3];
    for (int i = 0; i < N; i++) for (int k = 0; k < 3; k++) X[i][k] = arf_get_d(arb_midref(Xc[i][k]), ARF_RND_NEAR);
    /* lower bound 1/N */
    double best = (1.0 / N) * (1 - 1e-15);
    arb_t inv_nw, tmp, acc; arb_init(inv_nw); arb_init(tmp); arb_init(acc);
    for (int t = 0; t < NT3 + NP2 && best < stop; t++) {
        double w[3];
        if (t < NT3) {
            const double *a = X[T3[t][0]], *b = X[T3[t][1]], *e = X[T3[t][2]];
            double u0 = b[0] - a[0], u1 = b[1] - a[1], u2 = b[2] - a[2], v0 = e[0] - a[0], v1 = e[1] - a[1], v2 = e[2] - a[2];
            w[0] = u1 * v2 - u2 * v1; w[1] = u2 * v0 - u0 * v2; w[2] = u0 * v1 - u1 * v0;
        } else {
            const double *a = X[P2[t - NT3][0]], *b = X[P2[t - NT3][1]];
            w[0] = a[0] + b[0]; w[1] = a[1] + b[1]; w[2] = a[2] + b[2];
        }
        double n = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (!(n > 1e-150)) continue;
        for (int k = 0; k < 3; k++) w[k] /= n;
        /* 1/|w| as a ball (w is an exact double vector) */
        arb_zero(acc);
        for (int k = 0; k < 3; k++) { arb_set_d(tmp, w[k]); arb_addmul(acc, tmp, tmp, PREC); }
        arb_rsqrt(inv_nw, acc, PREC);
        for (int s = 0; s < 2 && best < stop; s++) {
            double m[3] = {s ? -w[0] : w[0], s ? -w[1] : w[1], s ? -w[2] : w[2]};
            double v = axis_bound(m, Y, inv_nw, tmp, acc);
            if (v > best) best = v;
        }
    }
    arb_clear(inv_nw); arb_clear(tmp); arb_clear(acc);
    for (int i = 0; i < N; i++) for (int k = 0; k < 3; k++) { arb_clear(Y[i][k]); arb_clear(Xc[i][k]); }
    return best;
}

static int misses_domain(const Cell *c) {
    int out = 0;
    arb_t A, B; arb_init(A); arb_init(B);
    for (int i = 2; i < N - 1 && !out; i++) {
        int a = 2 * i - 3, b = 2 * (i + 1) - 3;
        param_ball(A, c->q[a], c->h[a]); param_ball(B, c->q[b], c->h[b]);
        if (arb_gt(A, B)) out = 1;                       /* theta_i > theta_{i+1} throughout */
    }
    if (!out) {
        arb_t Y[NMAX][3], g, t, c01;
        for (int i = 0; i < N; i++) for (int k = 0; k < 3; k++) arb_init(Y[i][k]);
        arb_init(g); arb_init(t); arb_init(c01);
        points_ball(Y, c->q, c->h, 0);
        param_ball(A, c->q[0], c->h[0]); arb_cos(c01, A, PREC);     /* <y0,y1> = cos q0 */
        for (int i = 0; i < N && !out; i++)
            for (int j = i + 1; j < N && !out; j++) {
                if (i == 0 && j == 1) continue;
                arb_zero(g);
                for (int k = 0; k < 3; k++) { arb_mul(t, Y[i][k], Y[j][k], PREC); arb_add(g, g, t, PREC); }
                if (arb_gt(g, c01)) out = 1;               /* d(y_i,y_j) < d(y_0,y_1) throughout */
            }
        for (int i = 0; i < N; i++) for (int k = 0; k < 3; k++) arb_clear(Y[i][k]);
        arb_clear(g); arb_clear(t); arb_clear(c01);
    }
    arb_clear(A); arb_clear(B);
    return out;
}

/* exact child centre c + sgn*h/2 in Arb; returns the double centre and an enlarged half-width */
static void child(double c, double h, int sgn, double *cc, double *hh) {
    arb_t m, e; arb_init(m); arb_init(e);
    arb_set_d(m, h); arb_mul_2exp_si(m, m, -1); if (sgn < 0) arb_neg(m, m);
    arb_add_dbl(m, m, c, 2 * PREC);                        /* exact for doubles */
    *cc = arf_get_d(arb_midref(m), ARF_RND_NEAR);
    arb_sub_dbl(e, m, *cc, 2 * PREC); arb_abs(e, e);       /* rounding error of the new centre */
    arb_set_d(m, h); arb_mul_2exp_si(m, m, -1); arb_add(e, e, m, 2 * PREC);
    arf_t up; arf_init(up); arb_get_ubound_arf(up, e, PREC); *hh = arf_get_d(up, ARF_RND_CEIL); arf_clear(up);
    arb_clear(m); arb_clear(e);
}

static void split(const Cell *c, Cell *a, Cell *b) {
    int j = 0; double wmax = -1;
    for (int k = 0; k < NQ; k++) {
        double wk = c->h[k];
        if (k >= 2 && (k % 2 == 0)) wk = c->h[k] * (fabs(sin(c->q[k - 1])) + 1e-3);
        if (wk > wmax) { wmax = wk; j = k; }
    }
    *a = *c; *b = *c;
    child(c->q[j], c->h[j], -1, &a->q[j], &a->h[j]);
    child(c->q[j], c->h[j], +1, &b->q[j], &b->h[j]);
}

static void root(Cell *r) {
    /* theta-type and phi_2: centre P = fl(pi/2), half-width hP >= pi/2 + |P - pi/2| (so the box
       contains [0, pi]); other phi: centre 0, half-width hPi >= pi (contains [-pi, pi]) */
    arb_t pi, e; arb_init(pi); arb_init(e); arb_const_pi(pi, PREC);
    double P = M_PI / 2, hP, hPi;
    arb_mul_2exp_si(e, pi, -1);                 /* pi/2 */
    arb_sub_dbl(e, e, P, PREC); arb_abs(e, e);    /* |pi/2 - P| */
    arb_t half; arb_init(half); arb_mul_2exp_si(half, pi, -1); arb_add(e, e, half, PREC); arb_clear(half);
    arf_t up; arf_init(up);
    arb_get_ubound_arf(up, e, PREC); hP = arf_get_d(up, ARF_RND_CEIL);
    arb_get_ubound_arf(up, pi, PREC); hPi = arf_get_d(up, ARF_RND_CEIL); arf_clear(up);
    r->q[0] = P; r->h[0] = hP;
    for (int i = 2; i < N; i++) {
        r->q[2 * i - 3] = P; r->h[2 * i - 3] = hP;
        if (i == 2) { r->q[2] = P; r->h[2] = hP; } else { r->q[2 * i - 2] = 0; r->h[2 * i - 2] = hPi; }
    }
    arb_clear(pi); arb_clear(e);
}

static double emax_of(const Cell *c) {          /* stopping size (selection only) */
    double e = c->h[0];
    for (int i = 2; i < N; i++) { double v = c->h[2 * i - 3] + fabs(sin(c->q[2 * i - 3])) * c->h[2 * i - 2]; if (v > e) e = v; }
    return e;
}

static void dfs(const Cell *r, Stats *st, FILE *fp, int *ioerr) {
    size_t cap = 4096, top = 0; Cell *stack = malloc(cap * sizeof(Cell));
    if (!stack) { fprintf(stderr, "out of memory\n"); exit(3); }
    stack[top++] = *r;
    while (top) {
        Cell c = stack[--top]; st->processed++;
        if (misses_domain(&c)) { st->sym++; continue; }
        double lb = cell_bound(&c, TGT);
        if (lb >= TGT) { st->bounded++; continue; }
        if (emax_of(&c) < MIN_EPS) {
            st->surv++;
            if (fwrite(c.q, sizeof(double), NQ, fp) != (size_t)NQ || fwrite(c.h, sizeof(double), NQ, fp) != (size_t)NQ
                || fwrite(&lb, sizeof(double), 1, fp) != 1) *ioerr = 1;
            continue;
        }
        if (top + 2 > cap) { cap *= 2; stack = realloc(stack, cap * sizeof(Cell)); if (!stack) { fprintf(stderr, "out of memory\n"); exit(3); } }
        split(&c, &stack[top], &stack[top + 1]); top += 2;
    }
    free(stack);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s N T MIN_EPS OUT [--threads K]\n", argv[0]); return 1; }
    N = atoi(argv[1]); TGT = strtod(argv[2], NULL); MIN_EPS = strtod(argv[3], NULL); const char *out = argv[4];
    int threads = 0;
    for (int a = 5; a < argc; a++) if (!strcmp(argv[a], "--threads")) threads = atoi(argv[++a]);
    if (N < 3 || N > NMAX) return 1;
    NQ = 2 * N - 3;
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
        P2[NP2][0] = i; P2[NP2][1] = j; NP2++;
        for (int k = j + 1; k < N; k++) { T3[NT3][0] = i; T3[NT3][1] = j; T3[NT3][2] = k; NT3++; }
    }
    if (threads > 0) omp_set_num_threads(threads);
    int nthr = omp_get_max_threads();
    long frontier = 2000L * nthr;
    Cell r; root(&r);
    size_t nf = 1; Cell *F = malloc(sizeof(Cell)); F[0] = r;
    while ((long)nf < frontier) {
        Cell *G = malloc(2 * nf * sizeof(Cell));
        if (!G) return 3;
        for (size_t i = 0; i < nf; i++) split(&F[i], &G[2 * i], &G[2 * i + 1]);
        free(F); F = G; nf *= 2;
    }
    fprintf(stderr, "capcert_arb (Arb ball arithmetic) N=%d T=%.17g MIN_EPS=%g threads=%d frontier=%zu\n", N, TGT, MIN_EPS, nthr, nf);
    Stats tot = {0, 0, 0, 0}; int ioerr = 0; long finished = 0; time_t t0 = time(NULL), tl = t0;
#pragma omp parallel
    {
        char fn[4200]; snprintf(fn, sizeof fn, "%s.surv.%d", out, omp_get_thread_num());
        FILE *fp = fopen(fn, "wb"); if (!fp) ioerr = 1;
#pragma omp for schedule(dynamic, 1)
        for (long i = 0; i < (long)nf; i++) {
            if (ioerr) continue;
            Stats st = {0, 0, 0, 0};
            dfs(&F[i], &st, fp, &ioerr);
#pragma omp critical
            {
                tot.processed += st.processed; tot.bounded += st.bounded; tot.sym += st.sym; tot.surv += st.surv; finished++;
                time_t now = time(NULL);
                if (now - tl >= 60) { tl = now; fprintf(stderr, "[%6lds] roots %ld/%zu cells %ld bounded %ld sym %ld survivors %ld\n",
                                                        (long)(now - t0), finished, nf, tot.processed, tot.bounded, tot.sym, tot.surv); }
            }
        }
        if (fp && fclose(fp)) ioerr = 1;
    }
    if (ioerr || finished != (long)nf) { printf("INCOMPLETE\n"); return 2; }
    printf("RESULT N=%d T=%.17g MIN_EPS=%g cells=%ld bounded=%ld sym=%ld survivors=%ld time=%lds kernel=arb\n",
           N, TGT, MIN_EPS, tot.processed, tot.bounded, tot.sym, tot.surv, (long)(time(NULL) - t0));
    char dfn[4200]; snprintf(dfn, sizeof dfn, "%s.done", out);
    FILE *df = fopen(dfn, "w");
    if (!df) return 2;
    fprintf(df, "N=%d T=%a MIN_EPS=%a nf=%zu mode=arb keep=%a balls=0 survivors=%ld cells=%ld\n", N, TGT, MIN_EPS, nf, 1.0, tot.surv, tot.processed);
    fclose(df);
    if (tot.surv == 0) printf("ARB-VERIFIED: D >= T on the whole fundamental domain\n");
    return 0;
}
