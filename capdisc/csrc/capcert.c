/*
 * capcert.c -- branch-and-bound lower bounds for the spherical cap discrepancy
 *
 * Proves  D(X) >= T  for every configuration X of N points on S^2 (N <= 10),
 * cell by cell over the gauge- and symmetry-reduced parameter box, and writes
 * the cells it could not certify ("survivors") once they are smaller than MIN_EPS.
 *
 * Coordinates q (2N-3):  x0 = north pole, x1 = (sin q0, 0, cos q0),
 *   x_i = (sin th cos ph, sin th sin ph, cos th),  th = q[2i-3], ph = q[2i-2]  (i >= 2).
 * Fundamental domain (O(3) x S_N):  (x0,x1) a closest pair; th_2 <= th_3 <= ...;
 *   ph_2 in [0, pi].
 *
 * Cell bound (see sphcap/bnb.py and the paper): with per-point motion bounds eps_i
 * and any axis w,
 *   up : k/N - (1 - cos u_(k))/2,   u = sort(d(c_i,w) + eps_i)
 *   low: (1 - cos l_(k))/2 - (k-1)/N,  l = sort(max(d(c_i,w) - eps_i, 0))
 *   and 1/N.  Axes: the centre's 2- and 3-point cap axes, both signs.
 * Floating-point safety: every angle carries slack ANG_SLACK (angles via atan2,
 * which is well conditioned), every bound is lowered by BND_SLACK before comparing
 * with T.  These slacks exceed the accumulated rounding error by many orders of
 * magnitude given glibc's documented few-ulp accuracy for sin/cos/atan2.
 *
 * Exclusion balls (optional): cells entirely inside a ball
 *   max_i d(y_i, z_i) < r  around a listed gauge-fixed configuration z
 * are skipped (they are covered by a separate local certificate).
 *
 * Parallel: the domain is split deterministically into an initial frontier; OpenMP
 * threads take frontier cells dynamically and run an explicit-stack DFS on each.
 * Checkpoint: one line per finished frontier cell in <out>.progress; rerunning with
 * the same arguments skips finished cells.
 *
 * Build:  gcc -O3 -march=native -fopenmp -o capcert capcert.c -lm
 * Usage:  capcert N T MIN_EPS OUT [--threads K] [--frontier F] [--balls FILE]
 *                 [--max-cells-per-root M] [--keep-prob P]
 * --keep-prob P writes only a (hash-determined) fraction P of the survivor cells
 * (atlas mode); all survivors are still counted.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NMAX 10
#define QMAX (2 * NMAX - 3)
#define KMAX (2 * (120 + 45)) /* 2*(C(10,3)+C(10,2)) */
#define ANG_SLACK 1e-12
#define BND_SLACK 1e-12

static int N, NQ, NT3, NP2;
static int T3[120][3], P2[45][2];
static double TGT, MIN_EPS;
static double KEEP_PROB = 1.0; /* fraction of survivor cells written to disk (all are counted) */
static int NBALLS = 0;
static double (*BALL_Z)[NMAX][3] = NULL;
static double *BALL_R = NULL;

#ifdef RIGOROUS
#include "ia.h"
typedef struct { double q[QMAX], h[QMAX]; I sq[QMAX], cq[QMAX]; double E[NMAX]; I sE[NMAX], cE[NMAX]; } Cell;
#include "rig.h"
#else
typedef struct { double q[QMAX], h[QMAX]; } Cell;
#endif

static inline double angle(const double *a, const double *b) {
    double cx = a[1] * b[2] - a[2] * b[1], cy = a[2] * b[0] - a[0] * b[2], cz = a[0] * b[1] - a[1] * b[0];
    return atan2(sqrt(cx * cx + cy * cy + cz * cz), a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

static void points(const double *q, double X[][3]) {
    X[0][0] = 0; X[0][1] = 0; X[0][2] = 1;
    X[1][0] = sin(q[0]); X[1][1] = 0; X[1][2] = cos(q[0]);
    for (int i = 2; i < N; i++) {
        double th = q[2 * i - 3], ph = q[2 * i - 2], s = sin(th);
        X[i][0] = s * cos(ph); X[i][1] = s * sin(ph); X[i][2] = cos(th);
    }
}

static void motion(const double *q, const double *h, double *E) {
    E[0] = 0; E[1] = h[0];
    for (int i = 2; i < N; i++) E[i] = h[2 * i - 3] + fabs(sin(q[2 * i - 3])) * h[2 * i - 2];
    for (int i = 0; i < N; i++) { E[i] = E[i] * (1 + 1e-12) + 1e-15; if (E[i] > M_PI) E[i] = M_PI; }
}

static inline void sort_small(double *a, int n) {
    for (int i = 1; i < n; i++) { double v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
}

/* best bound along one axis w */
static inline double axis_bound(const double *w, double X[][3], const double *E) {
    double u[NMAX], l[NMAX];
    for (int i = 0; i < N; i++) {
        double d = angle(w, X[i]);
        double uu = d + E[i] + ANG_SLACK, ll = d - E[i] - ANG_SLACK;
        u[i] = uu > M_PI ? M_PI : uu;
        l[i] = ll < 0 ? 0 : ll;
    }
    sort_small(u, N); sort_small(l, N);
    double best = -1;
    for (int k = 1; k <= N; k++) {
        double up = (double)k / N - (1 - cos(u[k - 1])) / 2;
        double lo = (1 - cos(l[k - 1])) / 2 - (double)(k - 1) / N;
        if (up > best) best = up;
        if (lo > best) best = lo;
    }
    return best;
}

/* lower bound for D over the cell; early exit once >= stop */
static double cell_bound(double X[][3], const double *E, double stop) {
    double best = 1.0 / N, w[3], m[3];
    for (int t = 0; t < NT3 + NP2; t++) {
        if (t < NT3) {
            const double *a = X[T3[t][0]], *b = X[T3[t][1]], *c = X[T3[t][2]];
            double u0 = b[0] - a[0], u1 = b[1] - a[1], u2 = b[2] - a[2];
            double v0 = c[0] - a[0], v1 = c[1] - a[1], v2 = c[2] - a[2];
            w[0] = u1 * v2 - u2 * v1; w[1] = u2 * v0 - u0 * v2; w[2] = u0 * v1 - u1 * v0;
        } else {
            const double *a = X[P2[t - NT3][0]], *b = X[P2[t - NT3][1]];
            w[0] = a[0] + b[0]; w[1] = a[1] + b[1]; w[2] = a[2] + b[2];
        }
        double n = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (n < 1e-300) continue;
        for (int s = 0; s < 2; s++) {
            double sg = s ? -1.0 : 1.0;
            m[0] = sg * w[0]; m[1] = sg * w[1]; m[2] = sg * w[2];
            double v = axis_bound(m, X, E);
            if (v > best) { best = v; if (best - BND_SLACK >= stop) return best - BND_SLACK; }
        }
    }
    return best - BND_SLACK;
}

static int outside_domain(const double *q, const double *h, double X[][3], const double *E) {
    for (int i = 2; i < N - 1; i++) {
        int a = 2 * i - 3, b = 2 * (i + 1) - 3;
        if (q[a] - h[a] > q[b] + h[b]) return 1;
    }
    double th1_lo = q[0] - h[0];
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (i == 0 && j == 1) continue;
            if (angle(X[i], X[j]) + E[i] + E[j] + ANG_SLACK < th1_lo) return 1;
        }
    return 0;
}

static int inside_ball(double X[][3], const double *E) {
    for (int b = 0; b < NBALLS; b++) {
        double mx = 0;
        for (int i = 0; i < N; i++) { double d = angle(X[i], BALL_Z[b][i]) + E[i] + ANG_SLACK; if (d > mx) mx = d; }
        if (mx < BALL_R[b]) return 1;
    }
    return 0;
}

static double SPLIT_ALPHA = 0.0; /* >0: prefer splitting earlier points (incremental placement) */
#ifndef RIGOROUS
static inline int coord_point(int k) { return k == 0 ? 1 : (k + 3) / 2; }
#endif
static void split(const Cell *c, Cell *a, Cell *b) {
    int j = 0; double wmax = -1;
    for (int k = 0; k < NQ; k++) {
        double wk = c->h[k];
        if (k >= 2 && (k % 2 == 0)) wk = c->h[k] * (fabs(sin(c->q[k - 1])) + 1e-3); /* phi coordinate */
        if (SPLIT_ALPHA > 0) wk /= (1.0 + SPLIT_ALPHA * (coord_point(k) - 1));
        if (wk > wmax) { wmax = wk; j = k; }
    }
    *a = *c; *b = *c;
    a->h[j] = b->h[j] = c->h[j] / 2;
    a->q[j] = c->q[j] - c->h[j] / 2;
    b->q[j] = c->q[j] + c->h[j] / 2;
}

typedef struct { long processed, certified, sym, ball, surv; double min_surv_lb; } Stats;

/* DFS on one root; survivors appended to fp (thread-private) */
static void dfs(const Cell *root, Stats *st, FILE *fp, long max_cells) {
    size_t cap = 4096, top = 0;
    Cell *stack = malloc(cap * sizeof(Cell));
    stack[top++] = *root;
#ifndef RIGOROUS
    double X[NMAX][3], E[NMAX];
#else
    double X[1][3];
#endif
    long local = 0;
    while (top) {
        Cell c = stack[--top];
        st->processed++; local++;
#ifdef RIGOROUS
        CellR R; prep_r(&c, &R);
        if (outside_domain_r(&c, &R)) { st->sym++; continue; }
        double lb = cell_bound_r(&c, &R, TGT);
        (void)X;
#else
        points(c.q, X); motion(c.q, c.h, E);
        if (outside_domain(c.q, c.h, X, E)) { st->sym++; continue; }
        if (NBALLS && inside_ball(X, E)) { st->ball++; continue; }
        double lb = cell_bound(X, E, TGT);
#endif
        if (lb >= TGT) { st->certified++; continue; }
#ifdef RIGOROUS
        const double *E = c.E;
#endif
        double emax = 0; for (int i = 0; i < N; i++) if (E[i] > emax) emax = E[i];
        if (emax < MIN_EPS || (max_cells && local > max_cells)) {
            st->surv++;
            if (lb < st->min_surv_lb) st->min_surv_lb = lb;
            int keep = 1;
            if (KEEP_PROB < 1.0) { /* deterministic hash of the cell centre */
                unsigned long long hsh = 1469598103934665603ULL; const unsigned char *p = (const unsigned char *)c.q;
                for (size_t k = 0; k < NQ * sizeof(double); k++) { hsh ^= p[k]; hsh *= 1099511628211ULL; }
                keep = (double)(hsh >> 11) * (1.0 / 9007199254740992.0) < KEEP_PROB;
            }
            if (keep) { fwrite(c.q, sizeof(double), NQ, fp); fwrite(c.h, sizeof(double), NQ, fp); fwrite(&lb, sizeof(double), 1, fp); }
            continue;
        }
        if (top + 2 > cap) { cap *= 2; stack = realloc(stack, cap * sizeof(Cell)); }
#ifdef RIGOROUS
        split_r(&c, &stack[top], &stack[top + 1]); top += 2;
#else
        split(&c, &stack[top], &stack[top + 1]); top += 2;
#endif
    }
    free(stack);
}

static int load_balls(const char *fn) {
    FILE *f = fopen(fn, "r"); if (!f) { perror(fn); return -1; }
    int nb; if (fscanf(f, "%d", &nb) != 1) return -1;
    BALL_Z = malloc(nb * sizeof(*BALL_Z)); BALL_R = malloc(nb * sizeof(double));
    for (int b = 0; b < nb; b++) {
        if (fscanf(f, "%lf", &BALL_R[b]) != 1) return -1;
        for (int i = 0; i < N; i++) for (int k = 0; k < 3; k++) if (fscanf(f, "%lf", &BALL_Z[b][i][k]) != 1) return -1;
    }
    fclose(f); NBALLS = nb; return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s N T MIN_EPS OUT [--threads K] [--frontier F] [--balls FILE] [--max-cells-per-root M]\n", argv[0]);
        return 1;
    }
    N = atoi(argv[1]); TGT = atof(argv[2]); MIN_EPS = atof(argv[3]); const char *out = argv[4];
    int threads = 0; long frontier = 0, maxc = 0; const char *balls = NULL;
    for (int a = 5; a < argc; a++) {
        if (!strcmp(argv[a], "--threads")) threads = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--frontier")) frontier = atol(argv[++a]);
        else if (!strcmp(argv[a], "--balls")) balls = argv[++a];
        else if (!strcmp(argv[a], "--max-cells-per-root")) maxc = atol(argv[++a]);
        else if (!strcmp(argv[a], "--keep-prob")) KEEP_PROB = atof(argv[++a]);
        else if (!strcmp(argv[a], "--split-alpha")) SPLIT_ALPHA = atof(argv[++a]);
    }
    if (N < 3 || N > NMAX) { fprintf(stderr, "N must be in 3..%d\n", NMAX); return 1; }
    NQ = 2 * N - 3; NT3 = 0; NP2 = 0;
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
        P2[NP2][0] = i; P2[NP2][1] = j; NP2++;
        for (int k = j + 1; k < N; k++) { T3[NT3][0] = i; T3[NT3][1] = j; T3[NT3][2] = k; NT3++; }
    }
    if (balls && load_balls(balls)) return 1;
    if (getenv("CAPCERT_EVAL")) {   /* testing: read cells (q,h,lb doubles) from file, print bound and sym flag */
        FILE *ef = fopen(getenv("CAPCERT_EVAL"), "rb"); Cell c; double dummy;
#ifdef RIGOROUS
        init_trig();
#endif
        while (fread(c.q, sizeof(double), NQ, ef) == (size_t)NQ && fread(c.h, sizeof(double), NQ, ef) == (size_t)NQ
               && fread(&dummy, sizeof(double), 1, ef) == 1) {
#ifdef RIGOROUS
            init_cache(&c); CellR R; prep_r(&c, &R);
            printf("%.17g %d\n", cell_bound_r(&c, &R, 1e9), outside_domain_r(&c, &R));
#else
            double X[NMAX][3], E[NMAX]; points(c.q, X); motion(c.q, c.h, E);
            printf("%.17g %d\n", cell_bound(X, E, 1e9), outside_domain(c.q, c.h, X, E));
#endif
        }
        return 0;
    }
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    int nthr = omp_get_max_threads();
#else
    int nthr = 1;
#endif
    if (!frontier) frontier = 2000L * nthr;

    /* deterministic initial frontier: breadth-first splits */
    Cell root;
#ifdef RIGOROUS
    if (balls) { fprintf(stderr, "--balls is not available in rigorous mode (use the arb refiner)\n"); return 1; }
    init_trig(); root_r(&root);
#else
    root.q[0] = M_PI / 2; root.h[0] = M_PI / 2;
    for (int i = 2; i < N; i++) {
        root.q[2 * i - 3] = M_PI / 2; root.h[2 * i - 3] = M_PI / 2;
        if (i == 2) { root.q[2] = M_PI / 2; root.h[2] = M_PI / 2; }
        else { root.q[2 * i - 2] = 0; root.h[2 * i - 2] = M_PI; }
    }
#endif
    size_t nf = 1, capf = 1024; Cell *F = malloc(capf * sizeof(Cell)); F[0] = root;
    while ((long)nf < frontier) {
        Cell *G = malloc(2 * nf * sizeof(Cell));
        for (size_t i = 0; i < nf; i++)
#ifdef RIGOROUS
            split_r(&F[i], &G[2 * i], &G[2 * i + 1]);
#else
            split(&F[i], &G[2 * i], &G[2 * i + 1]);
#endif
        free(F); F = G; nf *= 2;
    }
    /* checkpoint */
    char fn[4096];
    snprintf(fn, sizeof fn, "%s.progress", out);
    char *done = calloc(nf, 1);
    Stats tot = {0, 0, 0, 0, 0, INFINITY};
    FILE *pf = fopen(fn, "r");
    if (pf) {
        long idx, p, c, s, b, v; double ml;
        while (fscanf(pf, "%ld %ld %ld %ld %ld %ld %lf", &idx, &p, &c, &s, &b, &v, &ml) == 7) {
            if (idx >= 0 && (size_t)idx < nf && !done[idx]) {
                done[idx] = 1; tot.processed += p; tot.certified += c; tot.sym += s; tot.ball += b; tot.surv += v;
                if (ml < tot.min_surv_lb) tot.min_surv_lb = ml;
            }
        }
        fclose(pf);
    }
    long ndone = 0; for (size_t i = 0; i < nf; i++) ndone += done[i];
    fprintf(stderr, "capcert%s N=%d T=%.17g MIN_EPS=%g threads=%d frontier=%zu (resuming %ld done) balls=%d\n",
#ifdef RIGOROUS
            " (RIGOROUS)",
#else
            "",
#endif
            N, TGT, MIN_EPS, nthr, nf, ndone, NBALLS);
    pf = fopen(fn, "a");
    time_t t0 = time(NULL), tlast = t0;
    long finished = ndone;

#pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        char sfn[4200]; snprintf(sfn, sizeof sfn, "%s.surv.%d", out, tid);
        FILE *sf = fopen(sfn, "ab");
#pragma omp for schedule(dynamic, 1)
        for (long r = 0; r < (long)nf; r++) {
            if (done[r]) continue;
            Stats st = {0, 0, 0, 0, 0, INFINITY};
            dfs(&F[r], &st, sf, maxc);
            fflush(sf);
#pragma omp critical
            {
                fprintf(pf, "%ld %ld %ld %ld %ld %ld %.17g\n", r, st.processed, st.certified, st.sym, st.ball, st.surv,
                        st.min_surv_lb);
                fflush(pf);
                tot.processed += st.processed; tot.certified += st.certified; tot.sym += st.sym; tot.ball += st.ball;
                tot.surv += st.surv; if (st.min_surv_lb < tot.min_surv_lb) tot.min_surv_lb = st.min_surv_lb;
                finished++;
                time_t now = time(NULL);
                if (now - tlast >= 60 || finished == (long)nf) {
                    tlast = now;
                    fprintf(stderr, "[%6lds] roots %ld/%zu  cells %ld  certified %ld  sym %ld  ball %ld  survivors %ld  min_surv_lb %.12g\n",
                            (long)(now - t0), finished, nf, tot.processed, tot.certified, tot.sym, tot.ball, tot.surv,
                            tot.min_surv_lb);
                }
            }
        }
        fclose(sf);
    }
    fclose(pf);
    printf("RESULT N=%d T=%.17g MIN_EPS=%g cells=%ld certified=%ld sym=%ld ball=%ld survivors=%ld min_surv_lb=%.17g time=%lds\n",
           N, TGT, MIN_EPS, tot.processed, tot.certified, tot.sym, tot.ball, tot.surv, tot.min_surv_lb, (long)(time(NULL) - t0));
    #ifdef RIGOROUS
    printf("MODE rigorous (interval arithmetic, outward rounding)\n");
#endif
    if (tot.surv == 0) printf("CERTIFIED: D >= T on the whole fundamental domain%s\n", NBALLS ? " minus the exclusion balls" : "");
    return 0;
}
