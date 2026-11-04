/*
 * capcert.c -- branch-and-bound lower bounds for the spherical cap discrepancy
 *
 * Proves  D(X) >= T  for every configuration X of N points on S^2 (N <= 10),
 * cell by cell over the gauge- and symmetry-reduced parameter box, and writes
 * the cells it could not bound ("survivors") once they are smaller than MIN_EPS.
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
 * are skipped (they are covered by a separate local verification).
 *
 * Parallel: the domain is split deterministically into an initial frontier; OpenMP
 * threads take frontier cells dynamically and run an explicit-stack DFS on each.
 * Checkpoint: one line per finished frontier cell in <out>.progress; rerunning with
 * the same arguments skips finished cells.
 *
 * Build:  gcc -O3 -march=native -fopenmp -o capcert capcert.c -lm
 * Usage:  capcert N T MIN_EPS OUT [--threads K] [--frontier F] [--balls FILE]
 *                 [--max-cells-per-root M] [--keep-prob P]
 *         capcert_rig N T MIN_EPS OUT [--threads K] [--frontier F] [--first-order] [--fo-emax E] [--smear]
 *                 [--cells PREFIX]          (second pass: start from the survivors PREFIX.surv.*)
 *         capcert_rig --version             (provenance baked in by build.sh)
 * --keep-prob P writes only a (hash-determined) fraction P of the survivor cells
 * (atlas mode); all survivors are still counted.  The rigorous build refuses --keep-prob,
 * --max-cells-per-root and --balls: each changes what a survivor file means.
 *
 * Completion stamp OUT.done (one line of key=value tokens; written to OUT.done.tmp, then renamed): the run's
 * parameters and counts, the SHA-256 of every survivor file OUT.surv.<t> (read back after they are closed, so a
 * resumed run is covered), and the provenance of the binary (commit, tree state, SHA-256 of the C sources, compiler
 * version, flags; build.sh passes them as -D macros).  The second pass also records the SHA-256 of the first
 * pass's stamp and of every input file it read.  Values that may contain blanks are %-encoded (%20, %25).
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NMAX 10
#define QMAX (2 * NMAX - 3)
#define KMAX (2 * (120 + 45)) /* 2*(C(10,3)+C(10,2)) */
#define ANG_SLACK 1e-12
#define BND_SLACK 1e-12

/* provenance, baked in by build.sh (-D...); a binary built otherwise says "unknown" and the chain checker rejects it */
#ifndef CAPCERT_COMMIT
#define CAPCERT_COMMIT "unknown"
#endif
#ifndef CAPCERT_TREE_DIRTY
#define CAPCERT_TREE_DIRTY "unknown"
#endif
#ifndef CAPCERT_SRC_FILES
#define CAPCERT_SRC_FILES "unknown"
#endif
#ifndef CAPCERT_SRC_SHA
#define CAPCERT_SRC_SHA "unknown"
#endif
#ifndef CAPCERT_FLAGS
#define CAPCERT_FLAGS "unknown"
#endif
#ifdef __VERSION__
#define CAPCERT_CC __VERSION__
#else
#define CAPCERT_CC "unknown"
#endif

static int N, NQ, NT3, NP2;
static int T3[120][3], P2[45][2];
static double TGT, MIN_EPS;
static int FO = 0;            /* rigorous mode: also try the first-order bound (fo.h) on cells where the zeroth-order bound fails */
static double FO_EMAX = 0.5;  /* ... if the largest motion bound of the cell is below this */
static int SMEAR = 0;         /* rigorous mode: split the coordinate with the largest smear (fo.h) */
static double KEEP_PROB = 1.0; /* fraction of survivor cells written to disk (all are counted) */
static int NBALLS = 0;
static double (*BALL_Z)[NMAX][3] = NULL;
static double *BALL_R = NULL;

#include "sha256.h"
#include <libgen.h>
#include <limits.h>

#ifdef RIGOROUS
#include "ia.h"
typedef struct { double q[QMAX], h[QMAX]; I sq[QMAX], cq[QMAX]; double E[NMAX]; I sE[NMAX], cE[NMAX]; } Cell;
#include "rig.h"
#include "fo.h"
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

typedef struct { long processed, bounded, sym, ball, surv; double min_surv_lb; long fo; } Stats;
static volatile int IO_ERROR = 0;
static void *xmalloc(size_t n) { void *p = malloc(n); if (!p) { fprintf(stderr, "out of memory\n"); exit(3); } return p; }

/* ---------------------------------------------------------------- completion stamps */
/* a value with blanks or '%' %-encoded, so that the stamp stays one line of whitespace-free key=value tokens */
static void put_enc(FILE *f, const char *s) {
    for (; *s; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch <= ' ' || ch == '%' || ch >= 127) fprintf(f, "%%%02X", ch); else fputc(ch, f);
    }
}

static void put_provenance(FILE *f) {
    fprintf(f, " commit=%s tree_dirty=%s src=%s src_sha256=%s cc=", CAPCERT_COMMIT, CAPCERT_TREE_DIRTY, CAPCERT_SRC_FILES, CAPCERT_SRC_SHA);
    put_enc(f, CAPCERT_CC); fputs(" flags=", f); put_enc(f, CAPCERT_FLAGS);
#ifdef RIGOROUS
    fputs(" build=rigorous", f);
#else
    fputs(" build=float", f);
#endif
}

/* the survivor files PREFIX.surv.0 .. PREFIX.surv.<n-1>: they must be contiguous (a gap would make the readers stop
   early) and hold whole records.  Returns n >= 0, or -1 (gap), -2 (partial record), -3 (too many files). */
static int surv_count(const char *prefix, long rec) {
    int last = -1, count = 0;
    for (int t = 0; t < 4096; t++) {
        char sfn[4200]; struct stat stt; snprintf(sfn, sizeof sfn, "%s.surv.%d", prefix, t);
        if (stat(sfn, &stt) != 0) continue;
        if (stt.st_size % rec) return -2;
        last = t; count++;
    }
    { char sfn[4200]; struct stat stt; snprintf(sfn, sizeof sfn, "%s.surv.4096", prefix); if (stat(sfn, &stt) == 0) return -3; }
    return count == last + 1 ? count : -1;
}

static const char *base_of(const char *p) { const char *b = strrchr(p, '/'); return b ? b + 1 : p; }

/* hash every survivor file of OUT (after they are closed) and append " files=n records=R surv.<t>=<sha256> ..." */
static int put_surv_hashes(FILE *f, const char *out, long rec) {
    int n = surv_count(out, rec);
    if (n < 0) { fprintf(stderr, "%s.surv.*: %s; no stamp written\n", out, n == -1 ? "gap in the file numbers" : n == -2 ? "partial record" : "too many files"); return -1; }
    long long records = 0; char (*hx)[65] = xmalloc((n + 1) * sizeof *hx);
    for (int t = 0; t < n; t++) {
        char sfn[4200]; long long nb; snprintf(sfn, sizeof sfn, "%s.surv.%d", out, t);
        if (sha256_file(sfn, hx[t], &nb)) { perror(sfn); free(hx); return -1; }
        records += nb / rec;
    }
    fprintf(f, " files=%d records=%lld", n, records);
    for (int t = 0; t < n; t++) fprintf(f, " surv.%d=%s", t, hx[t]);
    free(hx); return 0;
}

/* write the stamp atomically: OUT.done.tmp, then rename */
static FILE *stamp_open(const char *out, char *tmp, size_t ntmp) { snprintf(tmp, ntmp, "%s.done.tmp", out); return fopen(tmp, "w"); }
static int stamp_close(FILE *df, const char *out, const char *tmp) {
    char dfn[4200]; snprintf(dfn, sizeof dfn, "%s.done", out);
    fputc('\n', df);
    if (fflush(df) || ferror(df) || fsync(fileno(df)) || fclose(df)) { perror(tmp); return -1; }
    if (rename(tmp, dfn)) { perror(dfn); return -1; }
    return 0;
}

#ifdef RIGOROUS
/* do the directories of two prefixes resolve to the same directory? */
static int same_dir(const char *a, const char *b) {
    char ca[4200], cb[4200], ra[PATH_MAX], rb[PATH_MAX];
    snprintf(ca, sizeof ca, "%s", a); snprintf(cb, sizeof cb, "%s", b);
    if (!realpath(dirname(ca), ra) || !realpath(dirname(cb), rb)) return 0;
    return strcmp(ra, rb) == 0;
}
#endif

/* DFS on one root; survivors appended to fp (thread-private) */
static void dfs(const Cell *root, Stats *st, FILE *fp, long max_cells) {
    size_t cap = 4096, top = 0;
    Cell *stack = xmalloc(cap * sizeof(Cell));
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
        if (lb >= TGT) { st->bounded++; continue; }
#ifdef RIGOROUS
        const double *E = c.E;
#endif
        double emax = 0; for (int i = 0; i < N; i++) if (E[i] > emax) emax = E[i];
#ifdef RIGOROUS
        int have_smear = 0;
        if (FO && emax < FO_EMAX) {
            double lb1 = first_order_bound_r(&c, &R); have_smear = 1;
            if (lb1 >= TGT) { st->bounded++; st->fo++; continue; }
            if (lb1 > lb) lb = lb1;
        }
#endif
        if (emax < MIN_EPS || (max_cells && local > max_cells)) {
            st->surv++;
            if (lb < st->min_surv_lb) st->min_surv_lb = lb;
            int keep = 1;
            if (KEEP_PROB < 1.0) { /* deterministic hash of the cell centre */
                unsigned long long hsh = 1469598103934665603ULL; const unsigned char *p = (const unsigned char *)c.q;
                for (size_t k = 0; k < NQ * sizeof(double); k++) { hsh ^= p[k]; hsh *= 1099511628211ULL; }
                keep = (double)(hsh >> 11) * (1.0 / 9007199254740992.0) < KEEP_PROB;
            }
            if (keep) { if (fwrite(c.q, sizeof(double), NQ, fp) != (size_t)NQ || fwrite(c.h, sizeof(double), NQ, fp) != (size_t)NQ
                              || fwrite(&lb, sizeof(double), 1, fp) != 1) IO_ERROR = 1; }
            continue;
        }
        if (top + 2 > cap) { cap *= 2; stack = realloc(stack, cap * sizeof(Cell)); if (!stack) { fprintf(stderr, "out of memory\n"); exit(3); } }
#ifdef RIGOROUS
        if (SMEAR && have_smear) {
            int j = 0; for (int k = 1; k < NQ; k++) if (FO_SMEAR[k] > FO_SMEAR[j]) j = k;
            split_rj(&c, &stack[top], &stack[top + 1], j);
        } else split_r(&c, &stack[top], &stack[top + 1]);
        top += 2;
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

#ifdef RIGOROUS
/* Second pass (--cells PREFIX): every cell of PREFIX.surv.0, .1, ... (records q, h, lb as written by a first pass)
   is processed like a root cell, in chunks of CHUNK cells; survivors go to OUT.surv.<thread>.  Progress is kept per
   chunk in OUT.progress (header + "chunk processed bounded sym ball survivors min_lb" lines).  The input files are
   read in full on every run (reading precedes the skip of a finished chunk), and hashed while they are read; the
   completion stamp OUT.done records the SHA-256 of the first pass's stamp PREFIX.done (from_stamp=), of every input
   file (in.<t>=), the number of cells read, whether PREFIX and OUT are in the same directory, and the hashes of its
   own survivor files.  If PREFIX.done lists survivor hashes (surv.<t>=), the inputs must match them. */
#define CHUNK (1L << 20)
static int run_cells(const char *prefix, const char *out, long maxc, int nthr) {
    long rec = (long)((2 * NQ + 1) * sizeof(double));
    long fsize[4096]; long total = 0;
    if (same_dir(prefix, out) && !strcmp(base_of(prefix), base_of(out))) { fprintf(stderr, "--cells: input and output prefix are the same\n"); return 1; }
    int nfiles = surv_count(prefix, rec);
    if (nfiles < 0) { fprintf(stderr, "%s.surv.*: %s\n", prefix, nfiles == -1 ? "gap in the file numbers" : nfiles == -2 ? "partial record" : "too many files"); return 1; }
    for (int t = 0; t < nfiles; t++) {
        char sfn[4200]; struct stat stt; snprintf(sfn, sizeof sfn, "%s.surv.%d", prefix, t);
        if (stat(sfn, &stt) != 0) { perror(sfn); return 1; }
        fsize[t] = stt.st_size / rec; total += fsize[t];
    }
    /* the first pass's stamp: it must exist (the first pass completed); its hash goes into our stamp */
    char pdone[4200], from_sha[65]; snprintf(pdone, sizeof pdone, "%s.done", prefix);
    if (sha256_file(pdone, from_sha, NULL)) { fprintf(stderr, "%s: missing (the first pass did not complete); refusing to run\n", pdone); return 1; }
    char (*want)[65] = calloc(nfiles + 1, sizeof *want); int have_want = 0;     /* survivor hashes listed in PREFIX.done */
    {
        FILE *pd = fopen(pdone, "r"); char tok[512];
        while (pd && fscanf(pd, "%511s", tok) == 1) {
            int t; char hx[65];
            if (sscanf(tok, "surv.%d=%64[0-9a-f]", &t, hx) == 2) {
                if (t < 0 || t >= nfiles) { fprintf(stderr, "%s lists surv.%d, but there are %d input files\n", pdone, t, nfiles); return 1; }
                memcpy(want[t], hx, 65); have_want++;
            }
        }
        if (pd) fclose(pd);
        if (have_want && have_want != nfiles) { fprintf(stderr, "%s lists %d survivor files, found %d\n", pdone, have_want, nfiles); return 1; }
    }
    long nchunks = (total + CHUNK - 1) / CHUNK;
    char pfn[4200], hdr[1024]; snprintf(pfn, sizeof pfn, "%s.progress", out);
    snprintf(hdr, sizeof hdr, "#capcert-cells N=%d T=%a MIN_EPS=%a from=%s files=%d cells=%ld chunk=%ld mode=rigorous fo=%d fo_emax=%a smear=%d maxc=%ld\n",
             N, TGT, MIN_EPS, prefix, nfiles, total, CHUNK, FO, FO_EMAX, SMEAR, maxc);
    char *done = calloc(nchunks + 1, 1); Stats tot = {0, 0, 0, 0, 0, INFINITY, 0};
    FILE *pf = fopen(pfn, "r"); int have_header = 0; long resumed = 0;
    if (pf) {
        char line[1024] = "";
        if (!fgets(line, sizeof line, pf) || strcmp(line, hdr)) { fprintf(stderr, "refusing to resume: %s has other settings\n", pfn); return 2; }
        have_header = 1; long idx, p, cc, s, b, v; double ml;
        while (fscanf(pf, "%ld %ld %ld %ld %ld %ld %lf", &idx, &p, &cc, &s, &b, &v, &ml) == 7)
            if (idx >= 0 && idx < nchunks && !done[idx]) { done[idx] = 1; resumed++; tot.processed += p; tot.bounded += cc; tot.sym += s;
                tot.ball += b; tot.surv += v; if (ml < tot.min_surv_lb) tot.min_surv_lb = ml; }
        fclose(pf);
    }
    for (int t = 0; t < 4096; t++) {   /* survivor files of an interrupted run may end in a partial record */
        char sfn[4200]; struct stat stt; snprintf(sfn, sizeof sfn, "%s.surv.%d", out, t);
        if (stat(sfn, &stt) == 0 && stt.st_size % rec && truncate(sfn, stt.st_size - stt.st_size % rec)) { perror(sfn); return 2; }
    }
    pf = fopen(pfn, "a"); if (!pf) { perror(pfn); return 2; }
    if (!have_header) { fputs(hdr, pf); fflush(pf); }
    fprintf(stderr, "capcert (RIGOROUS) second pass N=%d T=%.17g MIN_EPS=%g threads=%d: %ld cells from %d files of %s, %ld chunks\n",
            N, TGT, MIN_EPS, nthr, total, nfiles, prefix, nchunks);
    Cell *F = xmalloc(CHUNK * sizeof(Cell)); FILE **sf = calloc(nthr, sizeof(FILE *));
    for (int t = 0; t < nthr; t++) { char sfn[4200]; snprintf(sfn, sizeof sfn, "%s.surv.%d", out, t); sf[t] = fopen(sfn, "ab"); if (!sf[t]) IO_ERROR = 1; }
    Sha256 *hin = xmalloc((nfiles + 1) * sizeof(Sha256)); for (int t = 0; t < nfiles; t++) sha256_init(&hin[t]);
    time_t t0 = time(NULL), tlast = t0; long finished = 0, nread = 0; for (long c = 0; c < nchunks; c++) finished += done[c];
    int fi = 0; long fpos = 0; FILE *in = NULL;           /* sequential reader over the input files */
    for (long c = 0; c < nchunks && !IO_ERROR; c++) {
        long n = 0;
        while (n < CHUNK && fi < nfiles) {
            if (!in) { char sfn[4200]; snprintf(sfn, sizeof sfn, "%s.surv.%d", prefix, fi); in = fopen(sfn, "rb"); fpos = 0; if (!in) { perror(sfn); return 2; } }
            if (fpos == fsize[fi]) { fclose(in); in = NULL; fi++; continue; }
            double lb; Cell *x = &F[n];
            if (fread(x->q, sizeof(double), NQ, in) != (size_t)NQ || fread(x->h, sizeof(double), NQ, in) != (size_t)NQ
                || fread(&lb, sizeof(double), 1, in) != 1) { fprintf(stderr, "read error\n"); return 2; }
            sha256_update(&hin[fi], x->q, NQ * sizeof(double)); sha256_update(&hin[fi], x->h, NQ * sizeof(double));
            sha256_update(&hin[fi], &lb, sizeof(double));
            fpos++; n++; nread++;
        }
        if (done[c]) continue;
        Stats cs = {0, 0, 0, 0, 0, INFINITY, 0};
#pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            Stats st = {0, 0, 0, 0, 0, INFINITY, 0};
#pragma omp for schedule(dynamic, 64)
            for (long r = 0; r < n; r++) { Cell x = F[r]; init_cache(&x); dfs(&x, &st, sf[tid], maxc); }
            if (fflush(sf[tid])) IO_ERROR = 1;
#pragma omp critical
            { cs.processed += st.processed; cs.bounded += st.bounded; cs.sym += st.sym; cs.surv += st.surv; cs.fo += st.fo;
              if (st.min_surv_lb < cs.min_surv_lb) cs.min_surv_lb = st.min_surv_lb; }
        }
        if (IO_ERROR) break;
        fprintf(pf, "%ld %ld %ld %ld %ld %ld %.17g\n", c, cs.processed, cs.bounded, cs.sym, cs.ball, cs.surv, cs.min_surv_lb); fflush(pf);
        tot.processed += cs.processed; tot.bounded += cs.bounded; tot.sym += cs.sym; tot.surv += cs.surv; tot.fo += cs.fo;
        if (cs.min_surv_lb < tot.min_surv_lb) tot.min_surv_lb = cs.min_surv_lb;
        finished++;
        time_t now = time(NULL);
        if (now - tlast >= 60 || finished == nchunks) { tlast = now;
            fprintf(stderr, "[%6lds] chunks %ld/%ld  cells %ld  bounded %ld  sym %ld  survivors %ld  min_surv_lb %.12g\n",
                    (long)(now - t0), finished, nchunks, tot.processed, tot.bounded, tot.sym, tot.surv, tot.min_surv_lb); }
    }
    if (in) fclose(in);
    for (int t = 0; t < nthr; t++) if (sf[t] && fclose(sf[t])) IO_ERROR = 1;
    fclose(pf);
    if (IO_ERROR) { printf("INCOMPLETE (I/O error)\n"); return 2; }
    printf("RESULT N=%d T=%.17g MIN_EPS=%g cells=%ld bounded=%ld (first-order %ld) sym=%ld ball=0 survivors=%ld min_surv_lb=%.17g time=%lds\n",
           N, TGT, MIN_EPS, tot.processed, tot.bounded, tot.fo, tot.sym, tot.surv, tot.min_surv_lb, (long)(time(NULL) - t0));
    printf("MODE rigorous (interval arithmetic, outward rounding)\n");
    if (finished != nchunks) { printf("INCOMPLETE: %ld of %ld chunks finished\n", finished, nchunks); return 2; }
    /* every input record was read and hashed; the files did not change size meanwhile; the hashes match PREFIX.done */
    if (nread != total) { fprintf(stderr, "read %ld of %ld input cells; no stamp written\n", nread, total); return 2; }
    char (*hx)[65] = xmalloc((nfiles + 1) * sizeof *hx);
    for (int t = 0; t < nfiles; t++) {
        char sfn[4200]; struct stat stt; snprintf(sfn, sizeof sfn, "%s.surv.%d", prefix, t);
        if (stat(sfn, &stt) != 0 || stt.st_size / rec != fsize[t] || stt.st_size % rec) { fprintf(stderr, "%s changed during the run; no stamp written\n", sfn); return 2; }
        sha256_hex(&hin[t], hx[t]);
        if (have_want && strcmp(hx[t], want[t])) { fprintf(stderr, "%s: SHA-256 %s differs from %s (%s); no stamp written\n", sfn, hx[t], pdone, want[t]); return 2; }
    }
    { char sha_now[65]; if (sha256_file(pdone, sha_now, NULL) || strcmp(sha_now, from_sha)) { fprintf(stderr, "%s changed during the run; no stamp written\n", pdone); return 2; } }
    char tmp[4200]; FILE *df = stamp_open(out, tmp, sizeof tmp);
    if (!df) { perror(tmp); return 2; }
    fprintf(df, "N=%d T=%a MIN_EPS=%a nf=%ld mode=rigorous keep=%a balls=%d survivors=%ld cells=%ld fo=%d fo_emax=%a smear=%d maxc=%ld resumed=%ld prefix=%s",
            N, TGT, MIN_EPS, nchunks, KEEP_PROB, NBALLS, tot.surv, tot.processed, FO, FO_EMAX, SMEAR, maxc, resumed, base_of(out));
    fprintf(df, " from=%s same_dir=%d from_stamp=%s input_cells=%ld input_files=%d", base_of(prefix), same_dir(prefix, out), from_sha, total, nfiles);
    for (int t = 0; t < nfiles; t++) fprintf(df, " in.%d=%s", t, hx[t]);
    free(hx);
    if (put_surv_hashes(df, out, rec)) { fclose(df); remove(tmp); return 2; }
    put_provenance(df);
    if (stamp_close(df, out, tmp)) return 2;
    if (tot.surv == 0) printf("INTERVAL-VERIFIED: D >= T on every input cell\n");
    return 0;
}
#endif

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--version")) {      /* provenance of this binary */
        fputs("capcert", stdout); put_provenance(stdout); fputc('\n', stdout); return 0;
    }
    if (argc < 5) {
        fprintf(stderr, "usage: %s N T MIN_EPS OUT [--threads K] [--frontier F] [--balls FILE] [--max-cells-per-root M]\n", argv[0]);
        return 1;
    }
    N = atoi(argv[1]); TGT = atof(argv[2]); MIN_EPS = atof(argv[3]); const char *out = argv[4];
    int threads = 0; long frontier = 0, maxc = 0; const char *balls = NULL, *roots = NULL, *cells_in = NULL;
    for (int a = 5; a < argc; a++) {
        if (!strcmp(argv[a], "--threads")) threads = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--frontier")) frontier = atol(argv[++a]);
        else if (!strcmp(argv[a], "--balls")) balls = argv[++a];
        else if (!strcmp(argv[a], "--max-cells-per-root")) maxc = atol(argv[++a]);
        else if (!strcmp(argv[a], "--keep-prob")) KEEP_PROB = atof(argv[++a]);
        else if (!strcmp(argv[a], "--split-alpha")) SPLIT_ALPHA = atof(argv[++a]);
        else if (!strcmp(argv[a], "--first-order")) FO = 1;
        else if (!strcmp(argv[a], "--roots")) roots = argv[++a];   /* testing only: run these root cells; never complete */
        else if (!strcmp(argv[a], "--fo-emax")) FO_EMAX = atof(argv[++a]);
        else if (!strcmp(argv[a], "--smear")) SMEAR = 1;
        else if (!strcmp(argv[a], "--cells")) cells_in = argv[++a];  /* second pass: start from the cells PREFIX.surv.* */
#ifdef RIGOROUS
        else if (!strcmp(argv[a], "--smear-mix")) SMEAR_MIX = atof(argv[++a]);
#endif
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 1; }
    }
#ifdef RIGOROUS
    /* proof mode: every survivor is written, and a survivor is a cell smaller than MIN_EPS that the bounds did not close */
    if (KEEP_PROB != 1.0) { fprintf(stderr, "--keep-prob is not available in the rigorous build (every survivor must be written)\n"); return 1; }
    if (maxc) { fprintf(stderr, "--max-cells-per-root is not available in the rigorous build (it changes what a survivor is)\n"); return 1; }
#endif
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
            printf("%.17g %d %.17g\n", cell_bound_r(&c, &R, 1e9), outside_domain_r(&c, &R), first_order_bound_r(&c, &R));
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
#ifdef RIGOROUS
    if (cells_in) { init_trig(); return run_cells(cells_in, out, maxc, nthr); }
#else
    if (cells_in) { fprintf(stderr, "--cells needs the rigorous build\n"); return 1; }
#endif

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
    size_t nf = 1, capf = 1024; Cell *F = xmalloc(capf * sizeof(Cell)); F[0] = root;
    while ((long)nf < frontier) {
        Cell *G = xmalloc(2 * nf * sizeof(Cell));
        for (size_t i = 0; i < nf; i++)
#ifdef RIGOROUS
            split_r(&F[i], &G[2 * i], &G[2 * i + 1]);
#else
            split(&F[i], &G[2 * i], &G[2 * i + 1]);
#endif
        free(F); F = G; nf *= 2;
    }
    /* checkpoint.  The progress file starts with a header recording everything that determines the
       frontier and the meaning of a finished root; resuming with different settings is refused. */
    char fn[4096], hdr[512];
    snprintf(fn, sizeof fn, "%s.progress", out);
#ifdef RIGOROUS
    const char *mode = "rigorous";
#else
    const char *mode = "float";
#endif
    snprintf(hdr, sizeof hdr, "#capcert N=%d T=%a MIN_EPS=%a nf=%zu mode=%s keep=%a balls=%d maxc=%ld fo=%d fo_emax=%a smear=%d\n",
             N, TGT, MIN_EPS, nf, mode, KEEP_PROB, NBALLS, maxc, FO, FO_EMAX, SMEAR);
    char *done = calloc(nf, 1);
    if (!done) { fprintf(stderr, "out of memory\n"); return 3; }
    Stats tot = {0, 0, 0, 0, 0, INFINITY, 0};
    FILE *pf = fopen(fn, "r");
    int have_header = 0;
    if (pf) {
        char line[512] = "";
        long pos = ftell(pf);
        if (fgets(line, sizeof line, pf) && line[0] == '#') {
            if (strcmp(line, hdr)) {
                fprintf(stderr, "refusing to resume: %s was written with different settings\n  file: %s  now:  %s", fn, line, hdr);
                return 2;
            }
            have_header = 1;
        } else {
            fseek(pf, pos, SEEK_SET);
#ifdef RIGOROUS
            if (line[0]) { fprintf(stderr, "refusing to resume from a progress file without header (%s)\n", fn); return 2; }
#else
            if (line[0]) fprintf(stderr, "warning: legacy progress file without header; resuming as before\n");
#endif
        }
        long idx, p, cc, s, b, v; double ml;   /* done[i]: 1 finished, 2 skipped (--roots) */
        while (fscanf(pf, "%ld %ld %ld %ld %ld %ld %lf", &idx, &p, &cc, &s, &b, &v, &ml) == 7) {
            if (idx >= 0 && (size_t)idx < nf && !done[idx]) {
                done[idx] = 1; tot.processed += p; tot.bounded += cc; tot.sym += s; tot.ball += b; tot.surv += v;
                if (ml < tot.min_surv_lb) tot.min_surv_lb = ml;
            }
        }
        fclose(pf);
    }
    /* survivor files of an interrupted run may end in a partial record: cut them back */
    {
        long rec = (long)((2 * NQ + 1) * sizeof(double));
        for (int t = 0; t < 4096; t++) {
            char sfn[4200]; struct stat stt;
            snprintf(sfn, sizeof sfn, "%s.surv.%d", out, t);
            if (stat(sfn, &stt) == 0 && stt.st_size % rec) {
                if (truncate(sfn, stt.st_size - stt.st_size % rec)) { perror(sfn); return 2; }
            }
        }
    }
    if (roots) {                        /* testing: skip every root not listed (the run then cannot complete) */
        char *sel = calloc(nf, 1), *buf = strdup(roots);
        for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) { long r = atol(t); if (r >= 0 && (size_t)r < nf) sel[r] = 1; }
        for (size_t i = 0; i < nf; i++) if (!sel[i]) done[i] = 2;
        free(buf); free(sel);
    }
    long ndone = 0; for (size_t i = 0; i < nf; i++) ndone += done[i] == 1;
    fprintf(stderr, "capcert%s N=%d T=%.17g MIN_EPS=%g threads=%d frontier=%zu (resuming %ld done) balls=%d\n",
#ifdef RIGOROUS
            " (RIGOROUS)",
#else
            "",
#endif
            N, TGT, MIN_EPS, nthr, nf, ndone, NBALLS);
    pf = fopen(fn, "a");
    if (!pf) { perror(fn); return 2; }
    if (!have_header && ndone == 0) { fputs(hdr, pf); fflush(pf); }
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
        if (!sf) IO_ERROR = 1;
#pragma omp for schedule(dynamic, 1)
        for (long r = 0; r < (long)nf; r++) {
            if (done[r]) continue;
            Stats st = {0, 0, 0, 0, 0, INFINITY, 0};
            if (IO_ERROR) continue;
            dfs(&F[r], &st, sf, maxc);
            if (fflush(sf)) IO_ERROR = 1;
            if (IO_ERROR) continue;   /* do not record a root whose survivors may be lost */
#pragma omp critical
            {
                fprintf(pf, "%ld %ld %ld %ld %ld %ld %.17g\n", r, st.processed, st.bounded, st.sym, st.ball, st.surv,
                        st.min_surv_lb);
                fflush(pf);
                tot.processed += st.processed; tot.bounded += st.bounded; tot.sym += st.sym; tot.ball += st.ball;
                tot.surv += st.surv; tot.fo += st.fo; if (st.min_surv_lb < tot.min_surv_lb) tot.min_surv_lb = st.min_surv_lb;
                finished++;
                time_t now = time(NULL);
                if (now - tlast >= 60 || finished == (long)nf) {
                    tlast = now;
                    fprintf(stderr, "[%6lds] roots %ld/%zu  cells %ld  bounded %ld  sym %ld  ball %ld  survivors %ld  min_surv_lb %.12g\n",
                            (long)(now - t0), finished, nf, tot.processed, tot.bounded, tot.sym, tot.ball, tot.surv,
                            tot.min_surv_lb);
                }
            }
        }
        if (sf && fclose(sf)) IO_ERROR = 1;
    }
    fclose(pf);
    if (IO_ERROR) { fprintf(stderr, "I/O ERROR writing survivors: run is incomplete\n"); printf("INCOMPLETE (I/O error)\n"); return 2; }
    printf("RESULT N=%d T=%.17g MIN_EPS=%g cells=%ld bounded=%ld (first-order %ld) sym=%ld ball=%ld survivors=%ld min_surv_lb=%.17g time=%lds\n",
           N, TGT, MIN_EPS, tot.processed, tot.bounded, tot.fo, tot.sym, tot.ball, tot.surv, tot.min_surv_lb, (long)(time(NULL) - t0));
    #ifdef RIGOROUS
    printf("MODE rigorous (interval arithmetic, outward rounding)\n");
#endif
    if (finished != (long)nf) { printf("INCOMPLETE: %ld of %zu roots finished\n", finished, nf); return 2; }
    /* completion stamp, read by proof/refine.py, capcert_rig --cells and check/check_chain.py */
    {
        long rec = (long)((2 * NQ + 1) * sizeof(double)); char tmp[4200];
        FILE *df = stamp_open(out, tmp, sizeof tmp);
        if (!df) { perror(tmp); return 2; }
        fprintf(df, "N=%d T=%a MIN_EPS=%a nf=%zu mode=%s keep=%a balls=%d survivors=%ld cells=%ld fo=%d fo_emax=%a smear=%d maxc=%ld resumed=%ld prefix=%s",
                N, TGT, MIN_EPS, nf, mode, KEEP_PROB, NBALLS, tot.surv, tot.processed, FO, FO_EMAX, SMEAR, maxc, ndone, base_of(out));
        if (put_surv_hashes(df, out, rec)) { fclose(df); remove(tmp); return 2; }
        put_provenance(df);
        if (stamp_close(df, out, tmp)) return 2;
    }
    /* labels follow VERIFICATION.md: a single-kernel run is "interval-verified"; "cross-verified" needs a second kernel */
#ifdef RIGOROUS
    if (tot.surv == 0) printf("INTERVAL-VERIFIED: D >= T on the whole fundamental domain\n");
#else
    if (tot.surv == 0) printf("FLOAT-CHECKED: D >= T on the whole fundamental domain%s (floating point with margins; not rigorous)\n", NBALLS ? " minus the exclusion balls" : "");
#endif
    return 0;
}
