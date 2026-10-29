/*
 * ia.h / rig.h -- rigorous mode of capcert (compile with -DRIGOROUS -ffp-contract=off, no -ffast-math)
 *
 * Interval arithmetic with outward rounding.  IEEE 754 binary64 +, -, *, /, sqrt are correctly
 * rounded (round-to-nearest), so the exact result lies within one ulp of the computed one; every
 * operation below moves the lower end down and the upper end up by one ulp (nup/ndn), which
 * therefore encloses the exact result.  No libm function is used: sin and cos are Taylor
 * polynomials evaluated in interval arithmetic plus the Lagrange remainder |x|^(n+1)/(n+1)!.
 *
 * Cell bound (same mathematics as the floating-point mode, without angles):
 *   cell = box c +- h in q (exact real endpoints of the stored doubles); x_i the point at c,
 *   E_i >= max over the cell of the geodesic distance d(y_i, x_i)  (E_i = h_th + |sin th_c| h_ph).
 *   For an axis w (any nonzero double vector), cos d_i = <w, x_i>/|w| is enclosed in an interval,
 *   and with cE_i, sE_i enclosures of cos E_i, sin E_i:
 *     c_i <= cos(min(d_i + E_i, pi))  : cd*cE - sd*sE  if  cd.lo + cE.lo >= 0 (so d_i + E_i <= pi), else -1
 *     C_i >= cos(max(d_i - E_i, 0))   : cd*cE + sd*sE  if  cd.hi <= cE.lo    (so d_i >= E_i),      else 1
 *   For every y in the cell, <y_i, w/|w|> >= c_i and <= C_i.  With c_(k), C_(k) the k-th largest:
 *     closed cap {<y,w> >= c_(k)} contains >= k points:   D >= k/N - (1 - c_(k))/2
 *     open cap   {<y,w> >  C_(k)} contains <= k-1 points: D >= (1 - C_(k))/2 - (k-1)/N
 *   and D >= 1/N.
 * Covering: a split cell [m -+ h'] uses h' = h/2 + (|q|+h) 2^-52 (rounded up), which covers the
 * rounding of the new centre m = fl(q -+ h/2); root boxes are widened by one ulp so that they
 * contain [0, pi] (theta) and [-pi, pi] (phi).
 */
#include <stdint.h>
#include <float.h>

typedef struct { double lo, hi; } I;

static inline double nup(double x) {
    if (x != x || x == INFINITY) return x;
    if (x == 0) return DBL_TRUE_MIN;
    union { double d; int64_t i; } u = {x};
    if (x > 0) u.i++; else u.i--;
    return u.d;
}
static inline double ndn(double x) { return -nup(-x); }
static inline I ipt(double x) { I r = {x, x}; return r; }
static inline I iadd(I a, I b) { I r = {ndn(a.lo + b.lo), nup(a.hi + b.hi)}; return r; }
static inline I isub(I a, I b) { I r = {ndn(a.lo - b.hi), nup(a.hi - b.lo)}; return r; }
static inline I ineg(I a) { I r = {-a.hi, -a.lo}; return r; }
static inline I imul(I a, I b) {
    double p1 = a.lo * b.lo, p2 = a.lo * b.hi, p3 = a.hi * b.lo, p4 = a.hi * b.hi;
    double lo = fmin(fmin(p1, p2), fmin(p3, p4)), hi = fmax(fmax(p1, p2), fmax(p3, p4));
    I r = {ndn(lo), nup(hi)}; return r;
}
static inline I ismul(double s, I a) { /* exact double s times interval */
    I r = s >= 0 ? (I){ndn(s * a.lo), nup(s * a.hi)} : (I){ndn(s * a.hi), nup(s * a.lo)}; return r;
}
static inline I isqr(I a) {
    double l = fabs(a.lo), h = fabs(a.hi), mn = fmin(l, h), mx = fmax(l, h);
    if (a.lo <= 0 && a.hi >= 0) mn = 0;
    I r = {mn == 0 ? 0 : ndn(mn * mn), nup(mx * mx)}; return r;
}
static inline I idiv(I a, I b) { /* requires 0 < b.lo */
    double p1 = a.lo / b.lo, p2 = a.lo / b.hi, p3 = a.hi / b.lo, p4 = a.hi / b.hi;
    double lo = fmin(fmin(p1, p2), fmin(p3, p4)), hi = fmax(fmax(p1, p2), fmax(p3, p4));
    I r = {ndn(lo), nup(hi)}; return r;
}
static inline I isqrt_(I a) { I r = {a.lo <= 0 ? 0 : ndn(sqrt(a.lo)), nup(sqrt(fmax(a.hi, 0)))}; if (r.lo < 0) r.lo = 0; return r; }

/* ---------------- sin / cos of a double by Taylor polynomials with remainder */
#define TK 17                       /* terms; remainder <= 4^34/34! < 1e-17 for |x| <= 4 */
static I COSC[TK], SINC[TK];        /* (-1)^k/(2k)!, (-1)^k/(2k+1)! */
static double RC4, RS4;             /* upper bounds of 4^(2TK)/(2TK)!, 4^(2TK+1)/(2TK+1)! */
static inline double rem_bound(double ax, int n) { /* upper bound of ax^n/n! */
    double r = 1.0;
    for (int m = 1; m <= n; m++) r = nup(nup(r * ax) / m);
    return r;
}
static void init_trig(void) {
    I f = ipt(1.0);                  /* enclosure of 1/m! */
    for (int m = 0; m < 2 * TK + 2; m++) {
        if (m > 0) f = idiv(f, ipt((double)m));
        int k = m / 2; double sg = (k % 2) ? -1.0 : 1.0;
        if (m < 2 * TK) { if (m % 2 == 0) COSC[k] = ismul(sg, f); else SINC[k] = ismul(sg, f); }
    }
    RC4 = rem_bound(4.0, 2 * TK); RS4 = rem_bound(4.0, 2 * TK + 1);
}
static void isincos(double x, I *s, I *c) {
    double ax = fabs(x), rc, rs;
    if (ax <= 4.0) { rc = RC4; rs = RS4; }   /* t^n/n! is increasing in t >= 0 */
    else { rc = rem_bound(ax, 2 * TK); rs = rem_bound(ax, 2 * TK + 1); }
    I X = ipt(x), Y = isqr(X), pc = COSC[TK - 1], ps = SINC[TK - 1];
    for (int k = TK - 2; k >= 0; k--) { pc = iadd(COSC[k], imul(Y, pc)); ps = iadd(SINC[k], imul(Y, ps)); }
    ps = imul(X, ps);
    c->lo = ndn(pc.lo - rc); c->hi = nup(pc.hi + rc);
    s->lo = ndn(ps.lo - rs); s->hi = nup(ps.hi + rs);
    if (c->lo < -1) c->lo = -1;
    if (c->hi > 1) c->hi = 1;
    if (s->lo < -1) s->lo = -1;
    if (s->hi > 1) s->hi = 1;
}

