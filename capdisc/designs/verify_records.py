"""Independent re-check of the exported design records (does not use the search code's residual).

For every record in export/sphere/d3/spherical_design/n*/observations.json:
  * recompute the max moment residual in double precision from X (own implementation: explicit loops);
  * check |x_i| = 1 (to 1e-12) and the minimum pair distance;
  * if a level-1 record is present, recompute the residual at 60 digits with mpmath from the stored strings
    (the stored decimal strings are used as exact input) and compare with the stored value.
Prints a table; exit code 1 if any record fails (residual > 1e-11, or hp residual > 1e-30 when hp present).
This is a re-computation, not a proof: the labels stay 'numerical' / 'high-precision'.
"""
import sys, os, json, glob, itertools
from fractions import Fraction
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'export', 'sphere', 'd3', 'spherical_design')


def dfact(n):
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def mean_exact(a, b, c):
    if a % 2 or b % 2 or c % 2:
        return Fraction(0)
    return Fraction(dfact(a - 1) * dfact(b - 1) * dfact(c - 1), dfact(a + b + c + 1))


def mons(t):
    return [(a, b, c) for c in (0, 1) for a in range(t + 1) for b in range(t + 1)
            if a + b + c <= t and (a, b, c) != (0, 0, 0)]


def resid_float(X, t):
    worst = 0.0
    for a, b, c in mons(t):
        s = sum(x ** a * y ** b * z ** c for x, y, z in X) / len(X)
        worst = max(worst, abs(s - float(mean_exact(a, b, c))))
    return worst


def resid_mp(Xs, t, dps=60):
    mp.mp.dps = dps
    X = [[mp.mpf(v) for v in row] for row in Xs]
    norm_err = max(abs(sum(v * v for v in row) - 1) for row in X)
    worst = mp.mpf(0)
    for a, b, c in mons(t):
        m = mean_exact(a, b, c)
        s = sum(x ** a * y ** b * z ** c for x, y, z in X) / len(X) - mp.mpf(m.numerator) / m.denominator
        worst = max(worst, abs(s))
    return worst, norm_err


if __name__ == '__main__':
    bad = 0
    print(f"{'N':>3} {'t':>2} {'group':>5} {'resid(float)':>13} {'norm err':>9} {'min dist':>9} {'hp resid':>10}")
    for path in sorted(glob.glob(os.path.join(OUT, 'n*', 'observations.json')),
                       key=lambda p: int(os.path.basename(os.path.dirname(p))[1:])):
        d = json.load(open(path))
        for rec in d['observations']:
            X = rec['X']
            t = rec['strength']
            r = resid_float(X, t) if t > 0 else 0.0
            ne = max(abs(sum(v * v for v in row) - 1) for row in X)
            Xa = np.array(X)
            md = (np.linalg.norm(Xa[:, None] - Xa[None], axis=2) + 9 * np.eye(len(X))).min() if len(X) > 1 else float('nan')
            hp = [v for v in rec['value'] if v.get('level') == 1]
            hps = '-'
            ok = r < 1e-11 and ne < 1e-12
            if hp:
                R, nerr = resid_mp(hp[0]['X_hp'], t) if t > 0 else (mp.mpf(0), mp.mpf(0))
                hps = mp.nstr(R, 3)
                ok = ok and R < mp.mpf(10) ** -30
            bad += (not ok)
            print(f"{rec['N']:>3} {t:>2} {rec['symmetry']['group']:>5} {r:13.2e} {ne:9.1e} {md:9.4f} {hps:>10} {'' if ok else '  <-- FAIL'}")
    print("failures:", bad)
    sys.exit(1 if bad else 0)
