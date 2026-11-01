"""jobs/identify_poly.py -- numerical search for the minimal polynomial of u = N D* (NOT A PROOF; labels: numerical).

The best known configuration of the atlas export (observation #0) is polished by Newton's method on the 2N-2
value equations f_b(q) = s (sphcap/highprec.py::polish_values) to DPS digits, and for each degree d in DEGREES
an LLL reduction of the lattice [1, u, ..., u^d | 10^(DPS-30) u^i] (python-flint) looks for an integer relation.
A relation is reported only if it is far shorter than a generic lattice vector (height^(d+1) << 10^(DPS-30)) and
a factor of it vanishes to DPS-80 digits; otherwise the line says "generic" and gives the height a relation would
have to beat.  For N = 5, 6 this finds P_N (degree 12, symmetry groups of order 4 and 6); for N >= 7 the best
configurations have only a C2 symmetry (or none), and the degree may be in the hundreds or more.

Degrees run in parallel (one process each).  Precision must grow with the degree: roughly
DPS > (d + 1) * log10(height) / 0.7, e.g. d = 144 with heights 10^40 needs DPS ~ 8,500.

usage: python jobs/identify_poly.py N DPS DEGREES [--workers K]      e.g.  10 9000 48,72,96,120,144 --workers 5
writes results/identify_N<N>_dps<DPS>.json
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import mpmath as mp
import flint
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, ROOT)
from sphcap import balance as B, highprec as H

sys.set_int_max_str_digits(0)     # Python >= 3.11 limits int <-> str conversion to 4300 digits by default

U = None; DPS = None


def init(u_str, dps):
    global U, DPS
    mp.mp.dps = dps; U = mp.mpf(u_str); DPS = dps


def search(d):
    t = time.time(); digits = DPS - 30; scale = mp.mpf(10) ** digits
    pw = [mp.mpf(1)]
    for _ in range(d): pw.append(pw[-1] * U)
    rows = [[1 if j == i else 0 for j in range(d + 1)] + [int(mp.nint(pw[i] * scale))] for i in range(d + 1)]
    M = flint.fmpz_mat(rows).lll()
    c = [int(M[0, i]) for i in range(d + 1)]
    logh = float(mp.log10(max(abs(v) for v in c) + 1))
    res = dict(degree=d, height_log10=round(logh, 2), generic_height_log10=round(digits / (d + 1), 2), seconds=None)
    p = flint.fmpz_poly(c)
    if p.degree() > 0:
        for f, _ in p.factor()[1]:
            fv = abs(mp.fsum(mp.mpf(int(f[i])) * pw[i] for i in range(f.degree() + 1)))
            if fv < mp.mpf(10) ** (-(DPS - 80)) and (f.degree() + 1) * logh < 0.7 * digits:
                res['candidate'] = [int(f[i]) for i in range(f.degree() + 1)]
                res['candidate_degree'] = f.degree()
    res['seconds'] = round(time.time() - t, 1)
    return res


if __name__ == '__main__':
    N, dps = int(sys.argv[1]), int(sys.argv[2]); degs = [int(x) for x in sys.argv[3].split(',')]
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else len(degs)
    obs = json.load(open(os.path.join(ROOT, f'export/sphere/d3/cap_discrepancy_sup/n{N}/observations.json')))['observations'][0]
    t0 = time.time(); os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    ufile = os.path.join(ROOT, 'results', f'identify_N{N}_u_dps{dps}.txt')     # the polish is the slow part: keep it
    mp.mp.dps = dps
    if os.path.exists(ufile):
        u = mp.mpf(open(ufile).read().split()[0]); resid = 'from ' + os.path.basename(ufile)
    else:
        r = H.polish_values(B.normalize(np.array(obs['X'])), dps=dps, maxsteps=200)
        mp.mp.dps = dps; u = N * r['D']; resid = mp.nstr(r['residual'], 3)
        open(ufile, 'w').write(mp.nstr(u, dps) + '\n')
        print(f"redundant caps residual {mp.nstr(r['extra_residual'], 3)}", flush=True)
    print(f"N={N} {obs['name']}: N D to {dps} digits ({time.time()-t0:.0f}s; residual {resid}); N D = {mp.nstr(u, 30)}", flush=True)
    out = dict(N=N, name=obs['name'], dps=dps, u=mp.nstr(u, 60), polish_residual=resid, status='numerical', results=[])
    with Pool(workers, initializer=init, initargs=(mp.nstr(u, dps), dps)) as pool:
        for res in pool.imap_unordered(search, degs):
            out['results'].append(res)
            print(f"  degree {res['degree']}: " + (f"CANDIDATE of degree {res['candidate_degree']}" if 'candidate' in res else
                  f"generic (height 10^{res['height_log10']}, a relation would need < 10^{res['generic_height_log10']})")
                  + f"  [{res['seconds']}s]", flush=True)
    json.dump(out, open(os.path.join(ROOT, 'results', f'identify_N{N}_dps{dps}.json'), 'w'), indent=1)
