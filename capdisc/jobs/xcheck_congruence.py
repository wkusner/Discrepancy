"""Second-kernel check of the congruence of the boxed images (proof/congruence.py), in mpmath interval arithmetic.

For every image k of the minimizer, jobs/xcheck_sym.py proves (mpmath.iv) that the symmetric system has a unique
zero in a box Z_k, enclosed in K_k = K(Z_k), and that this zero is the image's minimizer. Here we prove that all
images are the same configuration: for each k there is an isometry sigma of the symmetric family (N=5: the
identity; N=6: sign changes of (a, b, c), the reflections in the coordinate planes, which conjugate the
symmetry group into itself) with sigma(K_k) inside the interior of Z_0. Since the zero in Z_0 is unique,
sigma(s*_k) = s*_0, so Y(s*_k) is congruent to Y(s*_0). The choice of sigma is numerical; the containment is
decided in mpmath.iv on the exact endpoints stored by xcheck_sym.py. Independent of Arb/python-flint.
usage: python jobs/xcheck_congruence.py NAME [--dir DIR]   reads DIR/xcheck_sym_<NAME>_img*.json (default
certificates/), writes DIR/xcheck_congruence_<NAME>.json
"""
import sys, os, json, glob, itertools
from mpmath import mp, iv
mp.dps = 50; iv.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
CERT = sys.argv[sys.argv.index('--dir') + 1] if '--dir' in sys.argv else os.path.join(HERE, '..', 'certificates')
if '--dir' in sys.argv: args.remove(CERT)
name = args[0]
files = sorted(glob.glob(os.path.join(CERT, f'xcheck_sym_{name}_img*.json')))
certs = [json.load(open(f)) for f in files]
assert certs and all('s_enclosure' in c for c in certs), 'run jobs/xcheck_sym.py for every image first'
exact = lambda t: mp.make_mpf(tuple(t))
box = lambda pair: iv.mpf([exact(pair[0]), exact(pair[1])])
c0 = next(c for c in certs if c['image'] == 0)
Z0 = [box(z) for z in c0['s_box']]
k = len(Z0) - 1                                   # the last coordinate is D
sigmas = [(1,) * k] if name.startswith('N5') else list(itertools.product((1, -1), repeat=k))
out = []
for c in certs:
    K = [box(x) for x in c['s_enclosure']]
    best = min(sigmas, key=lambda sg: sum(abs(float(mp.mpf((sg[i] * K[i]).mid)) - float(mp.mpf(Z0[i].mid))) for i in range(k)))
    sK = [best[i] * K[i] for i in range(k)] + [K[k]]
    inside = all((sK[i].a > Z0[i].a) is True and (sK[i].b < Z0[i].b) is True for i in range(k + 1))
    out.append(dict(image=c['image'], sigma=list(best), contained_in_Z0=bool(inside), verified_image=bool(c['verified'])))
    print(f"{name} image {c['image']}: sigma {best}, sigma(K) inside Z_0: {inside}")
ok = all(o['contained_in_Z0'] and o['verified_image'] for o in out)
res = dict(name=name, images=out, congruent=ok, kernel='mpmath.iv (50 digits)', label='Krawczyk-verified (mpmath.iv)',
           independent_of='arb/python-flint and proof/')
json.dump(res, open(os.path.join(CERT, f'xcheck_congruence_{name}.json'), 'w'), indent=1)
print('ALL IMAGES CONGRUENT (mpmath.iv)' if ok else 'NOT VERIFIED')
