"""explore/make_hints.py -- NOT TRUSTED. Selects the input of the local verification (PROOF.md Step 4):
the images of the minimizer in the fundamental domain, and for each image its worst caps with counts, split
into a square system (2N-2 caps) and redundant caps. Everything here is a *choice*: a wrong choice makes the
verification fail (a missed image leaves a cell the refinement cannot close; a wrong cap or count makes the
Krawczyk or frozen-count test fail), it cannot make a false statement pass.

usage: python explore/make_hints.py N5_best|N6_best        writes proof/hints/<NAME>.json
"""
import sys, os, json
import numpy as np
import scipy.linalg as sl
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, ROOT)
from sphcap import balance as B, highprec as H, certify as Cf

name = sys.argv[1]
d = {c['name']: c for c in json.load(open(os.path.join(ROOT, 'sphcap/data/critical_configurations.json')))['classes']}
X = np.array(d[name]['points']); N = len(X); nq = 2 * N - 3


def worst_caps(Y):                       # as in jobs/local_cert.py (phase-1 refactor: unchanged selection)
    q0 = np.array(H.to_q(Y), float); rep = B.balance_report(Y); caps, counts = [], []
    for br in rep.active:
        I = tuple(sorted(br.I)); nC = br.n_cl if br.kind == 'up' else N - br.n_op
        if len(I) == 2:
            caps.append(Cf.CapSpec(I)); counts.append(nC); continue
        best = None
        for side in (1, -1):
            cs = Cf.CapSpec(I, side)
            dat = Cf.cap_data(q0, np.full(nq, 1e-13), N, [cs])[0]
            if dat is None or dat[3] != nC: continue
            val = float(dat[0].mid())
            if best is None or abs(val - rep.D) < abs(best[1] - rep.D): best = (cs, val)
        assert best is not None and abs(best[1] - rep.D) < 1e-9, (I, best)
        caps.append(best[0]); counts.append(nC)
    return q0, caps, counts


images = []
for k, (Yi, qi) in enumerate(Cf.domain_images(X)):
    q0, caps, counts = worst_caps(Yi); extra, extra_counts = [], []
    dq = (q0 - qi + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(dq, 0, atol=1e-9), (q0, qi)
    q0 = np.array(qi, float)
    if len(caps) != 2 * N - 2:           # redundant worst caps: keep an independent square subset
        J = np.array([np.append(Cf.cap_data(q0, np.full(nq, 1e-13), N, [cs])[0][1], -1.0) for cs in caps])
        _, _, piv = sl.qr(J.T, pivoting=True)
        keep = sorted(piv[:2 * N - 2])
        extra = [caps[i] for i in range(len(caps)) if i not in keep]
        extra_counts = [counts[i] for i in range(len(caps)) if i not in keep]
        caps = [caps[i] for i in keep]; counts = [counts[i] for i in keep]
    images.append(dict(image=k, q=[float(x).hex() for x in q0], caps=[repr(c) for c in caps], counts=counts,
                       extra_caps=[repr(c) for c in extra], extra_counts=extra_counts))
    print(f"{name} image {k}: {len(caps)} + {len(extra)} worst caps")
os.makedirs(os.path.join(ROOT, 'proof/hints'), exist_ok=True)
json.dump(dict(name=name, N=N, note='selection only (explore/make_hints.py); not trusted', images=images),
          open(os.path.join(ROOT, f'proof/hints/{name}.json'), 'w'), indent=1)
