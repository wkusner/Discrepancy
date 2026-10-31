"""explore/classify.py -- classify the stuck points of the descent (NOT TRUSTED; numerical labels for the atlas).

A stuck point X is where sequential-LP descent stops: the frozen-count model admits no first-order descent. That
does not yet make X a local minimizer, because the frozen model fixes two things that a nearby configuration may
change:

  (a) assignments: a point on the circle of a worst cap that is not one of its defining points is counted (closed
      cap, branch "up") or not (open complement, branch "low"); moving it off the circle to the other side lowers
      that branch by 1/N, so the branch stops being worst.  A direction may therefore descend by *switching* such a
      point instead of decreasing the branch smoothly.  (The other cap of the same circle is a branch of its own,
      the "low" branch of the same defining set, and is already in the active set when it ties.)
  (b) second order: when the active gradients balance with zero multipliers or do not span the quotient (rank
      < 2N-3), there are flat directions, and the second-order behaviour of the Lagrangian sum lam_b f_b along them
      decides.

Classes:
  sharp minimum       balanced, all multipliers > 0, gradients span: D grows linearly in every direction
  strict minimum      no first-order exit, Lagrangian Hessian positive definite on the flat directions
  non-strict minimum  flat directions along which D stays constant to second order, and probing does not descend
                      (a curve or family of minimizers)
  saddle              a flat direction of negative curvature: D decreases at second order
  exit                a first-order descent direction that switches assignments (a)
  not stationary      a first-order descent direction in the frozen model (the descent was stopped early)
Every non-sharp verdict is also probed: small random perturbations followed by descent must not go lower.

usage: python explore/classify.py N [N ...]      classifies the atlas observations in export/, writes
       explore/classify_N<N>.json and prints a table
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, ROOT)
from sphcap.stuck import classify, aligned_distance, margin, margin_grad, incidences, descent_lp   # noqa: F401


if __name__ == '__main__':
    for N in map(int, sys.argv[1:]):
        f = os.path.join(ROOT, f'export/sphere/d3/cap_discrepancy_sup/n{N}/observations.json')
        obs = json.load(open(f))['observations']
        res = []
        for o in obs:
            c = classify(np.array(o['X']))
            c['name'] = o['name']; res.append(c)
            print(f"N={N} {o['name']:<22} D={c['D']:.10f} |A|={c['n_active']:<3} rank={c['rank']}/{c['dim']} "
                  f"inc={c['incidences']} -> {c['cls']}" + (f" (flat dim {c['flat_dim']})" if 'flat_dim' in c else ''), flush=True)
        json.dump(res, open(os.path.join(HERE, f'classify_N{N}.json'), 'w'), indent=1)
