"""Add the stuck-point class (sphcap/stuck.py) to the atlas export (NOT TRUSTED; numerical labels, level 0).

For every observation in export/sphere/d3/cap_discrepancy_sup/n<N>/observations.json this adds

stuck_point   {class, flat_dim (if any), level 0, status numerical, script, note}; the classes are those of
              sphcap/stuck.py: sharp / strict / non-strict minimum, saddle, exit, not stationary.
              For an exit it also records where the exit leads: the descent restarted along the exit
              direction (follow_exits), the value reached, its class, and the observation of this export it
              coincides with: same value to 1e-8 and the same sorted list of pairwise inner products to 1e-6
              (invariant under rotations, reflections and relabelling; a numerical match, not a proof of
              congruence).

Records are not removed: an exit is a point the descent stopped at, which is itself a fact about the search, and
the atlas can filter on stuck_point.class.

usage: python jobs/atlas_stuck.py N [N ...]
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, ROOT)
from sphcap.stuck import classify, follow_exits
from sphcap import balance as B


def block(X, name):
    c = classify(X, rng=np.random.default_rng(0))
    out = dict(level=0, status='numerical', script='sphcap/stuck.py', **{'class': c['cls']})
    if 'flat_dim' in c: out['flat_dim'] = c['flat_dim']
    if c['cls'] == 'exit':
        D0, _ = B.active_set(X)
        Y, D1 = follow_exits(X, D0)
        c1 = classify(Y, rng=np.random.default_rng(0))
        out['exit'] = dict(rate=round(float(c['escape_rate']), 10), switched_points=c['switched'],
                           leads_to=dict(value=float(D1), **{'class': c1['cls']}), _Y=Y)
    return out


def shape(X):
    X = B.normalize(np.asarray(X, float)); G = X @ X.T
    return np.sort(G[np.triu_indices(len(X), 1)])


if __name__ == '__main__':
    for N in map(int, sys.argv[1:]):
        f = os.path.join(ROOT, f'export/sphere/d3/cap_discrepancy_sup/n{N}/observations.json')
        doc = json.load(open(f)); obs = doc['observations']
        for o in obs:
            o['stuck_point'] = block(B.normalize(np.array(o['X'])), o['name'])
        for o in obs:
            e = o['stuck_point'].get('exit')
            if not e: continue
            Y = e.pop('_Y')
            for p in obs:
                if p is not o and abs(p['value'][0]['value'] - e['leads_to']['value']) < 1e-8 \
                        and np.abs(shape(p['X']) - shape(Y)).max() < 1e-6:
                    e['leads_to']['same_as'] = p['name']; break
        json.dump(doc, open(f, 'w'), indent=1)
        for o in obs:
            s = o['stuck_point']
            print(f"N={N} {o['name']:<22} {o['value'][0]['value']:.10f} {s['class']}"
                  + (f" -> {s['exit']['leads_to']}" if 'exit' in s else ''), flush=True)
