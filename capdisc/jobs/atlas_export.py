"""Export configurations as observations for critconfig-atlas (spec v0.2, with the additions for the
cap-discrepancy functional proposed in the mapping report: a `functional` tag, the active hypergraph
in place of the contact graph, and the weight space in place of the stress space).

For each configuration X (N points on S^2):

functional    {"id": "cap_discrepancy_sup", ...}: D(X) = sup over closed caps C of |#(X n C)/N - sigma(C)|,
              minimized.  Every record carries it; the graph hash includes it.
active        the active hypergraph.  Vertices: the N points.  Hyperedges: the worst caps, each written
              as the *over-full closed cap* C (#C/N - sigma(C) = D; an under-full open cap is replaced by
              its complement, which is over-full and closed), as {boundary, interior}: the points on the
              boundary circle and the points strictly inside.  Caps that coincide (four or more points on
              one worst circle) are one hyperedge; `branches` lists the point sets that define it
              (2 points: the smallest cap through them; 3 points: the circle through them).
weights       the weight space W = {lambda >= 0, sum lambda = 1, sum_b lambda_b grad f_b = 0} over the
              branches (the KKT multipliers; the analogue of the stress polytope): its dimension, the
              support (branches positive somewhere on W), the range [min, max] of each lambda_b on W,
              the max-min point, and the dimension of the signed space {sum lambda_b grad f_b = 0}.
key_basis     the canonical code of the *support hypergraph* (hyperedges with positive weight somewhere
              on W; the analogue of the stressed graph), canonical under relabelling of the points
              (individualization-refinement); graph_hash = first 12 hex of SHA-256 of
              "sphere|d3|cap_discrepancy_sup|N<N>|<code>".  The ordinal is minted by the atlas.
first_order   rank of the active gradients, dimension 2N-3 of the quotient, kernel dimension, sharp.
second_order  only when the kernel is not 0: eigenvalues of the Lagrangian Hessian (finite differences,
              max-min multipliers) on the kernel modulo rotations; counts of negative (descending),
              positive and near-zero directions.

Levels follow the atlas: 0 numerical (all of the above unless stated), 2 interval-verified, 3 exact
(test-verified).  The minimizers N5_best and N6_best get level-2 and level-3 descriptions of the value
and a level-3 Gram matrix by reference (certificates/gramcert_*.json); their active hypergraph is level 2
(counts frozen on the box of the local verification, all other caps below D*).

usage: python jobs/atlas_export.py [EXTRA.json ...]   (extra inputs: outputs of search_parallel.py or
       atlas.py, lists or {"classes": [...]} of records with "points")
writes export/sphere/d3/cap_discrepancy_sup/n<N>/observations.json
"""
import sys, os, json, hashlib, subprocess, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, ROOT)
import numpy as np
from scipy.optimize import linprog
from sphcap import balance as B

FUNCTIONAL = dict(
    id='cap_discrepancy_sup',
    definition='D(X) = sup over closed spherical caps C of |#(X n C)/N - sigma(C)|, sigma the normalized area',
    sense='minimize', container='sphere', d=3,
    critical_point='balanced: a nonnegative combination of the gradients of the worst caps vanishes (Clarke stationary)',
    gauge='x0 = (0,0,1), x1 in the xz-plane with x > 0 (coordinates are stored as found; the gauge applies to q)')
LEVEL_LABEL = {0: 'numerical', 1: 'high-precision', 2: 'interval-verified', 3: 'exact (test-verified)'}
TOL = 1e-9


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def commit():
    try:
        c = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.', ':(exclude)export'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return c + ('+dirty' if dirty else '')
    except Exception:
        return None


# ------------------------------------------------------------------ active hypergraph
def hyperedges(X, act):
    """Merge the active branches into caps, each as the over-full closed cap {boundary, interior}."""
    caps = {}
    for k, b in enumerate(act):
        if b.kind == 'const':
            key = ('point',); caps.setdefault(key, dict(boundary=[], interior=[], point_cap=True, branches=[]))
            caps[key]['branches'].append(k); continue
        proj = X @ b.w
        on = np.abs(proj - b.t) <= TOL
        inside = proj > b.t + TOL if b.kind == 'up' else proj < b.t - TOL
        key = (tuple(np.flatnonzero(on)), tuple(np.flatnonzero(inside)))
        e = caps.setdefault(key, dict(boundary=[int(i) for i in key[0]], interior=[int(i) for i in key[1]], branches=[]))
        e['branches'].append(k)
    return list(caps.values())


# ------------------------------------------------------------------ canonical code (individualization-refinement)
def _refine(col, edges, N):
    while True:
        sig = []
        for v in range(N):
            inc = []
            for (Bd, In) in edges:
                if v in Bd or v in In:
                    role = 0 if v in Bd else 1
                    inc.append((role, tuple(sorted((0 if u in Bd else 1, col[u]) for u in Bd + In))))
            sig.append((col[v], tuple(sorted(inc))))
        ranks = {s: r for r, s in enumerate(sorted(set(sig)))}
        new = [ranks[s] for s in sig]
        if len(set(new)) == len(set(col)):
            return new
        col = new


def _code(order, edges):
    lab = {v: i for i, v in enumerate(order)}
    E = sorted((tuple(sorted(lab[u] for u in Bd)), tuple(sorted(lab[u] for u in In))) for Bd, In in edges)
    return ';'.join(','.join(map(str, b)) + '/' + ','.join(map(str, i)) for b, i in E)


def canonical_code(N, edges):
    edges = [(tuple(Bd), tuple(In)) for Bd, In in edges]
    best = [None]

    def search(col):
        col = _refine(col, edges, N)
        if len(set(col)) == N:
            c = _code(sorted(range(N), key=lambda v: col[v]), edges)
            if best[0] is None or c < best[0]: best[0] = c
            return
        cells = {}
        for v, c in enumerate(col): cells.setdefault(c, []).append(v)
        target = min((c for c in cells if len(cells[c]) > 1), key=lambda c: (len(cells[c]), c))
        for v in cells[target]:
            new = [2 * c + (1 if (c == target and u != v) else 0) for u, c in enumerate(col)]
            search(new)

    search([0] * N)
    return best[0]


# ------------------------------------------------------------------ weight space
def weight_space(G, names):
    m = len(G)
    A_eq = np.vstack([G.T, np.ones((1, m))]); b_eq = np.append(np.zeros(G.shape[1]), 1.0)
    rng = []; his = []
    for b in range(m):
        c = np.zeros(m); c[b] = 1
        lo = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * m, method='highs')
        hi = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * m, method='highs')
        if lo.status != 0 or hi.status != 0:
            return dict(nonempty=False)
        rng.append([float(lo.fun), float(-hi.fun)]); his.append(hi.x)
    supp = [b for b in range(m) if rng[b][1] > 1e-9]
    GS = G[supp]
    Mtop = np.vstack([GS.T, np.ones((1, len(supp)))])
    sv = np.linalg.svd(Mtop, compute_uv=False)
    rk = int(np.sum(sv > 1e-7 * sv.max()))
    # max-min point over the support
    ms = len(supp)
    c = np.zeros(ms + 1); c[-1] = -1
    lp = linprog(c, A_ub=np.hstack([-np.eye(ms), np.ones((ms, 1))]), b_ub=np.zeros(ms),
                 A_eq=np.hstack([Mtop, np.zeros((Mtop.shape[0], 1))]), b_eq=np.append(np.zeros(G.shape[1]), 1.0),
                 bounds=[(0, None)] * ms + [(None, None)], method='highs')
    note = None
    if lp.status == 0:
        lam = np.zeros(m); lam[supp] = lp.x[:ms]; mmv = float(lp.x[-1])
    else:   # numerically infeasible at HiGHS tolerances: use the centroid of the per-branch maximizers (a point of W, positive on the support)
        lam = np.mean([his[b] for b in supp], axis=0); mmv = float(min(lam[b] for b in supp))
        note = 'max-min LP failed numerically; centroid of the per-branch maximizers instead (a point of W, not the max-min point)'
    svG = np.linalg.svd(G, compute_uv=False)
    rkG = int(np.sum(svG > 1e-7 * svG.max()))
    return dict(nonempty=True, dim=len(supp) - rk, support=[names[b] for b in supp],
                ranges={names[b]: [round(r[0], 12), round(r[1], 12)] for b, r in enumerate(rng)},
                maxmin={names[b]: round(float(lam[b]), 12) for b in range(m)}, maxmin_value=mmv, **({'maxmin_note': note} if note else {}),
                signed_dim=m - rkG, _lam=lam, _supp=supp)


# ------------------------------------------------------------------ one record
def observation(X, name, source, extra=None):
    X = B.normalize(np.asarray(X, float)); N = len(X)
    D, act = B.active_set(X)
    act = [b for b in act if b.kind != 'const'] or act
    names = [f"{'up' if b.kind == 'up' else 'low'}:{'-'.join(map(str, b.I))}" for b in act]
    G, E = B.gradient_matrix(X, act)
    W = weight_space(G, names)
    edges = hyperedges(X, act)
    for e in edges:
        e['branches'] = [names[k] for k in e['branches']]
        if W.get('nonempty'):
            e['weight_range'] = [round(sum(W['ranges'][n][0] for n in e['branches']), 12),
                                 round(sum(W['ranges'][n][1] for n in e['branches']), 12)]
            e['in_support'] = any(n in W['support'] for n in e['branches'])
        e['count'] = len(e['boundary']) + len(e['interior'])
    full_code = canonical_code(N, [(e['boundary'], e['interior']) for e in edges if not e.get('point_cap')])
    supp_edges = [(e['boundary'], e['interior']) for e in edges if e.get('in_support') and not e.get('point_cap')]
    if W.get('nonempty'):
        code = canonical_code(N, supp_edges) if supp_edges else ''
        kb_method = 'support-hypergraph canonical code (individualization-refinement)'
    else:   # no balance at this tolerance (not first-order stationary): key on the full active hypergraph
        code = full_code
        kb_method = 'active-hypergraph canonical code (weight space empty at this tolerance)'
    sv = np.linalg.svd(G, compute_uv=False); rank = int(np.sum(sv > 1e-7 * sv.max()))
    dimq = 2 * N - 3; kdim = dimq - rank
    # sharp (first-order growth mod rotations): some balance with every lambda_b > 0, full rank, no coincident caps
    sharp = bool(W.get('nonempty') and len(W['support']) == len(act) and W['maxmin_value'] > 1e-9 and rank == dimq
                 and not any(len(e['branches']) > 1 for e in edges))
    rec = dict(
        kind='find', functional=FUNCTIONAL['id'], container='sphere', d=3, N=N, name=name,
        X=X.tolist(), gauge=FUNCTIONAL['gauge'],
        value=[dict(id='v0', value=float(D), level=0, status=LEVEL_LABEL[0], note='double precision')],
        active=dict(level=0, status=LEVEL_LABEL[0], tolerance=TOL, n_hyperedges=len(edges), hyperedges=edges,
                    code=full_code),
        weights=dict(level=0, status=LEVEL_LABEL[0], **{k: v for k, v in W.items() if not k.startswith('_')}),
        key_basis=dict(method=kb_method, code=code,
                       graph_hash=hashlib.sha256(f"sphere|d3|{FUNCTIONAL['id']}|N{N}|{code}".encode()).hexdigest()[:12]),
        first_order=dict(rank=rank, dim_quotient=dimq, kernel_dim=kdim, sharp=sharp, stationary=bool(W.get('nonempty')),
                         multipliers_unique=bool(W.get('nonempty') and W['dim'] == 0), level=0, status=LEVEL_LABEL[0]),
        source=source)
    if kdim > 0 and W.get('nonempty'):
        H = B.lagrangian_hessian(X, act, W['_lam'])
        R = B.rotation_generators(X, E)
        _, svK, Vt = np.linalg.svd(np.vstack([G, R]))   # first-order motions keeping all active caps, mod rotations
        svK = np.concatenate([svK, np.zeros(Vt.shape[0] - len(svK))])
        K = Vt[svK <= 1e-7 * svK.max()].T                 # same relative threshold as the rank
        ev = np.linalg.eigvalsh(K.T @ H @ K) if K.shape[1] else np.array([])
        scale = max(1.0, float(np.abs(ev).max())) if len(ev) else 1.0
        rec['second_order'] = dict(
            level=0, status=LEVEL_LABEL[0], kernel_dim=int(K.shape[1]),
            hessian_eigenvalues_on_kernel=[round(float(v), 8) for v in ev],
            descending=int(np.sum(ev < -1e-5 * scale)), ascending=int(np.sum(ev > 1e-5 * scale)),
            null=int(np.sum(np.abs(ev) <= 1e-5 * scale)),
            note='finite-difference Hessian (h = 1e-4) of sum lambda_b f_b at the max-min multipliers'
                 + ('; the weight space has positive dimension, so the result may depend on that choice' if W['dim'] > 0 else ''))
    if extra: rec.update(extra)
    return rec


def upgrades(name):
    """Level-2/3 descriptions for the proved minimizers, from the certificate files."""
    cert = os.path.join(ROOT, 'certificates')
    gc, lc = os.path.join(cert, f'gramcert_{name}.json'), os.path.join(cert, f'localcert_{name}.json')
    if not (os.path.exists(gc) and os.path.exists(lc)): return {}
    g, l = json.load(open(gc)), json.load(open(lc))
    if not (g.get('verified') and l.get('verified')): return {}
    ge = json.load(open(os.path.join(ROOT, 'sphcap/data/gram_exact.json')))[name]
    enc = l['images'][0]['D_enclosure']
    ref = lambda p: dict(file=p, file_sha256=sha(os.path.join(ROOT, p)))
    return dict(
        value_extra=[
            dict(id='v2', level=2, status=LEVEL_LABEL[2], value={'interval': [repr(enc[0]), repr(enc[1])]},
                 note='Krawczyk-verified (Arb), cross-verified with mpmath.iv',
                 evidence=[ref(f'certificates/localcert_{name}.json'), ref(f'certificates/xcheck_local_{name}.json')]),
            dict(id='v3', level=3, status=LEVEL_LABEL[3],
                 value=dict(minpoly=[str(c) for c in ge['D_minpoly_ascending']], isolating_interval=g['D_isolating_interval'],
                            closed_form=None, decimal=ge['D'][:40]),
                 note='minimal polynomial = P_N(N x) up to a constant; one root in the isolating interval',
                 evidence=[ref('certificates/xcheck_roots.json'), ref(f'certificates/gramcert_{name}.json')])],
        gram=dict(level=3, status=LEVEL_LABEL[3], ref=dict(ref('sphcap/data/gram_exact.json'), pointer=f'/{name}'),
                  cross_checks=[dict(kernel=k) for k in g['kernels'].split('; ')],
                  evidence=[ref(f'certificates/gramcert_{name}.json')],
                  note='entries in Q(D) as (sum coeffs[i] D^i)/denom, labelled as the X of this record (checked '
                       'numerically by the exporter); the certificate works in the labelling of the local verification, '
                       'its field relabelling maps one to the other'),
        active_level=2)


def main():
    data = json.load(open(os.path.join(ROOT, 'sphcap/data/critical_configurations.json')))['classes']
    src_cc = dict(file='sphcap/data/critical_configurations.json', script='jobs/atlas_export.py', commit=commit())
    items = [(c['points'], c['name'], dict(src_cc, pointer=c['name'])) for c in data]
    for f in sys.argv[1:]:
        d = json.load(open(f)); rows = d['classes'] if isinstance(d, dict) else d
        for k, r in enumerate(rows):
            if 'points' in r:
                items.append((r['points'], f"{os.path.basename(f).split('.')[0]}#{k}",
                              dict(file=os.path.relpath(f, ROOT), file_sha256=sha(f), pointer=k, script='jobs/atlas_export.py', commit=commit())))
    byN = {}
    for X, name, src in items:
        rec = observation(X, name, src)
        up = upgrades(name)
        if up:
            ge = json.load(open(os.path.join(ROOT, 'sphcap/data/gram_exact.json')))[name]
            vals = [float(e['value']) for e in ge['entries']]
            Gn = np.array([[1.0 if i == j else vals[ge['gram_pattern'][i][j]] for j in range(rec['N'])] for i in range(rec['N'])])
            Xr = np.array(rec['X'])
            assert np.allclose(Xr @ Xr.T, Gn, atol=1e-9), f'{name}: labelling of gram_exact.json differs from X'
            rec['value'] += up['value_extra']; rec['gram'] = up['gram']
            rec['active']['level'] = up['active_level']; rec['active']['status'] = LEVEL_LABEL[2]
            rec['active']['note'] = ('counts of the worst caps frozen on the box of the local verification and every '
                                     'other cap below D* there (Arb, cross-verified with mpmath.iv)')
        byN.setdefault(rec['N'], []).append(rec)
        w = rec['weights']
        print(f"{name}: N={rec['N']} D={rec['value'][0]['value']:.10f} caps={rec['active']['n_hyperedges']} "
              f"W dim={w.get('dim')} support={len(w.get('support', []))} sharp={rec['first_order']['sharp']} "
              f"kernel={rec['first_order']['kernel_dim']} hash={rec['key_basis']['graph_hash']} "
              f"levels={[v['level'] for v in rec['value']]}", flush=True)
    for N, recs in byN.items():
        seen = {}
        for r in recs:   # flag shared support hypergraphs inside this export (the atlas mints ordinals)
            seen.setdefault(r['key_basis']['graph_hash'], []).append(r['name'])
        out = os.path.join(ROOT, 'export', 'sphere', 'd3', FUNCTIONAL['id'], f'n{N}')
        os.makedirs(out, exist_ok=True)
        json.dump(dict(functional=FUNCTIONAL, N=N, observations=recs,
                       shared_hashes={h: v for h, v in seen.items() if len(v) > 1}),
                  open(os.path.join(out, 'observations.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
