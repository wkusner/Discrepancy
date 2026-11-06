"""Batch: for each N, search for an N-point design of the strength tau(N) listed in the Hardin-Sloane table,
refine it at high precision, and write atlas observations (spec v0.2 layout, new functional 'spherical_design').

usage: python designs/run_batch.py N_FIRST N_LAST [--tlim 40] [--tneg 15] [--dps 40] [--hpmax 30] [--procs 2]
writes export/sphere/d3/spherical_design/n<N>/observations.json and designs/search_log.json

Labels.  Every record is level 0 (numerical, double precision) with a level-1 (high-precision) refinement when
N <= hpmax.  Nothing is proved.  "found" means a configuration with max moment residual below 1e-12 was
found in the symmetry strata tried; "not found" in the negative search at strength tau+1 is *only* a statement
about this search (strata, starts, time budget), not evidence of non-existence.
"""
import sys, os, json, time, argparse, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sphdesign import search_design, refine_mp, Moments, GROUP_ORDER

ROOT = os.path.join(HERE, '..')
OUT = os.path.join(ROOT, 'export', 'sphere', 'd3', 'spherical_design')
TAU = {r['N']: r for r in json.load(open(os.path.join(HERE, 'literature', 'hs_tau_table.json')))['rows']}

FUNCTIONAL = dict(
    id='spherical_design',
    definition='X (N unit vectors) is a spherical t-design if (1/N) sum_i p(x_i) equals the sphere mean of p for every '
               'polynomial p of degree <= t; observed by the max abs residual over monomials x^a y^b z^c, c in {0,1}, '
               'a+b+c <= t',
    sense='maximize strength t at fixed N (zero set of the moment map, a feasibility problem)',
    container='sphere', d=3,
    critical_point='all harmonic moments of degrees 1..t vanish',
    gauge='none: coordinates are stored as found; a design is defined up to rotation and reflection')

LICENSE = 'data: CC-BY-4.0 (computed in this project); code: MIT'


def build_record(N, t, sol, hp):
    X = sol['X']
    G = X @ X.T
    iu = np.triu_indices(N, 1)
    minang = float(np.degrees(np.arccos(np.clip(G[iu].max(), -1, 1)))) if N > 1 else None
    row = TAU[N]
    vals = [dict(id='v0', level=0, status='numerical', value=float(sol['residual']),
                 note='max abs moment residual, double precision')]
    rec = dict(
        kind='find', functional='spherical_design', container='sphere', d=3, N=N,
        name=f"design_N{N}_t{t}_{sol['signature'][:8]}", strength=t, X=X.tolist(),
        symmetry=dict(group=sol['group'], order=sol['order'], special_orbit_sizes=sol['special'],
                      generic_orbits=sol['generic_orbits'],
                      note='largest symmetry found by this search among the groups tried; the set may have more'),
        invariants=dict(gram_signature=sol['signature'], min_angle_deg=minang),
        provenance=dict(
            coordinates='recomputed in this project (designs/sphdesign.py: random-start least squares on a symmetry '
                        'stratum, then refinement); no coordinates were copied from any other source',
            literature_claim=dict(
                source='hardin_sloane_1996 (tables)', tau_N=row['tau'], proof_code=row['proof_code'],
                group_listed=row['group_coxeter'], status='tau(N) is the authors\' conjectured maximum strength '
                'unless the proof code indicates a proof; table entry transcribed from an automated page summary '
                '(unverified)'),
            equivalence_to_literature='not compared (source coordinate files not downloaded)',
            attribution='existence/strength claims: see designs/literature/spherical_designs_sources.json; '
                        'coordinates computed here'),
        license=LICENSE)
    if hp is not None:
        coords, R = hp
        vals.append(dict(id='v1', level=1, status='high-precision', value=R, dps=hp_dps(),
                         note='max abs moment residual after Gauss-Newton refinement at this precision; '
                              'an approximation of a design, not a proof of existence',
                         X_hp=coords))
    rec['value'] = vals
    return rec


_DPS = [40]


def hp_dps():
    return _DPS[0]


def work(args):
    N, tlim, tneg, dps, hpmax = args
    _DPS[0] = dps
    t = TAU[N]['tau']
    t0 = time.time()
    log = dict(N=N, tau_listed=t, groups_tried=GROUP_ORDER)
    recs = []
    if t == 0:
        sol = dict(X=np.array([[0.0, 0.0, 1.0]]), group='C1', order=1, special=[], generic_orbits=1,
                   residual=0.0, signature='trivial0')
        recs.append(build_record(N, t, sol, None))
        log['found'] = True
    else:
        sols = search_design(N, t, nstart=12, want=2, time_limit=tlim)
        log['found'] = bool(sols)
        log['n_inequivalent_found'] = len(sols)
        for k, s in enumerate(sols):
            hp = None
            if N <= hpmax and k == 0:
                try:
                    hp = refine_mp(s['X'], t, dps=dps, iters=4)
                except Exception as e:  # keep going
                    log['hp_error'] = repr(e)
            recs.append(build_record(N, t, s, hp))
        neg = search_design(N, t + 1, nstart=8, want=1, time_limit=tneg)
        log['strength_tau_plus_1_found'] = bool(neg)
        log['tau_plus_1_search'] = 'strata/starts/time limited; not evidence of non-existence'
    log['seconds'] = round(time.time() - t0, 1)
    d = os.path.join(OUT, f'n{N}')
    os.makedirs(d, exist_ok=True)
    json.dump(dict(functional=FUNCTIONAL, N=N, observations=recs), open(os.path.join(d, 'observations.json'), 'w'))
    return log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('first', type=int)
    ap.add_argument('last', type=int)
    ap.add_argument('--tlim', type=float, default=40)
    ap.add_argument('--tneg', type=float, default=15)
    ap.add_argument('--dps', type=int, default=40)
    ap.add_argument('--hpmax', type=int, default=30)
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    Ns = list(range(a.first, a.last + 1))[::-1]
    with mp.Pool(a.procs) as pool:
        logs = []
        for lg in pool.imap_unordered(work, [(N, a.tlim, a.tneg, a.dps, a.hpmax) for N in Ns]):
            logs.append(lg)
            print(f"N={lg['N']:3d} tau={lg['tau_listed']:2d} found={lg['found']} "
                  f"tau+1 found={lg.get('strength_tau_plus_1_found')} {lg['seconds']}s", flush=True)
    path = os.path.join(HERE, 'search_log.json')
    old = json.load(open(path)) if os.path.exists(path) else {}
    for lg in logs:
        old[str(lg['N'])] = lg
    json.dump(old, open(path, 'w'), indent=1)
