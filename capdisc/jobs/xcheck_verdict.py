"""Verdict of the second-kernel pipeline B (VERIFICATION.md).  For each N, B proves Theorem 10 on its
own iff all of the following hold; then, with pipeline A, the theorem is cross-verified.
  local (mpmath.iv), symmetric for every image (mpmath.iv, exact orbits, D_s < T, N D_s in the
  theorem interval), congruence of the images (mpmath.iv), roots (sympy), elimination (msolve), mesh (capcert_arb, complete, T >= T_A)
  and refinement of all its survivors (mpmath.iv, complete, no failure).
usage: python jobs/xcheck_verdict.py [RESULTS_DIR]
  writes RESULTS_DIR/xcheck_verdict.json; also certificates/xcheck_verdict.json with UPDATE_CERTS=1, or when the
  mesh and refinement are taken from the records certificates/pipelineB_N*.json (no xproof files in RESULTS_DIR)"""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); CERT = os.path.join(ROOT, 'certificates')
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results')
ld = lambda f: json.load(open(f)) if os.path.exists(f) else None
roots, elim = ld(os.path.join(CERT, 'xcheck_roots.json')) or {}, ld(os.path.join(CERT, 'xcheck_elim.json')) or {}
out = {}
for N in (5, 6):
    nm = f'N{N}_best'; lc = ld(os.path.join(CERT, f'localcert_{nm}.json')); T = max(i['D_enclosure'][1] for i in lc['images'])
    xl = ld(os.path.join(CERT, f'xcheck_local_{nm}.json'))
    xs = [ld(os.path.join(CERT, f'xcheck_sym_{nm}_img{k}.json')) for k in range(len(lc['images']))]
    xc = ld(os.path.join(CERT, f'xcheck_congruence_{nm}.json'))
    done = os.path.join(R, f'xproof_N{N}.done'); xr = ld(os.path.join(R, f'xproof_N{N}.xrefine.json'))
    st = dict(kv.split('=', 1) for kv in open(done).read().split()) if os.path.exists(done) else None
    rec = ld(os.path.join(CERT, f'pipelineB_N{N}.json'))
    if (st is None or xr is None) and rec:      # the mesh and refinement ran elsewhere: their records, verbatim
        st = dict(kv.split('=', 1) for kv in rec['mesh']['done_stamp'].split()); xr = rec['refine']['xrefine']
    checks = dict(
        local=bool(xl and xl.get('verified') and xl.get('T') == T),
        symmetric=bool(all(d and d.get('verified') and d.get('T') == T for d in xs)),
        congruence=bool(xc and xc.get('congruent') and sorted(o['image'] for o in xc['images']) == list(range(len(lc['images'])))),
        roots=bool(roots.get(str(N), {}).get('verified')),
        elimination=bool(elim.get(str(N), {}).get('verified')),
        mesh=bool(st and st['mode'] == 'arb' and float.fromhex(st['T']) >= T and float.fromhex(st['keep']) == 1.0),
        refinement=bool(xr and xr.get('verified') and xr.get('complete') and xr.get('T') == T
                        and st and xr.get('mesh_stamp', {}).get('cells') == st['cells'] and xr['input_cells'] >= int(st['survivors'])))
    out[N] = dict(checks=checks, pipeline_B_proves=all(checks.values()), T=T)
    print(f"N={N}: " + ", ".join(f"{k}={'ok' if v else 'MISSING/FAILED'}" for k, v in checks.items())
          + f"  ->  {'pipeline B proves Theorem 10 (second kernels)' if all(checks.values()) else 'incomplete'}")
# results/ always; certificates/ (tracked in git) only on request, so that git pull on the server keeps working
os.makedirs(R, exist_ok=True); json.dump(out, open(os.path.join(R, 'xcheck_verdict.json'), 'w'), indent=1)
if os.environ.get('UPDATE_CERTS', '0') == '1' or len(sys.argv) <= 1 and not os.path.exists(os.path.join(R, 'xproof_N5.done')):
    json.dump(out, open(os.path.join(CERT, 'xcheck_verdict.json'), 'w'), indent=1)
