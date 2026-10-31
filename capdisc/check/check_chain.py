"""check/check_chain.py -- does the certificate chain of pipeline A fit together?  (PROOF.md Section 3)

Independent of the proof code: uses only the Python standard library (json, hashlib, fractions), and does no
numerics beyond exact rational comparisons of the values stored in the certificates. It checks that

  C1  the local certificate is verified for every image, and lists the images the papers expect;
  C2  every image has a verified symmetric certificate that refers to it, with the papers' P_N, P_N irreducible
      and identified as the factor that vanishes, and with the symmetric zero inside that image's enclosure;
  C3  the congruence certificate covers every image and is verified;
  C4  the mesh ran in rigorous mode, kept every survivor, used no exclusion balls, and its threshold T_mesh is
      at least the refinement threshold T;
  C5  the refinement is verified with no failed cell, used T = the largest upper end of the D* enclosures,
      excluded exactly the boxes of the local certificate, read the local certificate and the mesh output that
      are in this directory (SHA-256), and read at least as many survivors as the mesh wrote;
  C6  D* < T strictly, with D* taken from the 128-bit symmetric enclosure (this gives D >= T > D* outside the
      boxes, PROOF.md Section 3);
  C7  N D* lies in the papers' isolating interval of P_N, and D* agrees with the digits printed in the papers;
  C8  all certificates come from the same commit, with no modified proof/ or csrc/ files.

usage: python check/check_chain.py N DIR        (DIR as written by verify.sh, e.g. results/verify_N5)
writes DIR/chain_N<N>.json and exits 0 if every check passes.
"""
import hashlib, json, os, sys
from fractions import Fraction as Q

N, DIR = int(sys.argv[1]), sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
paper = json.load(open(os.path.join(HERE, 'paper_values.json')))[str(N)]
name = paper['name']
checks, notes = {}, []


def load(fn):
    return json.load(open(os.path.join(DIR, fn)))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def hexq(s):                      # a hex float string or a JSON number, as an exact rational
    return Q(float.fromhex(s)) if isinstance(s, str) else Q(s)


def ball(s):                      # an Arb ball string '[m +/- r]' (or an exact number) -> (lo, hi), exact rationals
    s = s.strip()
    if s.startswith('['):
        m, r = s[1:-1].split('+/-')
        m, r = Q(m.strip()), Q(r.strip())
        return m - r, m + r
    return Q(s), Q(s)


def check(key, ok, why=''):
    checks[key] = bool(ok) and checks.get(key, True)
    if not ok: notes.append(f'{key}: {why}')


lc = load(f'localcert_{name}.json')
imgs = lc['images']
check('C1', lc['verified'] and all(i['verified'] for i in imgs), 'local certificate not verified')
check('C1', len(imgs) == paper['images'], f"{len(imgs)} images, papers expect {paper['images']}")

Dstar = []
for i in imgs:
    k = i['image']
    sc = load(f'symcert_{name}_img{k}.json')
    check('C2', sc['verified'] and sc['image'] == k, f'image {k}: symmetric certificate not verified')
    check('C2', sc.get('P_N') == paper['P_N'], f'image {k}: P_N differs from the papers')
    check('C2', sc.get('P_N_irreducible') is True and sc.get('P_N_is_the_factor') is True, f'image {k}: P_N not shown irreducible / not the factor')
    check('C2', sc.get('group_order') == paper['group_order'], f'image {k}: group order')
    dlo, dhi = ball(sc['s_enclosure'][-1])
    elo, ehi = (hexq(x) for x in i['D_enclosure_hex']) if 'D_enclosure_hex' in i else (Q(x) for x in i['D_enclosure'])
    check('C2', elo <= dlo and dhi <= ehi, f'image {k}: symmetric D* not inside the local enclosure')
    Dstar.append((dlo, dhi))

cg = load(f'congruence_{name}.json')
check('C3', cg['congruent'] and sorted(o['image'] for o in cg['images']) == sorted(i['image'] for i in imgs)
      and all(o['contained_in_Z0'] and o['verified_image'] for o in cg['images']), 'congruence')

stamp = dict(kv.split('=', 1) for kv in open(os.path.join(DIR, 'mesh.done')).read().split())
T_mesh = Q(float.fromhex(stamp['T']))
check('C4', stamp['mode'] == 'rigorous' and float.fromhex(stamp['keep']) == 1.0 and int(stamp['balls']) == 0
      and int(stamp['N']) == N, 'mesh not run in proof mode')

rf = load('mesh.refine.json')
T = max(hexq(i['D_enclosure_hex'][1]) if 'D_enclosure_hex' in i else Q(i['D_enclosure'][1]) for i in imgs)
check('C4', T_mesh >= T, 'mesh threshold below the refinement threshold')
check('C5', rf['verified'] and rf['failed'] == 0, 'refinement left cells')
check('C5', hexq(rf['T_hex']) == T, 'refinement threshold is not the largest upper end of the D* enclosures')
boxes = sorted((tuple(hexq(x) for x in b['center']), hexq(b['radius'])) for b in rf['exclusion_boxes'])
local_boxes = sorted((tuple(Q(x) for x in i['q_center']), Q(lc['box_radius'])) for i in imgs)
check('C5', boxes == local_boxes, 'exclusion boxes differ from the local boxes')
ins = rf['stamp']['inputs']
lc_path = os.path.relpath(os.path.join(DIR, f'localcert_{name}.json'), REPO)
check('C5', ins.get(lc_path) == sha(os.path.join(REPO, lc_path)), 'refinement read a different local certificate')
done_path = os.path.relpath(os.path.join(DIR, 'mesh.done'), REPO)
check('C5', ins.get(done_path) == sha(os.path.join(REPO, done_path)), 'refinement read a different mesh stamp')
check('C5', rf['input_cells'] >= int(stamp['survivors']), 'refinement read fewer survivors than the mesh wrote')
if '--hash-survivors' in sys.argv:
    for p, h in ins.items():
        if '.surv.' in p: check('C5', sha(os.path.join(REPO, p)) == h, f'{p}: hash')

check('C6', all(dhi < T for _, dhi in Dstar), 'D* is not strictly below T')

lo, hi = (Q(x) for x in paper['u_interval'])
pr = Q(paper['D_printed']); ulp = Q(1, 10 ** (len(paper['D_printed']) - 2))
check('C7', all(lo <= N * dlo and N * dhi <= hi for dlo, dhi in Dstar), 'N D* outside the isolating interval')
check('C7', all(pr <= dlo and dhi < pr + ulp for dlo, dhi in Dstar), 'D* does not match the printed digits')

stamps = [lc['stamp'], cg['stamp'], rf['stamp']] + [load(f'symcert_{name}_img{i["image"]}.json')['stamp'] for i in imgs]
commits = {s['commit'] for s in stamps}
check('C8', len(commits) == 1 and not any(s['tree_dirty'] for s in stamps),
      f'commits {sorted(c[:8] for c in commits if c)}; dirty: {[s["step"] for s in stamps if s["tree_dirty"]]}')

ok = all(checks.values())
out = dict(N=N, name=name, directory=DIR, checks=checks, notes=notes, verdict='chain complete' if ok else 'CHAIN INCOMPLETE',
           T=str(T), D_star_upper=str(max(d for _, d in Dstar)),
           files={f: sha(os.path.join(DIR, f)) for f in sorted(os.listdir(DIR)) if f.endswith('.json') and not f.startswith('chain_')})
json.dump(out, open(os.path.join(DIR, f'chain_N{N}.json'), 'w'), indent=1)
for k in sorted(checks): print(f'{k}: {"ok" if checks[k] else "FAIL"}')
for n_ in notes: print('  ' + n_)
print(out['verdict'])
sys.exit(0 if ok else 1)
