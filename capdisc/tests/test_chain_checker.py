"""The chain checker must accept the stored reference chain and reject each kind of mismatch."""
import json, os, shutil, subprocess, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, 'certificates', 'verify_N5')


def run(d):
    return subprocess.run([sys.executable, os.path.join(ROOT, 'check', 'check_chain.py'), '5', d],
                          capture_output=True, text=True).returncode


@pytest.fixture
def chain(tmp_path):
    d = str(tmp_path / 'chain'); shutil.copytree(REF, d); return d


def edit(path, fn):
    d = json.load(open(path)); fn(d); json.dump(d, open(path, 'w'))


def test_reference_chain_passes(chain):
    assert run(chain) == 0


def test_lower_threshold_fails(chain):
    edit(os.path.join(chain, 'mesh.refine.json'), lambda d: d.update(T_hex=(float.fromhex(d['T_hex']) - 1e-16).hex()))
    assert run(chain) == 1


def test_moved_box_fails(chain):
    def f(d):
        c = d['exclusion_boxes'][0]['center']; c[0] = (float.fromhex(c[0]) + 1e-9).hex()
    edit(os.path.join(chain, 'mesh.refine.json'), f)
    assert run(chain) == 1


def test_missing_symmetric_certificate_fails(chain):
    os.remove(os.path.join(chain, 'symcert_N5_best_img2.json'))
    assert run(chain) == 1


def test_edited_local_certificate_fails(chain):
    edit(os.path.join(chain, 'localcert_N5_best.json'), lambda d: d.update(label=d['label'] + ' '))
    assert run(chain) == 1


def test_different_polynomial_fails(chain):
    edit(os.path.join(chain, 'symcert_N5_best_img0.json'), lambda d: d['P_N'].__setitem__(3, d['P_N'][3] + 1))
    assert run(chain) == 1


# ---------------------------------------------------------------- the 'verified' level (N = 7)
def synthetic_n7(tmp_path, T_fn):
    """a chain directory around the stored N = 7 local certificate, with a synthetic mesh stamp and refinement
    record (tests the checker's logic for the 'verified' level only; the real run is on the server)"""
    import hashlib, math
    d = tmp_path / 'n7'; d.mkdir()
    shutil.copy(os.path.join(ROOT, 'certificates', 'localcert_N7_best.json'), d)
    lc = json.load(open(d / 'localcert_N7_best.json'))
    hi = max(float.fromhex(i['D_enclosure_hex'][1]) for i in lc['images'])
    T = T_fn(hi)
    (d / 'mesh.done').write_text(f'N=7 T={T.hex()} MIN_EPS=0x1p-4 nf=4096 mode=rigorous keep=0x1p+0 balls=0 survivors=0 cells=1 fo=1 smear=0\n')
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    rf = dict(N=7, T=T, T_hex=T.hex(), input_cells=0, failed=0, verified=True,
              exclusion_boxes=[dict(center=[float(x).hex() for x in i['q_center']], radius=float(lc['box_radius']).hex())
                               for i in lc['images']],
              stamp=dict(lc['stamp'], step='PROOF.md Step 3 (refinement)',
                         inputs={'x/localcert_N7_best.json': sha(d / 'localcert_N7_best.json'), 'x/mesh.done': sha(d / 'mesh.done')}))
    json.dump(rf, open(d / 'mesh.refine.json', 'w'))
    return subprocess.run([sys.executable, os.path.join(ROOT, 'check', 'check_chain.py'), '7', str(d)],
                          capture_output=True, text=True).returncode


def test_verified_level_passes_with_T_above_the_enclosure(tmp_path):
    import math
    assert synthetic_n7(tmp_path, lambda hi: math.nextafter(hi, math.inf)) == 0


def test_verified_level_fails_with_T_at_the_enclosure(tmp_path):
    assert synthetic_n7(tmp_path, lambda hi: hi) == 1
