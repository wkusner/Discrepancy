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
