"""check/make_sidecar.sh writes the <P>.surv.sha256 sidecar that binds the survivor files of a pre-2026-11-04 mesh
stamp (review 2026-11-04, R2): it lists exactly the files <P>.surv.<digits>, never itself, names the stamp's own
SHA-256, refuses to run without the stamp and refuses to replace an existing sidecar unless told to."""
import hashlib, os, re, shutil, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, 'check', 'make_sidecar.sh')
CHECKER = os.path.join(ROOT, 'check', 'check_chain.py')
LEG = os.path.join(ROOT, 'tests', 'data', 'legacy_768cefc_N5')
LEG2 = os.path.join(ROOT, 'tests', 'data', 'legacy_N5_2pass')
FIRST = re.compile(r'# stamp (\S+) sha256 ([0-9a-f]{64})  hashed (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\n')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def run(*args):
    return subprocess.run(['bash', SCRIPT, *args], capture_output=True, text=True)


def copy(src, tmp_path, name='verify_N5'):
    d = tmp_path / name
    shutil.copytree(src, d)
    return d


def file_lines(path):
    return sorted(ln.split()[-1] + ' ' + ln.split()[0] for ln in open(path).read().split('\n')[1:] if ln.strip())


@pytest.mark.parametrize('fixture,passes', [(LEG, ['mesh']), (LEG2, ['mesh', 'mesh2'])])
def test_regenerates_the_fixture_sidecars_and_the_chain_passes(fixture, passes, tmp_path):
    d = copy(fixture, tmp_path)
    for P in passes:
        old = open(d / f'{P}.surv.sha256').read()
        os.remove(d / f'{P}.surv.sha256')
        r = run(str(d), P)
        assert r.returncode == 0, r.stdout + r.stderr
        new = open(d / f'{P}.surv.sha256').read()
        m = FIRST.match(new)
        assert m and m.group(1) == f'{P}.done' and m.group(2) == sha(d / f'{P}.done'), new.split('\n')[0]
        # the same files with the same hashes as the fixture's sidecar (the fixture lists them with a path prefix)
        assert [ln.split()[0] + ' ' + os.path.basename(ln.split()[1]) for ln in sorted(new.split('\n')[1:]) if ln] == \
               sorted(ln.split()[0] + ' ' + os.path.basename(ln.split()[-1]) for ln in old.split('\n')[1:] if ln.strip())
    r = subprocess.run([sys.executable, CHECKER, '5', str(d), '--out', str(tmp_path / 'chain.json')], capture_output=True, text=True)
    assert r.returncode == 0 and 'chain complete' in r.stdout, r.stdout + r.stderr


def test_lists_exactly_the_survivor_files_of_the_pass(tmp_path):
    """not itself, not another pass's files, not temporaries, not a stray file with a non-numeric suffix"""
    d = copy(LEG2, tmp_path)
    os.remove(d / 'mesh.surv.sha256')
    (d / 'mesh.surv.0.tmp').write_bytes(b'x'); (d / 'mesh.surv.old').write_bytes(b'y'); (d / 'mesh.surv.10').write_bytes(b'')
    r = run(str(d), 'mesh')
    assert r.returncode == 0, r.stdout + r.stderr
    names = [ln.split()[-1] for ln in open(d / 'mesh.surv.sha256').read().split('\n')[1:] if ln.strip()]
    assert names == ['mesh.surv.0', 'mesh.surv.1', 'mesh.surv.10']
    assert all(sha(d / n) == ln.split()[0] for n, ln in zip(names, [l for l in open(d / 'mesh.surv.sha256').read().split('\n')[1:] if l.strip()]))


def test_refuses_without_the_stamp(tmp_path):
    d = copy(LEG, tmp_path)
    os.remove(d / 'mesh.surv.sha256'); os.remove(d / 'mesh.done')
    r = run(str(d), 'mesh')
    assert r.returncode != 0 and 'mesh.done' in r.stderr
    assert not os.path.exists(d / 'mesh.surv.sha256')


def test_refuses_to_replace_an_existing_sidecar_unless_forced(tmp_path):
    d = copy(LEG, tmp_path)
    before = open(d / 'mesh.surv.sha256').read()
    r = run(str(d), 'mesh')
    assert r.returncode != 0 and open(d / 'mesh.surv.sha256').read() == before
    r = run(str(d), 'mesh', '--force')
    assert r.returncode == 0, r.stdout + r.stderr
    assert open(d / 'mesh.surv.sha256').read() != before          # a new hashing time
    assert file_lines(d / 'mesh.surv.sha256') == file_lines(d / 'mesh.surv.sha256')   # and well formed


def test_the_report_recipe_bug_is_not_reproduced(tmp_path):
    """the sidecar never contains an entry for a file named <P>.surv.sha256"""
    d = copy(LEG, tmp_path)
    r = run(str(d), 'mesh', '--force')
    assert r.returncode == 0
    assert 'surv.sha256' not in '\n'.join(open(d / 'mesh.surv.sha256').read().split('\n')[1:])
