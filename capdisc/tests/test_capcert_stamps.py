"""The completion stamps of capcert_rig bind the survivor files and record the binary's provenance (audit
2026-11-04, findings 3-5): SHA-256 (csrc/sha256.h) agrees with hashlib; mesh.done lists the hash of every survivor
file; the second pass records the first pass's stamp hash and the hashes of the files it read, and refuses inputs
that differ from the first pass's stamp; the rigorous build refuses --keep-prob and --max-cells-per-root."""
import hashlib, os, shutil, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.buildbin import binary as build_binary     # noqa: E402

N, T, EPS1, EPS2 = 3, '0.6', '0.3', '0.1'               # N = 3 well above its optimum: survivors in a fraction of a second


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def stamp(p):
    return dict(kv.split('=', 1) for kv in open(p).read().split())


@pytest.fixture(scope='module')
def rig():
    return build_binary('capcert_rig')


@pytest.fixture(scope='module')
def passes(rig, tmp_path_factory):
    d = tmp_path_factory.mktemp('mesh')
    r1 = subprocess.run([rig, str(N), T, EPS1, str(d / 'mesh'), '--threads', '2'], capture_output=True, text=True)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    r2 = subprocess.run([rig, str(N), T, EPS2, str(d / 'mesh2'), '--threads', '2', '--cells', str(d / 'mesh')], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    return d


def test_sha256_matches_hashlib(tmp_path):
    """csrc/sha256.h on the block-boundary lengths"""
    src = tmp_path / 't.c'
    src.write_text('#include "sha256.h"\nint main(int c, char **v) { for (int i = 1; i < c; i++) { char h[65]; long long n;\n'
                   '  if (sha256_file(v[i], h, &n)) return 1; printf("%s %lld\\n", h, n); } return 0; }\n')
    exe = tmp_path / 't'
    subprocess.run(['gcc', '-O2', '-I', os.path.join(ROOT, 'csrc'), '-o', str(exe), str(src)], check=True)
    files = []
    for n in (0, 1, 3, 55, 56, 57, 63, 64, 65, 119, 120, 128, 1000, 65535, 65536, 65537, 200003):
        p = tmp_path / f'f{n}'; p.write_bytes(os.urandom(n)); files.append(p)
    out = subprocess.run([str(exe), *map(str, files)], check=True, capture_output=True, text=True).stdout.split('\n')
    for p, line in zip(files, out):
        h, n = line.split()
        assert h == sha(p) and int(n) == os.path.getsize(p)


def test_first_pass_stamp_binds_every_survivor_file(passes):
    s = stamp(passes / 'mesh.done')
    nf = int(s['files']); assert nf >= 1
    assert sorted(f for f in os.listdir(passes) if f.startswith('mesh.surv.')) == sorted(f'mesh.surv.{t}' for t in range(nf))
    rec = (2 * (2 * N - 3) + 1) * 8; total = 0
    for t in range(nf):
        assert s[f'surv.{t}'] == sha(passes / f'mesh.surv.{t}')
        total += os.path.getsize(passes / f'mesh.surv.{t}') // rec
    assert int(s['records']) == total >= int(s['survivors']) > 0
    assert s['mode'] == 'rigorous' and s['keep'] == '0x1p+0' and s['balls'] == '0' and s['maxc'] == '0' and s['prefix'] == 'mesh'


def test_stamps_record_provenance(passes):
    for P in ('mesh', 'mesh2'):
        s = stamp(passes / f'{P}.done')
        assert len(s['commit']) == 40 and s['tree_dirty'] in ('0', '1') and s['build'] == 'rigorous'
        assert s['src'] == 'capcert.c,fo.h,ia.h,rig.h,sha256.h' and len(s['src_sha256']) == 64
        flags = s['flags'].replace('%20', ' ').split()
        assert {'-DRIGOROUS', '-ffp-contract=off', '-fno-fast-math'} <= set(flags)
        assert s['cc']


def test_source_hash_is_that_of_the_sources(passes):
    """src_sha256 = sha256 of the sha256sum lines of the C sources (as build.sh computes it and the checker recomputes)"""
    s = stamp(passes / 'mesh.done')
    lines = ''.join(f"{sha(os.path.join(ROOT, 'csrc', f))}  csrc/{f}\n" for f in s['src'].split(','))
    assert hashlib.sha256(lines.encode()).hexdigest() == s['src_sha256']


def test_second_pass_records_its_inputs(passes):
    s1, s2 = stamp(passes / 'mesh.done'), stamp(passes / 'mesh2.done')
    assert s2['from'] == 'mesh' and s2['same_dir'] == '1' and s2['prefix'] == 'mesh2'
    assert s2['from_stamp'] == sha(passes / 'mesh.done')
    assert int(s2['input_files']) == int(s1['files'])
    assert all(s2[f'in.{t}'] == s1[f'surv.{t}'] for t in range(int(s1['files'])))
    assert int(s2['input_cells']) == int(s1['records'])
    assert s2['keep'] == '0x1p+0' and s2['balls'] == '0'          # finding 5: the real values, not hardcoded
    for t in range(int(s2['files'])):
        assert s2[f'surv.{t}'] == sha(passes / f'mesh2.surv.{t}')


def test_second_pass_refuses_altered_input(rig, passes, tmp_path):
    """finding 3: first-pass survivors replaced by zeros of the same size -> the second pass writes no stamp"""
    d = tmp_path / 'alt'; shutil.copytree(passes, d)
    for f in os.listdir(d):
        if f.startswith('mesh2'): os.remove(d / f)
    p = d / 'mesh.surv.0'; p.write_bytes(bytes(os.path.getsize(p)))
    r = subprocess.run([rig, str(N), T, EPS2, str(d / 'mesh2'), '--threads', '2', '--cells', str(d / 'mesh')], capture_output=True, text=True)
    assert r.returncode != 0 and not (d / 'mesh2.done').exists(), r.stdout + r.stderr
    assert 'differs from' in r.stderr


def test_second_pass_refuses_a_gap_or_a_missing_first_pass(rig, passes, tmp_path):
    d = tmp_path / 'gap'; shutil.copytree(passes, d)
    for f in os.listdir(d):
        if f.startswith('mesh2'): os.remove(d / f)
    shutil.copy(d / 'mesh.surv.0', d / 'mesh.surv.7')            # a file after a gap
    r = subprocess.run([rig, str(N), T, EPS2, str(d / 'mesh2'), '--cells', str(d / 'mesh')], capture_output=True, text=True)
    assert r.returncode != 0 and 'gap' in r.stderr
    os.remove(d / 'mesh.surv.7'); os.remove(d / 'mesh.done')
    r = subprocess.run([rig, str(N), T, EPS2, str(d / 'mesh2'), '--cells', str(d / 'mesh')], capture_output=True, text=True)
    assert r.returncode != 0 and 'did not complete' in r.stderr
    r = subprocess.run([rig, str(N), T, EPS2, str(d / 'mesh'), '--cells', str(d / 'mesh')], capture_output=True, text=True)
    assert r.returncode != 0


@pytest.mark.parametrize('opt', [['--keep-prob', '0.5'], ['--max-cells-per-root', '100'], ['--balls', 'x'], ['--no-such-option']])
def test_rigorous_build_refuses_options_that_change_the_survivors(rig, tmp_path, opt):
    r = subprocess.run([rig, str(N), T, EPS1, str(tmp_path / 'mesh'), *opt], capture_output=True, text=True)
    assert r.returncode != 0 and not (tmp_path / 'mesh.done').exists()


def test_version_prints_provenance(rig):
    out = subprocess.run([rig, '--version'], check=True, capture_output=True, text=True).stdout
    assert 'commit=' in out and 'src_sha256=' in out and 'flags=' in out
