"""The chain checker must accept the stored reference chains and reject each kind of mismatch.

Reference chains (verify.sh at the commit in their stamps): certificates/verify_N5 (one mesh pass) and
certificates/verify_N5_2pass (MESH2_EPS=0.01: two passes). Legacy chains, in the format of the N = 7 run that was
started at 768cefc (stamps without hashes or provenance; survivor files bound by <P>.surv.sha256 sidecars):
tests/data/legacy_768cefc_N5 (the former certificates/verify_N5, all from 768cefc) and tests/data/legacy_N5_2pass
(the former certificates/verify_N5_2pass, f520db8 + e97ada7).

Mutations edit one certificate and then, where noted, recompute the hashes that bind it downstream (`rehash`), so
that each test exercises the one content check it is about, not only the hash chain."""
import hashlib, json, os, re, shutil, subprocess, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'check'))
import check_chain                                   # noqa: E402
REF = os.path.join(ROOT, 'certificates', 'verify_N5')
REF2 = os.path.join(ROOT, 'certificates', 'verify_N5_2pass')
LEG = os.path.join(ROOT, 'tests', 'data', 'legacy_768cefc_N5')
LEG2 = os.path.join(ROOT, 'tests', 'data', 'legacy_N5_2pass')


def run(d, N=5, out=True):
    args = [sys.executable, os.path.join(ROOT, 'check', 'check_chain.py'), str(N), d]
    if out: args += ['--out', d + '.result.json']
    r = subprocess.run(args, capture_output=True, text=True)
    return r.returncode, r.stdout


def result(d):
    return json.load(open(d + '.result.json'))


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def copy(src, tmp_path, name='verify_N5'):
    d = str(tmp_path / name); shutil.copytree(src, d)
    for f in os.listdir(d):
        if f.startswith('chain_'): os.remove(os.path.join(d, f))
    return d


def edit(path, fn):
    d = json.load(open(path)); fn(d); json.dump(d, open(path, 'w'))


def edit_text(path, a, b):
    s = open(path).read(); assert a in s, (a, s[:200]); open(path, 'w').write(s.replace(a, b))


def rehash(d):
    """recompute the hashes that bind the stamps and the local certificate downstream (mesh2.done from_stamp, the
    refinement's inputs), as a consistent forger would"""
    if os.path.exists(os.path.join(d, 'mesh2.done')):
        p = os.path.join(d, 'mesh2.done'); s = open(p).read()
        if 'from_stamp=' in s: open(p, 'w').write(re.sub(r'from_stamp=[0-9a-f]+', 'from_stamp=' + sha(os.path.join(d, 'mesh.done')), s))
        rf = os.path.join(d, 'mesh2.refine.json')
    else:
        rf = os.path.join(d, 'mesh.refine.json')
    edit(rf, lambda r: r['stamp']['inputs'].update({k: sha(os.path.join(d, os.path.basename(k))) for k in r['stamp']['inputs']
                                                   if not '.surv.' in k}))


@pytest.fixture
def chain(tmp_path): return copy(REF, tmp_path)


@pytest.fixture
def chain2(tmp_path): return copy(REF2, tmp_path)


# ---------------------------------------------------------------- the reference chains
def test_reference_chain_passes(chain):
    rc, out = run(chain); assert rc == 0, out
    r = result(chain)
    assert r['level'] == 'full' and r['routes']['mesh']['binding'] == 'stamp'
    assert set(r['code']) >= {'certificate_commits', 'mesh_commits', 'checker_commit', 'proof_code_same_as_checker'}
    assert all(r['code']['mesh_commits'].values())
    # the only flag a new-format chain can carry: later proof code in this checkout
    assert {f['kind'] for f in r['flags']} <= {'proof_code_differs_from_checker'}
    assert r['verdict'] == 'chain complete' + (f" ({len(r['flags'])} flags)" if r['flags'] else '')


def test_two_pass_chain_passes(chain2):
    rc, out = run(chain2); assert rc == 0, out
    r = result(chain2)
    assert r['routes']['mesh']['binding'] == r['routes']['mesh2']['binding'] == 'stamp'
    assert r['routes']['mesh2_input']['binding'].startswith('hashes')


def test_stored_chain_results_use_the_level_full():
    for d in (REF, REF2):
        r = json.load(open(os.path.join(d, 'chain_N5.json')))
        assert r['level'] == 'full' and r['verdict'] == 'chain complete'


def test_the_checker_does_not_write_into_the_directory(chain):
    before = sorted(os.listdir(chain))
    rc, out = run(chain, out=False); assert rc == 0, out
    assert sorted(os.listdir(chain)) == before and os.path.exists(chain + '.chain_N5.json')


# ---------------------------------------------------------------- content mutations (existing)
def test_lower_threshold_fails(chain):
    edit(os.path.join(chain, 'mesh.refine.json'), lambda d: d.update(T_hex=(float.fromhex(d['T_hex']) - 1e-16).hex()))
    assert run(chain)[0] == 1


def test_moved_box_fails(chain):
    def f(d):
        c = d['exclusion_boxes'][0]['center']; c[0] = (float.fromhex(c[0]) + 1e-9).hex()
    edit(os.path.join(chain, 'mesh.refine.json'), f)
    assert run(chain)[0] == 1


def test_missing_symmetric_certificate_fails(chain):
    os.remove(os.path.join(chain, 'symcert_N5_best_img2.json'))
    assert run(chain)[0] == 1


def test_edited_local_certificate_fails(chain):
    edit(os.path.join(chain, 'localcert_N5_best.json'), lambda d: d.update(label=d['label'] + ' '))
    assert run(chain)[0] == 1


def test_different_polynomial_fails(chain):
    edit(os.path.join(chain, 'symcert_N5_best_img0.json'), lambda d: d['P_N'].__setitem__(3, d['P_N'][3] + 1))
    assert run(chain)[0] == 1


def test_missing_mesh_stamp_fails_cleanly(chain):
    os.remove(os.path.join(chain, 'mesh.done'))
    rc, out = run(chain)
    assert rc == 1 and 'missing certificate: mesh.done' in out and 'Traceback' not in out


def test_local_certificate_for_another_N_fails(chain):
    edit(os.path.join(chain, 'localcert_N5_best.json'), lambda d: d.update(N=6)); rehash(chain)
    rc, out = run(chain); assert rc == 1 and 'C1: FAIL' in out


def test_refinement_for_another_N_fails(chain):
    edit(os.path.join(chain, 'mesh.refine.json'), lambda d: d.update(N=6))
    rc, out = run(chain); assert rc == 1 and 'C5: FAIL' in out


# ---------------------------------------------------------------- survivor files bound by hash (findings 3, 6)
def test_zeroed_first_pass_survivors_fail_in_the_two_pass_chain(chain2):
    """the audit's reproduction: mesh.surv.0 replaced by zero bytes of the same size"""
    p = os.path.join(chain2, 'mesh.surv.0'); open(p, 'wb').write(bytes(os.path.getsize(p)))
    rc, out = run(chain2); assert rc == 1 and 'mesh.surv.0: SHA-256 differs' in out


def test_truncated_survivor_file_fails(chain):
    p = os.path.join(chain, 'mesh.surv.0'); rec = (2 * 7 + 1) * 8
    b = open(p, 'rb').read(); open(p, 'wb').write(b[:-rec])
    assert run(chain)[0] == 1


def test_missing_survivor_file_fails(chain2):
    os.remove(os.path.join(chain2, 'mesh2.surv.1'))
    assert run(chain2)[0] == 1


def test_extra_survivor_file_fails(chain):
    shutil.copy(os.path.join(chain, 'mesh.surv.0'), os.path.join(chain, 'mesh.surv.2'))
    assert run(chain)[0] == 1


def test_resumed_first_pass_is_accepted(chain2):
    """finding 6: a resumed first pass leaves more records than survivors= (duplicates); with the files bound by
    hash, C4 accepts input_cells >= survivors. Simulated by lowering survivors= by one and re-binding."""
    p = os.path.join(chain2, 'mesh.done'); s = open(p).read()
    n = int(re.search(r'survivors=(\d+)', s).group(1))
    open(p, 'w').write(s.replace(f'survivors={n} ', f'survivors={n - 1} ')); rehash(chain2)
    rc, out = run(chain2); assert rc == 0, out


def test_fewer_records_than_survivors_fail(chain2):
    p = os.path.join(chain2, 'mesh.done'); s = open(p).read()
    n = int(re.search(r'survivors=(\d+)', s).group(1))
    open(p, 'w').write(s.replace(f'survivors={n} ', f'survivors={n + 1} ')); rehash(chain2)
    assert run(chain2)[0] == 1


def test_second_pass_read_fewer_cells_fails(chain2):
    p = os.path.join(chain2, 'mesh2.done'); s = open(p).read()
    n = int(re.search(r'input_cells=(\d+)', s).group(1))
    open(p, 'w').write(s.replace(f'input_cells={n} ', f'input_cells={n - 1} ')); rehash(chain2)
    assert run(chain2)[0] == 1


def test_second_pass_input_hash_mismatch_fails(chain2):
    p = os.path.join(chain2, 'mesh2.done'); s = open(p).read()
    open(p, 'w').write(re.sub(r' in\.0=[0-9a-f]+', ' in.0=' + '0' * 64, s)); rehash(chain2)
    rc, out = run(chain2); assert rc == 1 and 'second pass read survivor files other than' in out


def test_second_pass_from_another_directory_fails(chain2):
    edit_text(os.path.join(chain2, 'mesh2.done'), 'same_dir=1', 'same_dir=0'); rehash(chain2)
    assert run(chain2)[0] == 1


def test_two_pass_chain_fails_without_the_second_refinement(chain2):
    os.remove(os.path.join(chain2, 'mesh2.refine.json'))
    assert run(chain2)[0] == 1


def test_keep_fraction_in_the_second_pass_fails(chain2):
    """finding 5: the second-pass stamp records KEEP_PROB; anything but 1 is not proof mode"""
    edit_text(os.path.join(chain2, 'mesh2.done'), 'keep=0x1p+0', 'keep=0x1p-1'); rehash(chain2)
    rc, out = run(chain2); assert rc == 1 and 'mesh2 not run in proof mode' in out


# ---------------------------------------------------------------- C8 (finding 1) and mesh provenance (finding 4)
def test_same_code_compares_the_proof_code():
    same, files = check_chain.same_code('f32be61', 'ce7901e')             # proof/ differs (audit, finding 1)
    assert not same and any(f.startswith('capdisc/proof/') for f in files)
    assert check_chain.same_code('ce7901e', 'fbc61b9') == (True, [])      # differ only in jobs/


def test_git_fails_loudly():
    with pytest.raises(check_chain.GitError):
        check_chain.git('rev-parse', '--verify', '0123456789abcdef0123456789abcdef01234567^{commit}')


def test_certificates_from_different_proof_code_fail(chain):
    """the audit's reproduction (iii): the refinement stamp's commit changed to f32be61"""
    edit(os.path.join(chain, 'mesh.refine.json'), lambda d: d['stamp'].update(commit=_full('f32be61')))
    rc, out = run(chain); assert rc == 1 and 'C8: FAIL' in out and 'differ in the proof code' in out


def test_modified_tree_in_a_stamp_fails(chain):
    edit(os.path.join(chain, 'localcert_N5_best.json'), lambda d: d['stamp'].update(tree_dirty=True)); rehash(chain)
    rc, out = run(chain); assert rc == 1 and 'modified proof code' in out


def test_mesh_binary_from_a_dirty_tree_fails(chain):
    edit_text(os.path.join(chain, 'mesh.done'), 'tree_dirty=0', 'tree_dirty=1'); rehash(chain)
    rc, out = run(chain); assert rc == 1 and 'modified tree' in out


def test_mesh_source_hash_mismatch_fails(chain):
    p = os.path.join(chain, 'mesh.done'); s = open(p).read()
    open(p, 'w').write(re.sub(r'src_sha256=[0-9a-f]+', 'src_sha256=' + '1' * 64, s)); rehash(chain)
    rc, out = run(chain); assert rc == 1 and 'do not have the hash baked into the binary' in out


def test_mesh_binary_from_other_proof_code_fails(chain):
    """the mesh's commit is compared with the Python stamps' commits (f32be61 has other csrc and proof code)"""
    p = os.path.join(chain, 'mesh.done'); s = open(p).read()
    open(p, 'w').write(re.sub(r'commit=[0-9a-f]+', 'commit=' + _full('f32be61'), s)); rehash(chain)
    rc, out = run(chain); assert rc == 1 and 'C8: FAIL' in out


def test_mesh_without_proof_flags_fails(chain):
    edit_text(os.path.join(chain, 'mesh.done'), '%20-ffp-contract=off', ''); rehash(chain)
    rc, out = run(chain); assert rc == 1 and 'C4: FAIL' in out


def _full(c):
    return check_chain.git('rev-parse', c).decode().strip()


# ---------------------------------------------------------------- legacy chains (the N = 7 run at 768cefc)
def test_legacy_single_pass_chain_passes_with_the_sidecar(tmp_path):
    d = copy(LEG, tmp_path)
    rc, out = run(d); assert rc == 0, out
    r = result(d)
    assert r['routes']['mesh']['binding'] == 'sidecar' and r['routes']['mesh']['stamp_sha256'] == sha(os.path.join(d, 'mesh.done'))
    assert r['code']['mesh_commits'] == {'mesh': None}
    kinds = {f['kind'] for f in r['flags']}
    assert {'survivors_bound_by_sidecar', 'tree_state_unreliable', 'mesh_provenance_missing',
            'proof_code_differs_from_checker'} <= kinds
    assert r['verdict'] == f"chain complete ({len(r['flags'])} flags)" and 'FLAG [mesh_provenance_missing]' in out
    assert set(r['code']['certificate_commits'].values()) == {_full('768cefc')}


def test_legacy_two_pass_chain_passes_with_sidecars(tmp_path):
    d = copy(LEG2, tmp_path)                    # in a directory named like the one it was written in (verify_N5)
    rc, out = run(d); assert rc == 0, out
    r = result(d)
    assert r['routes']['mesh']['binding'] == r['routes']['mesh2']['binding'] == 'sidecar'
    assert r['routes']['mesh2_input']['binding'].startswith('record count')
    assert 'second_pass_input_by_count' in {f['kind'] for f in r['flags']}


def test_legacy_two_pass_chain_in_a_renamed_directory_fails(tmp_path):
    d = copy(LEG2, tmp_path, name='elsewhere')
    rc, out = run(d); assert rc == 1 and 'not in this directory' in out


def test_legacy_chain_without_sidecar_fails(tmp_path):
    d = copy(LEG, tmp_path); os.remove(os.path.join(d, 'mesh.surv.sha256'))
    rc, out = run(d); assert rc == 1 and 'no mesh.surv.sha256' in out


def test_legacy_sidecar_must_name_the_stamp_hash(tmp_path):
    d = copy(LEG, tmp_path); p = os.path.join(d, 'mesh.surv.sha256'); s = open(p).read()
    open(p, 'w').write(re.sub(r'sha256 [0-9a-f]{64}', 'sha256 ' + 'a' * 64, s, count=1))
    rc, out = run(d); assert rc == 1 and 'the stamp hash is not the SHA-256 of mesh.done' in out


def test_legacy_sidecar_without_the_stamp_line_fails(tmp_path):
    d = copy(LEG, tmp_path); p = os.path.join(d, 'mesh.surv.sha256')
    open(p, 'w').write(''.join(open(p).readlines()[1:]))
    assert run(d)[0] == 1


def test_legacy_sidecar_must_list_every_file(tmp_path):
    d = copy(LEG, tmp_path); p = os.path.join(d, 'mesh.surv.sha256')
    open(p, 'w').write(''.join(open(p).readlines()[:-1]))
    assert run(d)[0] == 1


def test_legacy_survivors_must_match_the_sidecar(tmp_path):
    d = copy(LEG2, tmp_path); p = os.path.join(d, 'mesh.surv.1'); open(p, 'wb').write(bytes(os.path.getsize(p)))
    rc, out = run(d); assert rc == 1 and 'mesh.surv.1: SHA-256 differs from mesh.surv.sha256' in out


def test_legacy_stamp_edited_after_hashing_fails(tmp_path):
    d = copy(LEG, tmp_path); edit_text(os.path.join(d, 'mesh.done'), 'cells=', 'cells=1'); rehash(d)
    assert run(d)[0] == 1


# ---------------------------------------------------------------- the 'verified' level (N = 7)
def synthetic_n7(tmp_path, T_fn, lc_fn=None, stamp_fmt='legacy'):
    """a chain directory around the stored N = 7 local certificate, with a synthetic mesh stamp (no survivors) and
    refinement record (tests the checker's logic for the 'verified' level only; the real run is on the server).
    stamp_fmt 'legacy': a stamp without hashes plus a sidecar, as the 768cefc run will have."""
    d = tmp_path / 'n7'; d.mkdir()
    shutil.copy(os.path.join(ROOT, 'certificates', 'localcert_N7_best.json'), d)
    if lc_fn: edit(str(d / 'localcert_N7_best.json'), lc_fn)
    lc = json.load(open(d / 'localcert_N7_best.json'))
    hi = max(float.fromhex(i['D_enclosure_hex'][1]) for i in lc['images'])
    T = T_fn(hi)
    line = f'N=7 T={T.hex()} MIN_EPS=0x1p-4 nf=4096 mode=rigorous keep=0x1p+0 balls=0 survivors=0 cells=1 fo=1 smear=0'
    if stamp_fmt == 'stamp': line += ' maxc=0 files=0 records=0'
    (d / 'mesh.done').write_text(line + '\n')
    if stamp_fmt == 'legacy':
        (d / 'mesh.surv.sha256').write_text(f'# stamp mesh.done sha256 {sha(d / "mesh.done")}  hashed 2026-11-04T00:00:00Z\n')
    rf = dict(N=7, T=T, T_hex=T.hex(), input_cells=0, failed=0, verified=True,
              exclusion_boxes=[dict(center=[float(x).hex() for x in i['q_center']], radius=float(lc['box_radius']).hex())
                               for i in lc['images']],
              stamp=dict(lc['stamp'], step='PROOF.md Step 3 (refinement)',
                         inputs={'x/localcert_N7_best.json': sha(d / 'localcert_N7_best.json'), 'x/mesh.done': sha(d / 'mesh.done')}))
    json.dump(rf, open(d / 'mesh.refine.json', 'w'))
    return run(str(d), N=7)


def test_verified_level_passes_with_T_above_the_enclosure(tmp_path):
    import math
    rc, out = synthetic_n7(tmp_path, lambda hi: math.nextafter(hi, math.inf)); assert rc == 0, out


def test_verified_level_passes_with_a_new_format_stamp(tmp_path):
    import math
    rc, out = synthetic_n7(tmp_path, lambda hi: math.nextafter(hi, math.inf), stamp_fmt='stamp'); assert rc == 0, out


def test_verified_level_fails_with_T_at_the_enclosure(tmp_path):
    assert synthetic_n7(tmp_path, lambda hi: hi)[0] == 1


def test_verified_level_refuses_extra_caps(tmp_path):
    """finding 2: an image with redundant worst caps is not closed at the level 'verified' (C1), even if the local
    certificate says verified (e.g. one written before the fix)"""
    import math
    def f(d): d['images'][0]['extra_caps'] = ['Cap(0, 1, 2)+']; d['images'][0]['extra_counts'] = [3]
    rc, out = synthetic_n7(tmp_path, lambda hi: math.nextafter(hi, math.inf), f)
    assert rc == 1 and 'C1: FAIL' in out and 'extra_caps' in out


def test_old_level_name_is_accepted():
    assert check_chain.LEVEL_ALIASES['certified'] == 'full'
