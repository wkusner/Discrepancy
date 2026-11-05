"""Tests of the atlas pipeline (jobs/atlas_pipeline.py, jobs/atlas_export.py, jobs/atlas_stuck.py).

Numerical tooling, not part of the proof chain.  Run:  python -m pytest tests/test_atlas_pipeline.py -q
"""
import glob, hashlib, json, os, shutil, subprocess, sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'jobs'))
import atlas_export as E          # noqa: E402
import atlas_pipeline as P        # noqa: E402
import atlas_stuck as S           # noqa: E402

SCHEMA_DIR = os.path.join(ROOT, 'tests', 'atlas_schema_v02')
SEARCH = os.path.join(ROOT, 'sphcap', 'data', 'search')
quiet = lambda *a, **k: None


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def tree(d):
    return {os.path.relpath(p, d): sha(p) for p in sorted(glob.glob(os.path.join(d, '**', '*'), recursive=True)) if os.path.isfile(p)}


def obs_validator():
    jsonschema = pytest.importorskip('jsonschema')
    referencing = pytest.importorskip('referencing')
    reg = referencing.Registry()
    for p in sorted(glob.glob(os.path.join(SCHEMA_DIR, '*.schema.json'))):
        s = json.load(open(p))
        jsonschema.Draft202012Validator.check_schema(s)
        reg = reg.with_resource(s['$id'], referencing.Resource.from_contents(s))
    return jsonschema.Draft202012Validator(json.load(open(os.path.join(SCHEMA_DIR, 'observation.schema.json'))), registry=reg)


@pytest.fixture(scope='module')
def small(tmp_path_factory):
    """the pipeline for N = 5..8 (inputs: critical_configurations.json, atlas_N5_levels.json, certificates)"""
    out = str(tmp_path_factory.mktemp('export'))
    docs = P.run([5, 6, 7, 8], out, workers=2, log=quiet)
    return out, docs


def write_search(d, rows, N=9):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'search_N{N}.json'), 'w') as f:
        json.dump(rows, f, indent=1)


# ------------------------------------------------------------------ the log in spec v0.2 form
def test_observation_log_validates_against_spec_schema(small):
    out, _ = small
    v = obs_validator()
    n = 0
    for p in sorted(glob.glob(os.path.join(out, 'n*', 'observations.jsonl'))):
        for line in open(p):
            o = json.loads(line)
            errs = sorted(v.iter_errors(o), key=lambda e: list(e.absolute_path))
            assert not errs, f'{p}: ' + '; '.join(f"{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in errs[:5])
            assert o['obs_id'] == P.obs_id(o['source']['file_sha256'], o['source']['record_index'], o['raw'])
            n += 1
    assert n == 2 + 3 + 3 + 1 + 3      # N5: 3 classes + 3 atlas levels; N6: 2; N7: 1; N8: 3


def test_negative_schema_cases(small):
    out, _ = small
    v = obs_validator()
    o = json.loads(open(os.path.join(out, 'n7', 'observations.jsonl')).readline())
    for mutate in (lambda o: o['source'].update(file_name='/an/absolute/path/x.json'),
                   lambda o: o['ingested_by'].update(note='certified result'),
                   lambda o: o.update(obs_id='o-XYZ'),
                   lambda o: o.update(extra=1)):
        bad = json.loads(json.dumps(o)); mutate(bad)
        assert list(v.iter_errors(bad)), 'a mutated line was accepted'


def test_committed_logs_validate():
    """the logs committed in the repository (if any) validate too"""
    v = obs_validator()
    files = sorted(glob.glob(os.path.join(P.EXPORT, 'n*', 'observations.jsonl')))
    if not files:
        pytest.skip('no committed observation logs')
    for p in files:
        seqs = []
        for line in open(p):
            o = json.loads(line)
            assert not list(v.iter_errors(o)), p
            seqs.append(o['seq'])
        assert seqs == list(range(1, len(seqs) + 1)), f'{p}: seq not 1..n'


# ------------------------------------------------------------------ idempotent, manifest
def test_idempotent(small, tmp_path):
    out, _ = small
    before = tree(out)
    P.run([5, 6, 7, 8], out, workers=2, log=quiet)
    assert tree(out) == before, 'a second run with the same inputs changed a file'
    # and a fresh run elsewhere gives the same bytes
    other = str(tmp_path / 'again')
    P.run([5, 6, 7, 8], other, workers=1, log=quiet)
    assert tree(other) == before


def test_manifest(small):
    out, _ = small
    m = json.load(open(os.path.join(out, 'n5', 'manifest.json')))
    files = {i['file']: i for i in m['inputs']}
    for f in ('sphcap/data/critical_configurations.json', 'sphcap/data/atlas/atlas_N5_levels.json',
              'certificates/gramcert_N5_best.json', 'certificates/localcert_N5_best.json'):
        assert f in files and files[f]['sha256'] == sha(os.path.join(ROOT, f))
    assert set(m['generated_by']['scripts']) >= {'jobs/atlas_pipeline.py', 'jobs/atlas_export.py', 'jobs/atlas_stuck.py', 'sphcap/stuck.py'}
    for f, h in m['outputs'].items():
        if f != 'observations.json':          # the manifest is written after observations.json; both are checked here
            assert h == sha(os.path.join(out, 'n5', f))
    assert m['outputs']['observations.json'] == sha(os.path.join(out, 'n5', 'observations.json'))
    text = open(os.path.join(out, 'n5', 'manifest.json')).read()
    assert not any(s in text for s in ('/home/', '/Users/', '/root/', '/tmp/'))


# ------------------------------------------------------------------ geometry, Gram matrix, certificates
def test_cap_geometry_and_gram(small):
    _, docs = small
    for N, doc in docs.items():
        for o in doc['observations']:
            X = np.array(o['X']); D = o['value'][0]['value']
            G = np.array(o['gram_numerical']['value'])
            assert o['gram_numerical']['level'] == 0 and np.allclose(G, X @ X.T, atol=0) and np.allclose(np.diag(G), 1)
            for e in o['active']['hyperedges']:
                if e.get('point_cap'):
                    continue
                c = e['cap']; w = np.array(c['center']); h = c['height']
                assert abs(np.linalg.norm(w) - 1) < 1e-12 and abs(np.cos(c['angular_radius']) - h) < 1e-12
                proj = X @ w
                on = np.flatnonzero(np.abs(proj - h) <= 1e-9); inside = np.flatnonzero(proj > h + 1e-9)
                assert list(on) == e['boundary'] and list(inside) == e['interior']
                # the over-full closed cap: count/N - sigma = D
                assert abs(e['count'] / N - (1 - h) / 2 - D) < 1e-7


def test_local_enclosures_n7_n8(small):
    """N7_best: level-2 enclosure of the local minimum (one kernel, Arb).  N8_best: no level-2 value, because its
    local verification lists redundant worst caps that it does not bound (audit 2026-11-04, finding 2): level 0
    with a note."""
    _, docs = small
    o = next(o for o in docs[7]['observations'] if o['name'] == 'N7_best')
    v2 = [v for v in o['value'] if v['level'] == 2]
    assert len(v2) == 1 and v2[0]['status'] == 'interval-verified' and 'no global claim' in v2[0]['note']
    assert v2[0]['kernels'] == ['Arb']
    lo, hi = map(float, v2[0]['value']['interval'])
    # the enclosure is of the local minimum value; D at the stored double configuration is at least that
    # and exceeds it only by rounding (2e-14 for N = 7)
    assert lo <= o['value'][0]['value'] <= hi + 1e-12
    o8 = next(o for o in docs[8]['observations'] if o['name'] == 'N8_best')
    assert [v['level'] for v in o8['value']] == [0]
    assert 'local verification incomplete (audit 2026-11-04)' in o8['value'][0]['note']
    # the N = 8 local verification file lists redundant caps on every image
    l = json.load(open(os.path.join(ROOT, 'certificates', 'localcert_N8_best.json')))
    assert all(im.get('extra_caps') for im in l['images'])
    # N = 8 second and third have no local verification
    assert all(len(o['value']) == 1 and 'audit' not in o['value'][0]['note']
               for o in docs[8]['observations'] if o['name'] != 'N8_best')


def test_kernel_names_on_proved_descriptions(small):
    """level-2/3 descriptions name their kernels: N5/N6 value level 2 and the active hypergraph by Arb + mpmath.iv,
    value level 3 and the Gram matrix by sympy + FLINT"""
    _, docs = small
    for N, name in ((5, 'N5_best'), (6, 'N6_best')):
        o = next(o for o in docs[N]['observations'] if o['name'] == name)
        by = {v['level']: v for v in o['value']}
        assert by[2]['kernels'] == ['Arb', 'mpmath.iv'] and by[3]['kernels'] == ['sympy', 'FLINT']
        assert o['gram']['kernels'] == ['sympy', 'FLINT'] and o['active']['kernels'] == ['Arb', 'mpmath.iv']
        assert o['active']['level'] == 2
    for N in (5, 6, 7, 8):
        for o in docs[N]['observations']:
            for v in o['value']:
                assert (v['level'] >= 2) == ('kernels' in v)


def test_no_forbidden_word(small):
    """'certified' appears nowhere (it is reserved for level 4, and nothing here is level 4); in the free text of the
    spec log not even 'certif' (the schema rejects it; the folder name certificates/ is cited as it is)"""
    out, _ = small
    for p in glob.glob(os.path.join(out, 'n*', '*')):
        assert 'certified' not in open(p).read().lower(), p
    for p in glob.glob(os.path.join(out, 'n*', 'observations.jsonl')):
        for line in open(p):
            assert 'certif' not in json.loads(line)['ingested_by'].get('note', '').lower()


# ------------------------------------------------------------------ add, never overwrite
def test_passes_add_never_overwrite(tmp_path):
    rows = json.load(open(os.path.join(SEARCH, 'search_N9.json')))
    sd1, sd2, out = str(tmp_path / 's1'), str(tmp_path / 's2'), str(tmp_path / 'x')
    # pass 1: the records 1..3 of the real file (its best is the file's record 1), each found once
    p1 = [dict(r, hits=1) for r in rows[1:4]]
    write_search(sd1, p1)
    P.run([9], out, search_dir=sd1, stuck=False, log=quiet,
          pass_info=dict(seed=5, rounds=3, workers=1, runs_per_task=2, resumes_previous=False))
    log1 = open(os.path.join(out, 'n9', 'observations.jsonl')).read()
    d1 = json.load(open(os.path.join(out, 'n9', 'observations.json')))
    assert [o['search']['pass'] for o in d1['observations']] == [1, 1, 1]
    assert d1['convergence']['found_once'] is True and 'not converged' in d1['convergence']['converged_note']
    assert 'not converged' in d1['observations'][0]['search']['convergence']
    # pass 2 resumes pass 1 (cumulative hits): finds the better record 0 once, and record 1 four more times;
    # record 3 is no longer among the exported records
    p2 = [dict(rows[0], hits=1), dict(p1[0], hits=5), dict(p1[1], hits=1)]
    write_search(sd2, p2)
    log_lines = os.path.join(tmp_path, 'round.log')
    with open(log_lines, 'w') as f:
        for r, b in enumerate((0.2170, 0.21695, 0.216890310773870), 1):
            f.write(f'[ 10s] N=9 round {r}/3: 40 classes; best {b:.12f} (hits 1); next []\n')
    P.run([9], out, search_dir=sd2, stuck=False, log=quiet, search_log=log_lines,
          pass_info=dict(seed=6, rounds=3, workers=1, runs_per_task=2, resumes_previous=True))
    log2 = open(os.path.join(out, 'n9', 'observations.jsonl')).read()
    assert log2.startswith(log1), 'the log was rewritten, not appended'
    d2 = json.load(open(os.path.join(out, 'n9', 'observations.json')))
    obs = d2['observations']
    assert len(obs) == 6 and [o['search']['pass'] for o in obs] == [1, 1, 1, 2, 2, 2]
    old = {o['obs_id']: o for o in d1['observations']}
    for o in obs[:3]:      # every pass-1 observation is still there, unchanged (the convergence note follows the best)
        assert [d for d in P.compare(old[o['obs_id']], o) if d[0] != '/search/convergence'] == []
    new_best, again1, again2 = obs[3], obs[4], obs[5]
    assert 'same_configuration_as' not in new_best['search']
    assert new_best['name'] == 'search_N9.pass2#0' and again1['name'] == 'search_N9.pass2#1'
    assert again1['search']['same_configuration_as'] == obs[0]['name'] and again1['search']['hits_in_pass'] == 4
    assert again2['search']['same_configuration_as'] == obs[1]['name'] and again2['search']['hits_in_pass'] == 0
    assert new_best['search']['hits_in_pass'] is None        # resumed pass, not seen before: unknown
    passes = d2['search_passes']
    assert [p['pass'] for p in passes] == [1, 2] and passes[1]['improving_in_last_round'] is True
    assert abs(passes[1]['best_by_round'][-1] - 0.21689031077387) < 1e-11
    c = d2['convergence']
    assert c['best'] == [new_best['name']] and c['found_in_pass'] == 2 and c['found_once'] is True
    assert 'still improving' in c['converged_note'] and 'lower value than the earlier passes' in c['converged_note']
    assert 'convergence' not in obs[0]['search']                 # the note moved to the new best
    # a third run with the same inputs adds nothing
    before = tree(out)
    P.run([9], out, search_dir=sd2, stuck=False, log=quiet)
    assert tree(out) == before


def test_unknown_pass_info_is_recorded_as_not_recorded(tmp_path):
    rows = json.load(open(os.path.join(SEARCH, 'search_N9.json')))
    sd, out = str(tmp_path / 's'), str(tmp_path / 'x')
    write_search(sd, rows[:1])
    P.run([9], out, search_dir=sd, stuck=False, log=quiet)
    p = json.loads(open(os.path.join(out, 'n9', 'passes.jsonl')).readline())
    assert p['seed'] is None and p['runs'] is None and p['source'].startswith('not recorded')


def test_known_passes_of_repository_files():
    """the provenance table covers the search files in the repository (by SHA-256) and names no machine"""
    for f in glob.glob(os.path.join(SEARCH, 'search_N*.json')):
        info = P.KNOWN_PASSES.get(sha(f))
        assert info is not None, f'{f}: no provenance recorded'
        for word in ('server', 'laptop', 'desktop', 'host', 'machine'):
            assert word not in info['source'].lower()


# ------------------------------------------------------------------ the stuck-point classes: cache, resume, parallel
def test_stuck_cache_resumes(tmp_path, monkeypatch):
    calls = []
    def fake_block(X):
        calls.append(len(X)); return dict(level=0, status='numerical', script='sphcap/stuck.py', **{'class': 'sharp minimum'}), None
    monkeypatch.setattr(S, 'block', fake_block)
    cache = str(tmp_path / 'stuck.jsonl')
    Xs = [np.eye(3).tolist() + [[0.6, 0.8, 0.0]], np.eye(3).tolist() + [[0.0, 0.6, 0.8]]]
    S.classify_all([(cache, Xs[0])], workers=1, log=quiet)
    assert calls == [4]
    with open(cache, 'a') as f:
        f.write('{"key": "trunc')          # an interrupted write
    res = S.classify_all([(cache, X) for X in Xs], workers=1, log=quiet)
    assert calls == [4, 4], 'the cached configuration was classified again'
    assert set(res) == {(cache, S.x_key(X)) for X in Xs}
    S.compact_cache(cache, [S.x_key(X) for X in Xs])
    lines = open(cache).read().splitlines()
    assert len(lines) == 2 and all(json.loads(l)['version'] == S.classifier_version() for l in lines)


def test_stuck_parallel_equals_serial(tmp_path):
    cc = json.load(open(os.path.join(ROOT, 'sphcap/data/critical_configurations.json')))['classes']
    Xs = [E.B.normalize(np.array(c['points'])).tolist() for c in cc if c['N'] in (5, 6)]
    a = S.classify_all([(str(tmp_path / 'a.jsonl'), X) for X in Xs], workers=1, log=quiet)
    b = S.classify_all([(str(tmp_path / 'b.jsonl'), X) for X in Xs], workers=2, log=quiet)
    assert [a[(str(tmp_path / 'a.jsonl'), S.x_key(X))] for X in Xs] == [b[(str(tmp_path / 'b.jsonl'), S.x_key(X))] for X in Xs]


def test_classes_reproduce_committed_export(small):
    """N = 5..8: the stuck-point classes and every value of the committed export are reproduced (where the
    committed export was written by the same code; the provenance field source/commit and appended descriptions
    are not compared; floats to a relative 1e-9, since the LP solver of another scipy version may differ in the
    last digits)"""
    out, docs = small
    for N in (5, 6, 7, 8):
        ref = json.load(open(os.path.join(P.EXPORT, f'n{N}', 'observations.json')))
        new = {o['name']: o for o in docs[N]['observations']}
        for o in ref['observations']:
            diffs = [d for d in P.compare(o, new[o['name']], rtol=1e-9) if not d[0].endswith('/source/commit')
                     # finite-difference eigenvalues (h = 1e-4, rounded to 8 digits) at noise level; counts stay exact
                     and not ('hessian_eigenvalues' in d[0] and abs(d[1] - d[2]) <= 1e-6)]
            assert diffs == [], (o['name'], diffs[:3])


def test_legacy_export_refuses_to_overwrite_a_logged_export(tmp_path, monkeypatch):
    d = tmp_path / 'n9'; d.mkdir()
    (d / 'observations.jsonl').write_text('{}\n')
    monkeypatch.setattr(E, 'EXPORT', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['atlas_export.py'])
    with pytest.raises(SystemExit):
        E.main()
