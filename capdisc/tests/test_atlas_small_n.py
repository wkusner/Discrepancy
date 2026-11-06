"""jobs/atlas_small_n.py: the closed forms of N = 1..4 and their atlas export (strata of the minimizer sets).

What is tested: the exact arithmetic used for the checks; the committed input file is what the script writes and
passes the exact checks; the export (written to a temporary folder) is deterministic, its observation ids recompute
by the rule of spec 7.1, every value is cross-checked by sphcap.discrepancy (exact and bruteforce) against the
closed form, the numerical probe agrees with the stated transverse type at every sample, the strata form a poset
(closure transitive, dimension strictly smaller below), every sample and closure reference resolves, and no stored
text uses the restricted word."""
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'jobs'))

import atlas_small_n as SN  # noqa: E402
import atlas_pipeline as P  # noqa: E402
from sphcap import gram as GR  # noqa: E402
from sphcap.discrepancy import discrepancy  # noqa: E402


def test_quadratic_field_arithmetic():
    QF, F = SN.QF, SN.F
    r = QF(0, 1, 105)
    assert (r * r - 105).sign() == 0
    assert str((13 - r) / 32) == '13/32 - 1/32*sqrt(105)'
    assert ((13 - r) / 32).sign() > 0 and (r - 11).sign() < 0 and (r - 10).sign() > 0
    assert abs(float((19 + r) / 64) - (19 + 105 ** 0.5) / 64) < 1e-15
    x, y = SN.n4_xy(F(1, 2))
    assert str(y) == '1/4 + 1/28*sqrt(21)' and (x + y - F(1, 2)).sign() == 0


def test_closed_forms_match_gram_py():
    xs, ys, zs = GR.N4_ARC_END_SYMMETRIC
    xe, ye, ze = GR.N4_ARC_END_EDGE
    by = {c['name']: c for c in SN.classes()}
    assert np.allclose([by['N4_arc_end_symmetric']['params'][k] for k in 'xyz'], [xs, ys, zs], atol=1e-15)
    assert np.allclose([by['N4_arc_end_edge']['params'][k] for k in 'xyz'], [xe, ye, ze], atol=1e-15)
    for c in by.values():
        if c['N'] == 4:
            p = c['params']
            assert abs(GR.n4_cubic(p['x'], p['y'], p['z'])) < 1e-14
            assert abs(GR.disphenoid_discrepancy(p['x'], p['y'], p['z']) - 3 / 8) < 1e-14
        if c['N'] == 3:
            assert GR.n3_is_minimiser_parameters(c['params']['a'], c['params']['b'], tol=1e-12)


def test_input_file_is_what_the_script_writes_and_passes_the_exact_checks():
    blob = open(os.path.join(ROOT, SN.INPUT), 'rb').read()
    assert P.canon(json.loads(blob.decode('utf-8'))) == P.canon(SN.input_doc())
    assert len(SN.check_classes()) == len(SN.classes()) == 16


@pytest.fixture(scope='module')
def export(tmp_path_factory):
    out = str(tmp_path_factory.mktemp('small_n'))
    docs = SN.run(out, log=lambda *a: None)
    return out, docs


def test_export_is_deterministic(export, tmp_path):
    out, _ = export
    again = str(tmp_path / 'again')
    SN.run(again, log=lambda *a: None)
    for N in SN.NS:
        for f in ('observations.jsonl', 'observations.json', 'manifest.json'):
            assert open(os.path.join(out, 'n%d' % N, f), 'rb').read() == open(os.path.join(again, 'n%d' % N, f),
                                                                              'rb').read(), (N, f)


def test_observation_ids_and_records(export):
    out, docs = export
    blob = open(os.path.join(ROOT, SN.INPUT), 'rb').read()
    fsha = P.sha_bytes(blob)
    for N in SN.NS:
        lines = [json.loads(l) for l in open(os.path.join(out, 'n%d' % N, 'observations.jsonl'))]
        assert [l['seq'] for l in lines] == list(range(1, len(lines) + 1))
        for l in lines:
            assert l['source']['file_sha256'] == fsha
            assert P.obs_id(fsha, l['source']['record_index'], l['raw']) == l['obs_id']
        d = docs[N]
        ids = {r['obs_id'] for r in d['observations']} | {f['obs_id'] for f in d['families']}
        assert ids == {l['obs_id'] for l in lines}
        kinds = [l['kind'] for l in lines]
        assert kinds == sorted(kinds, key=lambda k: k == 'family')       # finds first, then the strata


def test_values_cross_checked(export):
    _, docs = export
    want = {1: 1.0, 2: 0.5, 3: 0.5, 4: 0.375}
    for N, d in docs.items():
        for r in d['observations']:
            lv = {v['level'] for v in r['value']}
            assert lv == {0, 3}
            v3 = [v for v in r['value'] if v['level'] == 3][0]
            assert v3['status'] == 'proved (manuscript)' and v3['proof']['kind'] == 'manuscript'
            assert float(SN.F(v3['value']['closed_form'])) == want[N]
            x = [v for v in r['value'] if v.get('id') == 'v0x'][0]['cross_check']
            assert abs(x['exact'] - want[N]) < 1e-12 and abs(x['bruteforce'] - want[N]) < 1e-12
            X = np.array(r['X'])
            assert abs(discrepancy(X, 'exact').value - want[N]) < 1e-12
            if N >= 3:
                G = np.array([[float(eval(e.replace('sqrt', 'np.sqrt'))) for e in row] for row in r['gram']['entries']])
                assert np.abs(G - X @ X.T).max() < 1e-14


def test_probe_agrees_with_the_stated_type(export):
    _, docs = export
    for N in (3, 4):
        for f in docs[N]['families']:
            for p in f['probe']['results']:
                assert p['consistent'], (f['name'], p)
                assert p['index'] == 0
                stated = {k: [t['value'] for t in f['tags'][k] if t['level'] in (0, 3)] for k in ('coindex', 'nullity')}
                assert set(stated['coindex']) == {p['coindex']} and set(stated['nullity']) == {p['nullity']}, f['name']
            t = f['tags']
            amb, dim = t['ambient_dim'][0]['value'], f['dimension'][0]['value']
            assert amb == 2 * N - 3 and t['codim'][0]['value'] == amb - dim
            assert t['index'][0]['value'] + t['coindex'][0]['value'] + t['nullity'][0]['value'] == amb - dim
            assert {x['value'] for x in t['stratum_dim']} == {dim}


def test_strata_form_a_poset(export):
    _, docs = export
    for N in (3, 4):
        fams = {f['name']: f for f in docs[N]['families']}
        dim = {k: f['dimension'][0]['value'] for k, f in fams.items()}
        below = {k: {c['name'] for c in f['closure']} for k, f in fams.items()}
        for k, lower in below.items():
            for m in lower:
                assert m in fams and dim[m] < dim[k], (k, m)
                assert below[m] <= lower, (k, m)          # closure is transitive
        names = {r['name'] for r in docs[N]['observations']}
        for f in fams.values():
            assert {s['name'] for s in f['samples']} <= names
            assert f['generic_sample'] if 'generic_sample' in f else True
    assert len(docs[3]['families']) == 7 and len(docs[4]['families']) == 3
    # the poset of N = 3 is the face poset of the fundamental domain (a triangle)
    by = {f['name']: f for f in docs[3]['families']}
    assert {c['name'] for c in by['N3_S']['closure']} == {'N3_A', 'N3_Fp', 'N3_Fm', 'N3_C', 'N3_M', 'N3_E'}


def test_corners_are_one_configuration(export):
    """the three corners of the (a, b) region are relabellings of each other: equal sorted Gram entries"""
    _, docs = export
    cs = [r for r in docs[3]['observations'] if r['name'].startswith('N3_corner_')]
    sig = [sorted(np.round(np.array(r['X']) @ np.array(r['X']).T, 12).ravel()) for r in cs]
    assert sig[0] == sig[1] == sig[2]


def test_no_restricted_word(export):
    out, _ = export
    pat = re.compile(r'[Cc][Ee][Rr][Tt][Ii][Ff](?![Ii][Cc][Aa][Tt][Ee][Ss]?(?![A-Za-z]))')
    for N in SN.NS:
        for f in ('observations.jsonl', 'observations.json'):
            assert not pat.search(open(os.path.join(out, 'n%d' % N, f)).read()), (N, f)
    assert not pat.search(open(os.path.join(ROOT, SN.INPUT)).read())
