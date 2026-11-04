"""The capdisc side of the atlas as one reproducible command (NOT TRUSTED; everything it computes is numerical,
level 0, except the descriptions copied from the certificates).

    python jobs/atlas_pipeline.py N [N ...] [--workers K] [--pass-info JSON] [--search-log FILE]
                                            [--search-dir DIR] [--out DIR] [--no-stuck]

For each N, from the inputs

    sphcap/data/critical_configurations.json      classes with this N (N = 5..8)
    sphcap/data/atlas/atlas_N<N>_levels.json       if present (N = 5)
    sphcap/data/search/search_N<N>.json            if present (N >= 9), output of jobs/search_parallel.py
    certificates/{gramcert,localcert,xcheck_local}_<name>.json, certificates/xcheck_roots.json   (N = 5..8)

it writes, under export/sphere/d3/cap_discrepancy_sup/n<N>/ (or DIR/n<N>/ with --out):

observations.jsonl  the observation log in the form of atlas spec v0.2 (schema/observation.schema.json): one line
                    per record found, append-only.  A record is identified by obs_id = "o-" + 16 hex of SHA-256 of
                    the canonical JSON of {file_sha256, record_index, raw}; a record that is already in the log is
                    not added again, so a re-run with the same inputs adds nothing.  When a later search pass
                    replaces search_N<N>.json, its records are new observations and are appended; the earlier ones
                    stay (with their raw records, coordinates included), so nothing found is ever lost.
passes.jsonl        one line per search pass (one version of search_N<N>.json): pass number, file SHA-256, seed,
                    rounds, workers, runs per task, runs, whether it resumed the previous pass (hits cumulative),
                    the best value per round if a log was given.  No machine names.
observations.json   the export (jobs/atlas_export.py observation(), as before, plus the cap geometry and the Gram
                    matrix), rebuilt from the log in log order, with for each observation its obs_id, the search
                    pass that found it, its hits, the earlier observation of the same configuration if any, and
                    the stuck-point class (jobs/atlas_stuck.py).  The best configuration gets a convergence note
                    when it was found only once or when its pass was still improving in its last round.
stuck.jsonl         cache of the stuck-point classes (resumable; see jobs/atlas_stuck.py).
manifest.json       inputs with SHA-256 and the commit that last changed them, the code commit and the SHA-256 of
                    every script involved, library versions, and the SHA-256 of the outputs.

Idempotent: the outputs depend only on the inputs, the log and the code; no wall-clock time is written, and a second
run with the same inputs rewrites the same bytes.

--pass-info   JSON object for a *new* search file of these N, e.g.
              '{"seed": 1, "rounds": 15, "workers": 4, "runs_per_task": 12, "resumes_previous": true}';
              without it a new search file gets the provenance recorded below for the files already in the
              repository (by SHA-256), or null fields marked "not recorded".
--search-log  the stdout of search_parallel.py for that pass (lines "N=.. round r/R: ... best D (hits h) ..."):
              gives the best value per round and so whether the pass was still improving in its last round.
--search-dir  where search_N<N>.json is read (default sphcap/data/search).
--no-stuck    do not run the stuck-point classification: use the cached classes if every observation of the N has
              one, else write the N without classes.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse, hashlib, json, platform, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import scipy
import atlas_export as E
import atlas_stuck as S

SCHEMA_VERSION = '0.2.0'
EXPORT = os.path.join(ROOT, 'export', 'sphere', 'd3', E.FUNCTIONAL['id'])
CONTAINER = dict(id='sphere', d=3, params=dict(functional=E.FUNCTIONAL['id']))
# the tolerances of the export, in the slots of the spec (whose names come from the kissing problem):
# contact = incidence of a point on a cap circle (atlas_export.TOL); stress_support = support threshold of the weight
# space (atlas_export.weight_space); eigenvalue = relative threshold of the second-order counts (atlas_export).
TOLERANCES = dict(contact=1e-9, stress_support=1e-9, eigenvalue=1e-5)
CODE = ['jobs/atlas_pipeline.py', 'jobs/atlas_export.py', 'jobs/atlas_stuck.py',
        'sphcap/balance.py', 'sphcap/stuck.py', 'sphcap/__init__.py']
CC = 'sphcap/data/critical_configurations.json'

# Provenance of the search files already in the repository, by SHA-256, as recorded in the commit messages and
# scripts that added them (the search JSON itself does not record it).  Fields not recorded there are null.
KNOWN_PASSES = {}
_p_9_12 = dict(seed=0, rounds=12, workers=8, runs_per_task=25, resumes_previous=None, improving_in_last_round=None,
               source='run_night.sh and jobs/search_parallel.py at commit c6c15c5 (defaults: 8 workers, 25 runs per '
                      'task; seed 0: no --seed option yet, the streams equal those of seed 0); files added in abcc7c6. '
                      'Whether an earlier results/search_N<N>.pkl was resumed is not recorded.')
_p_13_20 = dict(seed=0, rounds=10, workers=8, runs_per_task=12, resumes_previous=False, improving_in_last_round=None,
                source='jobs/run_atlas_search.sh and commit 590fe7d (first pass: 10 rounds x 8 workers x 12 runs, seed 0)')
_p_21_22 = dict(seed=None, rounds=10, workers=2, runs_per_task=12, resumes_previous=False, improving_in_last_round=None,
                source='commit c3ae1de (first pass: 10 rounds, 2 workers, 12 runs per task, 240 runs); seed not recorded')
_p_24_25 = dict(_p_21_22, workers=4, source='commit c3ae1de (first pass: 10 rounds, 4 workers, 12 runs per task, '
                                           '480 runs); seed not recorded')
_p_23 = dict(_p_24_25, improving_in_last_round=True,
             source='commit 86088fd (first pass: 10 rounds x 4 workers x 12 runs; "best still improving at round 10"); '
                    'seed not recorded')
for _sha, _p in [
        ('7b3dd5650d1f5abc4fdc2d409426e79832e1b14b9a22f641b38351d831df33b4', _p_9_12),
        ('c4e017ba790159ca9e54ddfa4563bc3bbfaad0f03f98289a2e7da810b2d85b3f', _p_9_12),
        ('035f17158e10be9a2db1df78c43fe721e91302854d842d4358b331737e69ecef', _p_9_12),
        ('92ecd5a04007c0fedf685e70159beda533980778fa7becf87b4621674419c604', _p_9_12),
        ('c60499ad6da2a667fe0925ad915e0c4d7833a4d1633ca1d688af44d70b02a4db', _p_13_20),
        ('77ee750d74b93f2e6ac1e479884d38f0e3de967fd59bded8b3e15d4af9b7b8de', _p_13_20),
        ('8cd492b79363a795a5ba4d5df9e4a6af591168e6b307793d66d3594dee9316c0', _p_13_20),
        ('10f870eb4c378e6b3b498b8599e574640bb4949102e73ad8777f32f3ad39b73c', _p_13_20),
        ('bbc886f9df6c2570beea376401e96f82b33525d65bcf0e63400a58caa63994ab', _p_13_20),
        ('2ded2fa6e9abac3e6627a7402d99ecdcd725b73ad1d9d5fe28fd5c0e1dad2989', _p_13_20),
        ('b3129d914656d177a6c7fdf3dd4576d4ad3bf231693755b08a35c5be54ae1ca9', _p_13_20),
        ('41827fa1a6a25682405f8799108bee9952b76b029de6d6bc02d0f2093ac572f7', _p_13_20),
        ('57498123a90732833be1c25ffee402c3f25e7cd4f4563e7e98c9ea6ecb2173d3', _p_21_22),
        ('f245ecec1a52f17cefb0ea5bd580df61f59c6c6ca846cfe50b7b761cfa74e45c', _p_21_22),
        ('1ee82b1fb616cda80f76dd67ee39a7ef8edf3a8b46fef1c778bfc562100ad49c', _p_23),
        ('9567eba0c64d0153036e83d21055a03f6935b2b500b6aee1ebe1c2b27864a216', _p_24_25),
        ('10400777cb2637fb339ec51e9db00166fbc6d01c7bd465ae8c88a535e4387d54', _p_24_25)]:
    KNOWN_PASSES[_sha] = _p
PASS_FIELDS = ('seed', 'rounds', 'workers', 'runs_per_task', 'resumes_previous', 'improving_in_last_round', 'source')


# ------------------------------------------------------------------ small tools
def canon(obj):
    """canonical JSON of spec v0.2 section 1.5: sorted keys, no spaces, shortest round-trip floats, UTF-8"""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    return sha_bytes(open(path, 'rb').read())


def obs_id(file_sha256, record_index, raw):
    return 'o-' + sha_bytes(canon(dict(file_sha256=file_sha256, record_index=record_index, raw=raw)).encode())[:16]


def git(*args):
    try:
        r = subprocess.run(['git'] + list(args), cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    except OSError:
        return None


def rel(path):
    """repository-relative path, or the bare file name for a file outside the repository"""
    p = os.path.relpath(os.path.abspath(path), ROOT)
    return os.path.basename(path) if p.startswith('..') else p


def file_commit(path):
    """the commit that last changed the file, if the file is committed and unmodified; else 'unknown'"""
    p = rel(path)
    if p == os.path.basename(path) and not os.path.exists(os.path.join(ROOT, p)):
        return 'unknown'
    if git('status', '--porcelain', '--', p):
        return 'unknown'
    c = git('log', '-1', '--format=%h', '--', p)
    return c or 'unknown'


def code_state():
    files = [p for p in CODE if os.path.exists(os.path.join(ROOT, p))]
    c = git('log', '-1', '--format=%h', '--', *files) or 'unknown'
    dirty = bool(git('status', '--porcelain', '--', *files))
    return dict(commit='unknown' if dirty else c, last_commit=c, modified=dirty,
                scripts={p: sha_file(os.path.join(ROOT, p)) for p in files})


def versions():
    try:
        from importlib.metadata import version
        js = version('jsonschema')
    except Exception:
        js = 'unknown'
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, jsonschema=js)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def write_text(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


# ------------------------------------------------------------------ inputs
def inputs_for(N, search_dir):
    """[(path, format, [(record_index, raw, name)])] in the order of the export"""
    out = []
    cc = os.path.join(ROOT, CC)
    recs = [(c['name'], c, c['name']) for c in json.load(open(cc))['classes'] if c['N'] == N and 'points' in c]
    if recs:
        out.append((cc, 'critical-configurations', recs))
    for path, fmt in ((os.path.join(ROOT, 'sphcap', 'data', 'atlas', f'atlas_N{N}_levels.json'), 'atlas-levels'),
                      (os.path.join(search_dir, f'search_N{N}.json'), 'search')):
        if not os.path.exists(path):
            continue
        d = json.load(open(path)); rows = d['classes'] if isinstance(d, dict) else d
        stem = os.path.basename(path).split('.')[0]
        recs = [(k, r, f'{stem}#{k}') for k, r in enumerate(rows) if 'points' in r]
        if recs:
            out.append((path, fmt, recs))
    return out


def parse_search_log(path, N):
    """best value per round from the stdout of search_parallel.py"""
    pat = re.compile(r'N=(\d+) round (\d+)/(\d+): .*?best ([0-9.eE+-]+) \(hits (\d+)\)')
    best = {}
    for line in open(path):
        m = pat.search(line)
        if m and int(m.group(1)) == N:
            best[int(m.group(2))] = float(m.group(4))
    return [best[r] for r in sorted(best)]


# ------------------------------------------------------------------ the log
def ingest(N, outdir, search_dir, pass_info=None, search_log=None, code=None):
    """append the new records of the inputs to observations.jsonl and passes.jsonl; returns (log, passes, inputs)"""
    os.makedirs(outdir, exist_ok=True)
    logf, passf = os.path.join(outdir, 'observations.jsonl'), os.path.join(outdir, 'passes.jsonl')
    log, passes = read_jsonl(logf), read_jsonl(passf)
    have = {o['obs_id'] for o in log}
    code = code or code_state()
    new_lines, new_passes, inputs = [], [], []
    for path, fmt, recs in inputs_for(N, search_dir):
        fsha = sha_file(path); fcommit = file_commit(path)
        inputs.append(dict(file=rel(path), sha256=fsha, commit=fcommit, format=fmt, records=len(recs)))
        pas = None
        if fmt == 'search':
            pas = next((p for p in passes + new_passes if p['file_sha256'] == fsha), None)
            if pas is None:
                info = dict(KNOWN_PASSES.get(fsha) or {})
                if pass_info:
                    info.update(pass_info)
                if not info:
                    info = dict(source='not recorded (no --pass-info given for this file)')
                p = {'pass': len(passes) + len(new_passes) + 1, 'file': rel(path), 'file_sha256': fsha, 'file_commit': fcommit}
                for k in PASS_FIELDS:
                    p[k] = info.get(k)
                p['runs'] = (p['rounds'] * p['workers'] * p['runs_per_task']
                             if None not in (p['rounds'], p['workers'], p['runs_per_task']) else None)
                p['best_by_round'] = parse_search_log(search_log, N) if search_log else None
                if p['best_by_round'] and len(p['best_by_round']) >= 2:
                    p['improving_in_last_round'] = bool(p['best_by_round'][-1] < p['best_by_round'][-2] - 1e-12)
                new_passes.append(p); pas = p
        for idx, raw, name in recs:
            oid = obs_id(fsha, idx, raw)
            if oid in have:
                continue
            have.add(oid)
            note = f'record {name} of {rel(path)}'
            if pas is not None:
                note += (f"; search pass {pas['pass']}: seed {pas['seed']}, {pas['rounds']} rounds x {pas['workers']} "
                         f"workers x {pas['runs_per_task']} runs per task")
            line = dict(record='observation', schema_version=SCHEMA_VERSION, obs_id=oid,
                        seq=len(log) + len(new_lines) + 1, kind='find', container=CONTAINER, N=N, raw=raw,
                        source=dict(file_name=os.path.basename(path), file_sha256=fsha, record_index=idx,
                                    format='other', commit=fcommit),
                        tolerances=TOLERANCES,
                        ingested_by=dict(script='jobs/atlas_pipeline.py', commit=code['commit'], inputs=[rel(path)],
                                         note=note))
            if pas is not None:
                line['prov'] = dict(run_id=f"search_N{N}-pass{pas['pass']}", **({'seed': pas['seed']} if pas['seed'] is not None else {}))
            new_lines.append(line)
    if new_lines:
        with open(logf, 'a') as f:
            f.write(''.join(canon(l) + '\n' for l in new_lines))
    if new_passes:
        with open(passf, 'a') as f:
            f.write(''.join(canon(p) + '\n' for p in new_passes))
    # what is returned is what a later run reads back from the files (same key order, same bytes downstream)
    return log + [json.loads(canon(l)) for l in new_lines], passes + [json.loads(canon(p)) for p in new_passes], inputs


def pass_of(line, passes):
    if 'prov' not in line:
        return None
    return next((p for p in passes if p['file_sha256'] == line['source']['file_sha256']), None)


def build_records(N, log, passes):
    """the export records of the log, in seq order, without stuck-point classes"""
    recs = []
    for line in sorted(log, key=lambda l: l['seq']):
        raw, src = line['raw'], line['source']; path = line['ingested_by']['inputs'][0]
        idx = src['record_index']
        p = pass_of(line, passes)
        stem = os.path.basename(path).split('.')[0]
        # pass 1 keeps the names of the earlier exports; a later pass gets its number in the name (unique names)
        name = raw['name'] if isinstance(idx, str) else (f'{stem}#{idx}' if not p or p['pass'] == 1 else f"{stem}.pass{p['pass']}#{idx}")
        rec = E.observation(raw['points'], name, dict(file=path, file_sha256=src['file_sha256'], pointer=idx,
                                                     script='jobs/atlas_export.py', commit=src['commit'],
                                                     pipeline='jobs/atlas_pipeline.py'))
        if not raw.get('points') or rec['N'] != N:
            raise ValueError(f'{name}: not an N = {N} configuration')
        E.apply_upgrades(rec, name)
        rec['obs_id'] = line['obs_id']; rec['seq'] = line['seq']
        rec['search'] = {'pass': p['pass'] if p else None, 'hits': raw.get('hits'), 'sources': raw.get('sources')}
        if p:
            rec['search'].update({k: p[k] for k in ('seed', 'rounds', 'workers', 'runs_per_task', 'runs')})
        recs.append(rec)
    # the same configuration seen again (a later pass, or another input file): link to its first observation
    for i, r in enumerate(recs):
        for q in recs[:i]:
            if S.same_configuration(q['value'][0]['value'], q['X'], r['value'][0]['value'], r['X']):
                r['search']['same_configuration_as'] = q['search'].get('same_configuration_as', q['name'])
                break
    _hits_in_pass(recs, passes)
    return recs


def _hits_in_pass(recs, passes):
    """hits found in the observation's own pass.  The file's hits are cumulative when the pass resumed the previous
    one (jobs/search_parallel.py merges into results/search_N<N>.pkl and adds the hits); then the hits in this pass
    are the difference to the observation of the previous pass, and unknown (null) when the configuration was not
    among the exported records of the previous pass."""
    for r in recs:
        s = r['search']
        if s['pass'] is None or s['hits'] is None:
            continue
        p = next(p for p in passes if p['pass'] == s['pass'])
        if s['pass'] == 1 or p['resumes_previous'] is False:
            s['hits_in_pass'] = s['hits']
            continue
        root = s.get('same_configuration_as')
        prev = [q for q in recs if root and q['search']['pass'] == s['pass'] - 1 and q['search']['hits'] is not None
                and (q['name'] == root or q['search'].get('same_configuration_as') == root)]
        s['hits_in_pass'] = s['hits'] - prev[0]['search']['hits'] if (p['resumes_previous'] is True and prev) else None


def total_hits(group, passes):
    """hits of a configuration over all passes: the latest file's count when every later pass resumed the one
    before (cumulative counts), else the sum over passes; None when this is not recorded"""
    gs = sorted([r for r in group if r['search']['pass'] is not None], key=lambda r: r['search']['pass'])
    if not gs:
        return None
    last = max(r['search']['pass'] for r in gs); first = gs[0]['search']['pass']
    chain = [p for p in passes if first < p['pass'] <= last]
    if all(p['resumes_previous'] is True for p in chain):
        return gs[-1]['search']['hits']
    if all(p['resumes_previous'] is False for p in chain):
        return sum(r['search']['hits'] for r in gs)
    return None


def convergence(recs, passes):
    """note on the best configuration: found once, or its pass still improving in the last round"""
    srch = [r for r in recs if r['search']['pass'] is not None]
    if not srch:
        return None
    best_D = min(r['value'][0]['value'] for r in recs)
    root_of = lambda r: r['search'].get('same_configuration_as', r['name'])
    best = [r for r in recs if r['value'][0]['value'] - best_D < 1e-8]
    roots = {root_of(r) for r in best}
    group = [r for r in recs if root_of(r) in roots]
    gs = [r for r in group if r['search']['pass'] is not None]
    if not gs:
        return dict(best=[r['name'] for r in group], value=best_D, note='the best is not from a search file; no search statistics')
    total = total_hits(group, passes)
    found_once = (total == 1) if total is not None else None
    first_pass = min(r['search']['pass'] for r in gs)
    last_pass = max(p['pass'] for p in passes)
    lp = next(p for p in passes if p['pass'] == max(r['search']['pass'] for r in gs))
    improving = lp.get('improving_in_last_round')
    notes = []
    if found_once:
        notes.append('the best was found once (hits 1)')
    elif found_once is None and max(r['search']['hits'] or 0 for r in gs) == 1:
        notes.append('the best has hits 1 in every pass that found it; whether the passes shared their hit counts is '
                     'not recorded')
    if improving:
        notes.append(f"search pass {lp['pass']} was still improving in its last round ({lp['rounds']})")
    if first_pass > 1:
        earlier = [r['value'][0]['value'] for r in srch if r['search']['pass'] < first_pass]
        if earlier:
            notes.append(f'pass {first_pass} found a lower value than the earlier passes (best before: {min(earlier):.10f})')
    out = dict(best=sorted({r['name'] for r in group}), value=best_D, hits_total=total, found_once=found_once,
               improving_in_last_round=improving, found_in_pass=first_pass, passes=last_pass,
               converged_note=None)
    if found_once or improving or (found_once is None and notes):
        out['converged_note'] = ('not converged: ' + '; '.join(notes) +
                                 '. A further search may find a lower value (numerical, level 0).')
    elif notes:
        out['converged_note'] = '; '.join(notes) + '.'
    return out


# ------------------------------------------------------------------ assembly
def assemble(N, outdir, recs, passes, stuck, inputs, code):
    if stuck is not None:
        S.attach(recs, stuck)
    conv = convergence(recs, passes)
    if conv and conv['converged_note']:
        for r in recs:
            if r['name'] in conv['best']:
                r['search']['convergence'] = conv['converged_note']
    seen = {}
    for r in recs:
        seen.setdefault(r['key_basis']['graph_hash'], []).append(r['name'])
    doc = dict(functional=E.FUNCTIONAL, N=N, observations=recs,
               shared_hashes={h: v for h, v in seen.items() if len(v) > 1},
               search_passes=passes, convergence=conv,
               label='Every value is numerical (level 0, double precision) unless its level says otherwise; '
                     'stuck-point classes are numerical labels. Nothing here is a proof input.')
    write_text(os.path.join(outdir, 'observations.json'), json.dumps(doc, indent=1) + '\n')
    outs = {}
    for f in ('observations.jsonl', 'passes.jsonl', 'observations.json', 'stuck.jsonl'):
        p = os.path.join(outdir, f)
        if os.path.exists(p):
            outs[f] = sha_file(p)
    man = dict(N=N, functional=E.FUNCTIONAL['id'], schema_version=SCHEMA_VERSION,
               generated_by=dict(script='jobs/atlas_pipeline.py', commit=code['commit'],
                                 last_code_commit=code['last_commit'], code_modified=code['modified'],
                                 scripts=code['scripts'], classifier_version=S.classifier_version(),
                                 versions=versions()),
               inputs=inputs + [dict(file=c, sha256=sha_file(os.path.join(ROOT, c)), commit=file_commit(os.path.join(ROOT, c)))
                                for c in certificate_inputs(recs)],
               observations=len(recs), search_passes=len(passes), outputs=outs,
               command=f'python jobs/atlas_pipeline.py {N}')
    write_text(os.path.join(outdir, 'manifest.json'), json.dumps(man, indent=1, sort_keys=True) + '\n')
    return doc


def certificate_inputs(recs):
    out = []
    for r in recs:
        for v in r['value']:
            for e in v.get('evidence', []):
                out.append(e['file'])
        for e in (r.get('gram') or {}).get('evidence', []):
            out.append(e['file'])
        if r.get('gram'):
            out.append(r['gram']['ref']['file'])
    return sorted(set(out))


def run(Ns, outroot=EXPORT, search_dir=None, workers=None, pass_info=None, search_log=None, stuck=True, log=print):
    search_dir = search_dir or os.path.join(ROOT, 'sphcap', 'data', 'search')
    code = code_state()
    built = {}
    for N in Ns:
        outdir = os.path.join(outroot, f'n{N}')
        lg, passes, inputs = ingest(N, outdir, search_dir, pass_info, search_log, code)
        recs = build_records(N, lg, passes)
        built[N] = (outdir, recs, passes, inputs)
        log(f'N={N}: {len(recs)} observations ({len(passes)} search passes)')
    blocks = {}
    # one pool for all N (better use of the workers); with --no-stuck only the cached classes are used
    jobs = [(os.path.join(outdir, 'stuck.jsonl'), r['X']) for outdir, recs, _, _ in built.values() for r in recs]
    res = S.classify_all(jobs, workers, log=log, compute=stuck)
    for N, (outdir, recs, _, _) in built.items():
        cp = os.path.join(outdir, 'stuck.jsonl')
        keys = [S.x_key(r['X']) for r in recs]
        if all((cp, k) in res for k in keys):
            blocks[N] = [res[(cp, k)] for k in keys]
            S.compact_cache(cp, keys)
        elif stuck:
            raise RuntimeError(f'N={N}: stuck-point classes missing')
    docs = {}
    for N, (outdir, recs, passes, inputs) in built.items():
        docs[N] = assemble(N, outdir, recs, passes, blocks.get(N), inputs, code)
        c = docs[N]['convergence']
        log(f"N={N}: best {min(r['value'][0]['value'] for r in recs):.10f}"
            + (f"; {c['converged_note']}" if c and c.get('converged_note') else ''))
    return docs


def compare(old, new, path='', rtol=0.0):
    """leaves of `old` that differ in `new` or are missing there (additions in `new` are not reported):
    a list of (path, old value, new value or '<missing>').  Floats are equal when |a - b| <= rtol max(1, |a|)."""
    out = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            out += compare(old[k], new[k], f'{path}/{k}', rtol) if k in new else [(f'{path}/{k}', old[k], '<missing>')]
    elif isinstance(old, list) and isinstance(new, list) and len(old) <= len(new) and old and isinstance(old[0], dict):
        for i, (a, b) in enumerate(zip(old, new)):     # descriptions appended at the end are additions
            out += compare(a, b, f'{path}/{i}', rtol)
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new)):
            out += compare(a, b, f'{path}/{i}', rtol)
    elif isinstance(old, float) and isinstance(new, (int, float)) and not isinstance(new, bool):
        if abs(old - new) > rtol * max(1.0, abs(old)):
            out.append((path, old, new))
    elif old != new or type(old) is not type(new):
        out.append((path, old, new))
    return out


def compare_with_ref(N, ref, outroot=EXPORT, rtol=0.0):
    """compare observations.json of N with the version at git ref `ref` (observations matched by name)"""
    p = os.path.relpath(os.path.join(EXPORT, f'n{N}', 'observations.json'), os.path.join(ROOT, '..'))
    old = json.loads(subprocess.run(['git', 'show', f'{ref}:{p}'], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    new = json.load(open(os.path.join(outroot, f'n{N}', 'observations.json')))
    diffs = compare({k: v for k, v in old.items() if k != 'observations'}, new, '', rtol)
    newobs = {o['name']: o for o in new['observations']}
    for o in old['observations']:
        diffs += compare(o, newobs[o['name']], '/' + o['name'], rtol) if o['name'] in newobs else [('/' + o['name'], '<observation>', '<missing>')]
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('N', type=int, nargs='+')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--pass-info', default=None)
    ap.add_argument('--search-log', default=None)
    ap.add_argument('--search-dir', default=None)
    ap.add_argument('--out', default=EXPORT)
    ap.add_argument('--no-stuck', action='store_true')
    a = ap.parse_args(argv)
    info = json.loads(a.pass_info) if a.pass_info else None
    if info:
        bad = set(info) - set(PASS_FIELDS)
        if bad:
            ap.error(f'--pass-info: unknown fields {sorted(bad)} (allowed: {PASS_FIELDS})')
    run(a.N, a.out, a.search_dir, a.workers, info, a.search_log, not a.no_stuck)


if __name__ == '__main__':
    main()
