"""Add the stuck-point class (sphcap/stuck.py) to the atlas export (NOT TRUSTED; numerical labels, level 0).

For every observation in export/sphere/d3/cap_discrepancy_sup/n<N>/observations.json this adds

stuck_point   {class, flat_dim (if any), level 0, status numerical, script, note}; the classes are those of
              sphcap/stuck.py: sharp / strict / non-strict minimum, saddle, exit, not stationary.
              For an exit it also records where the exit leads: the descent restarted along the exit
              direction (follow_exits), the value reached, its class, and the observation of this export it
              coincides with: same value to 1e-8 and the same sorted list of pairwise inner products to 1e-6
              (invariant under rotations, reflections and relabelling; a numerical match, not a proof of
              congruence).

Records are not removed: an exit is a point the descent stopped at, which is itself a fact about the search, and
the atlas can filter on stuck_point.class.

Parallel and resumable.  The observations are classified independently in a process pool (one BLAS thread per
worker).  Each result is appended to the cache n<N>/stuck.jsonl as soon as it is computed, keyed by the SHA-256 of
the stored coordinates and the version of the classifier (SHA-256 of sphcap/stuck.py, sphcap/balance.py and
BLOCK_VERSION below); a re-run after an interruption computes only what is missing, and a change of the classifier
recomputes everything (old lines stay in the cache until it is compacted at the end of a complete run).  The result
for one observation does not depend on the order or the number of workers: classify() is called with its own
random generator, seeded 0, for each observation.

usage: python jobs/atlas_stuck.py N [N ...] [--workers K]        (default K: number of CPUs)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'          # before numpy is imported, here and in every worker
import hashlib, json, sys
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, ROOT)
from sphcap.stuck import classify, follow_exits
from sphcap import balance as B

BLOCK_VERSION = 1      # bump when block() below changes what it computes
EXPORT = os.path.join(ROOT, 'export', 'sphere', 'd3', 'cap_discrepancy_sup')


def classifier_version():
    h = hashlib.sha256()
    for p in ('sphcap/stuck.py', 'sphcap/balance.py'):
        h.update(open(os.path.join(ROOT, p), 'rb').read())
    h.update(b'block %d' % BLOCK_VERSION)
    return h.hexdigest()[:16]


def x_key(X):
    """identifier of a configuration as stored (the list of coordinates, shortest round-trip floats)"""
    return hashlib.sha256(json.dumps([[float(v) for v in row] for row in X], separators=(',', ':')).encode()).hexdigest()[:16]


def block(X):
    """the stuck_point block of one configuration, and the end point of the exit descent (or None)"""
    X = B.normalize(np.asarray(X, float))
    c = classify(X, rng=np.random.default_rng(0))
    out = dict(level=0, status='numerical', script='sphcap/stuck.py', **{'class': c['cls']})
    if 'flat_dim' in c: out['flat_dim'] = c['flat_dim']
    Y = None
    if c['cls'] == 'exit':
        D0, _ = B.active_set(X)
        Y, D1 = follow_exits(X, D0)
        c1 = classify(Y, rng=np.random.default_rng(0))
        out['exit'] = dict(rate=round(float(c['escape_rate']), 10), switched_points=c['switched'],
                           leads_to=dict(value=float(D1), **{'class': c1['cls']}))
        Y = np.asarray(Y).tolist()
    return out, Y


def _work(task):
    key, X = task
    out, Y = block(X)
    return key, out, Y


def _init_worker():
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = '1'


def append_line(path, line):
    """append one line; if an interrupted run left a partial last line, start on a new line"""
    with open(path, 'ab+') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':
                f.write(b'\n')
        f.write(line.encode() + b'\n'); f.flush(); os.fsync(f.fileno())


def read_cache(path):
    """{(key, version): (block, Y)}; a truncated last line (interrupted run) is ignored"""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        out[(r['key'], r['version'])] = (r['block'], r.get('Y'))
    return out


def compact_cache(path, keep):
    """rewrite the cache with the lines of the current version for the given keys, sorted by key (deterministic)"""
    ver = classifier_version(); cache = read_cache(path)
    lines = []
    for k in sorted(set(keep)):
        if (k, ver) in cache:
            b, Y = cache[(k, ver)]
            lines.append(json.dumps(dict(key=k, version=ver, block=b, **({'Y': Y} if Y is not None else {})),
                                    separators=(',', ':')))    # the block keeps its own key order
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(''.join(l + '\n' for l in lines))
    os.replace(tmp, path)


def classify_all(jobs, workers=None, log=print, compute=True):
    """jobs: list of (cache_path, X).  Classifies what is not in the caches yet, in parallel, appending each result
    to its cache when it arrives.  Returns {(cache_path, key): (block, Y)}; with compute=False only the cached ones."""
    ver = classifier_version()
    caches = {p: read_cache(p) for p in {p for p, _ in jobs}}
    result, todo, seen = {}, [], set()
    for p, X in jobs:
        k = x_key(X)
        if (k, ver) in caches[p]:
            result[(p, k)] = caches[p][(k, ver)]
        elif (p, k) not in seen:
            seen.add((p, k)); todo.append((p, k, X))
    if todo and compute:
        workers = workers or os.cpu_count() or 1
        log(f'stuck points: {len(todo)} to classify ({len(result)} cached), {workers} workers')
        by_key = {}
        for p, k, X in todo:
            by_key.setdefault(k, []).append(p)
        tasks = [(k, X) for k, X in {k: X for _, k, X in todo}.items()]
        # the largest configurations first: they take longest (better balance at the end of the run)
        tasks.sort(key=lambda t: -len(t[1]))
        def record(k, out, Y):
            for p in by_key[k]:
                line = json.dumps(dict(key=k, version=ver, block=out, **({'Y': Y} if Y is not None else {})),
                                  separators=(',', ':'))
                append_line(p, line)
                result[(p, k)] = (out, Y)
            log(f"  {k} N={len(dict(tasks)[k])}: {out['class']}")
        if workers == 1:
            for k, X in tasks:
                record(*_work((k, X)))
        else:
            ctx = mp.get_context('spawn')          # same behaviour on Linux and macOS
            with ctx.Pool(workers, initializer=_init_worker) as pool:
                for k, out, Y in pool.imap_unordered(_work, tasks):
                    record(k, out, Y)
    return result


def shape(X):
    X = B.normalize(np.asarray(X, float)); G = X @ X.T
    return np.sort(G[np.triu_indices(len(X), 1)])


def same_configuration(D1, X1, D2, X2):
    """numerical match: value to 1e-8 and the sorted pairwise inner products to 1e-6"""
    return abs(D1 - D2) < 1e-8 and len(X1) == len(X2) and np.abs(shape(X1) - shape(X2)).max() < 1e-6


def attach(obs, blocks):
    """write stuck_point into each observation (blocks: list parallel to obs of (block, Y)); link exits to the
    observation they lead to (the first in the order of obs)"""
    order = ['level', 'status', 'script', 'class', 'flat_dim', 'exit']
    for o, (b, Y) in zip(obs, blocks):
        b = json.loads(json.dumps(b))     # a copy, in the key order of block() whatever the cache stored
        sp = {k: b[k] for k in order if k in b}; sp.update({k: v for k, v in b.items() if k not in sp})
        e = sp.get('exit')
        if e is not None and Y is not None:
            for p in obs:
                if p is not o and same_configuration(p['value'][0]['value'], p['X'], e['leads_to']['value'], Y):
                    e['leads_to']['same_as'] = p['name']; break
        o['stuck_point'] = sp


def main(argv):
    workers = None
    if '--workers' in argv:
        i = argv.index('--workers'); workers = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    for N in map(int, argv):
        f = os.path.join(EXPORT, f'n{N}', 'observations.json')
        doc = json.load(open(f)); obs = doc['observations']
        cache = os.path.join(EXPORT, f'n{N}', 'stuck.jsonl')
        res = classify_all([(cache, o['X']) for o in obs], workers)
        attach(obs, [res[(cache, x_key(o['X']))] for o in obs])
        compact_cache(cache, [x_key(o['X']) for o in obs])
        json.dump(doc, open(f, 'w'), indent=1)
        for o in obs:
            s = o['stuck_point']
            print(f"N={N} {o['name']:<22} {o['value'][0]['value']:.10f} {s['class']}"
                  + (f" -> {s['exit']['leads_to']}" if 'exit' in s else ''), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
