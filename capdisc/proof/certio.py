"""proof.certio -- writing certificates (TRUSTED CORE, bookkeeping only; PROOF.md Section 6, items 4-5).

Every certificate written by the proof scripts carries a `stamp`: the claim (the Step of PROOF.md), the
commit, the library versions, the working precision, the command line (paths relative to the repository; other
absolute paths shortened to their last component, so no home directory or user name is recorded), the machine
type (no host names),
the date, the run time, and the SHA-256 of every input file. `tree_dirty` is true when the proof code (the paths
CODE_PATHS, relative to this directory, untracked files included) differs from the commit; `tree_checked` names
those paths, so that a reader can tell a stamp of the fixed check from one written before 2026-11-04, when the
pathspecs matched nothing and `tree_dirty` was always false (audit 2026-11-04, finding 1). Every git call fails
loudly: a stamp without a commit is never written. Certificates are written atomically
(temporary file, then rename), so an interrupted run leaves no partial certificate.

Exact values: `exact(x)` stores a double as a hex float (round-trips exactly in every language with C99
`strtod`); `ball(b)` stores an Arb ball as its exact midpoint and an upper bound for its radius, both as
hex floats when they are doubles, otherwise as Arb's own decimal string (which Arb reads back outward).
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# the proof code, as git pathspecs relative to REPO (= capdisc/, the directory given to `git -C`)
CODE_PATHS = ('proof', 'csrc', 'verify.sh', 'build.sh')


class GitError(RuntimeError):
    pass


def _git(*args, repo=None, strip=True):
    """run git in `repo` (default REPO); any failure, and any message on stderr, raises GitError"""
    cmd = ['git', '-C', repo or REPO, *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        raise GitError(f'cannot run git: {e}') from e
    if r.returncode != 0 or r.stderr.strip():
        raise GitError(f"git {' '.join(args)}: exit {r.returncode}: {r.stderr.strip()}")
    return r.stdout.strip() if strip else r.stdout


def tree_state(repo=None):
    """(commit, dirty, dirty_files): dirty if any file under CODE_PATHS is modified, staged, deleted or untracked"""
    commit = _git('rev-parse', 'HEAD', repo=repo)
    lines = _git('status', '--porcelain', '--untracked-files=all', '--', *CODE_PATHS, repo=repo, strip=False)
    files = [ln[3:] for ln in lines.splitlines() if ln.strip()]
    return commit, bool(files), files


def versions():
    out = dict(python=platform.python_version())
    for mod in ('flint', 'numpy', 'scipy', 'sympy', 'mpmath'):
        try:
            m = __import__(mod)
            out['python-flint' if mod == 'flint' else mod] = m.__version__
            if mod == 'flint':
                out['FLINT'] = getattr(m, '__FLINT_VERSION__', None)
        except Exception:
            out[mod] = None
    return out


def _public_arg(a):
    if not os.path.isabs(a): return a
    r = os.path.relpath(a, REPO)
    return r if not r.startswith('..') else '.../' + os.path.basename(a.rstrip('/'))


def _public_path(p):
    """an input file as recorded in a stamp: relative to REPO if inside it, else '.../<file name>' (no directory
    names outside the repository; the chain checker matches inputs by file name)"""
    r = os.path.relpath(os.path.abspath(p), REPO)
    return r if not r.startswith('..') else '.../' + os.path.basename(p)


def stamp(step, inputs=(), t0=None, prec=None):
    """step: e.g. 'PROOF.md Step 4 (local verification)'; inputs: files read."""
    commit, dirty, files = tree_state()
    ins = {}
    for p in inputs:
        k = _public_path(p)
        if k in ins: raise ValueError(f'two inputs recorded as {k}')
        ins[k] = sha256(p)
    out = dict(step=step, commit=commit, tree_dirty=dirty, tree_checked=list(CODE_PATHS), versions=versions(), prec=prec,
               argv=[_public_arg(a) for a in sys.argv], machine=f"{platform.system()} {platform.machine()}, {os.cpu_count()} cpus",
               date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               seconds=None if t0 is None else round(time.time() - t0, 1),
               inputs=ins)
    if dirty: out['tree_dirty_files'] = files
    return out


def exact(x):
    """a double, exactly: hex float string."""
    return float(x).hex()


def float_up(x):
    """the smallest double >= the exact Arb value x (x a point ball, e.g. b.upper())."""
    import math
    from flint import arb
    f = float(x)
    while arb(f) < x: f = math.nextafter(f, math.inf)
    return f


def float_down(x):
    """the largest double <= the exact Arb value x."""
    import math
    from flint import arb
    f = float(x)
    while arb(f) > x: f = math.nextafter(f, -math.inf)
    return f


def from_exact(s):
    return float.fromhex(s) if isinstance(s, str) else float(s)


def ball(b, digits=40):
    """an Arb ball as a string that Arb reads back to a ball containing b (outward)."""
    return b.str(digits, radius=True, more=True)


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
