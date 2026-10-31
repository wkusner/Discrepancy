"""proof.certio -- writing certificates (TRUSTED CORE, bookkeeping only; PROOF.md Section 6, items 4-5).

Every certificate written by the proof scripts carries a `stamp`: the claim (the Step of PROOF.md), the
commit, the library versions, the working precision, the command line, the machine type (no host names),
the date, the run time, and the SHA-256 of every input file. Certificates are written atomically
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


def _git(*args):
    try:
        return subprocess.run(['git', '-C', REPO, *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


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


def stamp(step, inputs=(), t0=None, prec=None):
    """step: e.g. 'PROOF.md Step 4 (local verification)'; inputs: files read."""
    commit = _git('rev-parse', 'HEAD')
    dirty = _git('status', '--porcelain', '--', 'capdisc/proof', 'capdisc/csrc', 'capdisc/certificates')
    return dict(step=step, commit=commit, tree_dirty=bool(dirty), versions=versions(), prec=prec,
                argv=sys.argv, machine=f"{platform.system()} {platform.machine()}, {os.cpu_count()} cpus",
                date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                seconds=None if t0 is None else round(time.time() - t0, 1),
                inputs={os.path.relpath(p, REPO): sha256(p) for p in inputs})


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
