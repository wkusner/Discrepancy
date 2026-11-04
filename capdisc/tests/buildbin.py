"""Test helpers: the binaries that build.sh produces (the proof uses exactly those; audit 2026-11-04, finding 9),
built once per test session, and CAPCERT_EVAL evaluation of cells."""
import os, subprocess, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_built = {}


def binary(name='capcert_rig'):
    """path of ROOT/<name>, freshly built by build.sh (once per process)"""
    if 'ok' not in _built:
        r = subprocess.run(['bash', os.path.join(ROOT, 'build.sh')], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        _built['ok'] = True
    return os.path.join(ROOT, name)


def evaluate(binary, N, cells, prefix=()):
    """[(zeroth-order bound, outside-F flag, first-order bound)] for cells [(centre, half-widths)]"""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        for c, w in cells:
            np.concatenate([c, w, [0.0]]).astype(np.float64).tofile(f)
    try:
        out = subprocess.run([*prefix, binary, str(N), '1', '1', '/dev/null'], env=dict(os.environ, CAPCERT_EVAL=f.name),
                             check=True, capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    return [tuple(float(x) for x in line.split()) for line in out.stdout.split('\n') if line.strip()], out
