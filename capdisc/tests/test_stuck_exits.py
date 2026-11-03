"""Regression test for sphcap.stuck.follow_exits (numerical; not part of the proof).

The atlas point search_N16#2 is a stuck point with a genuine first-order exit whose linear regime ends below a
step of 1e-5. follow_exits must use it and descend (it reaches the N = 16 best value 0.1426321606).

    python -m pytest tests/test_stuck_exits.py -q
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sphcap import balance as B                     # noqa: E402
from sphcap.stuck import follow_exits                # noqa: E402


def test_short_exit_is_followed():
    doc = json.load(open(os.path.join(ROOT, 'export/sphere/d3/cap_discrepancy_sup/n16/observations.json')))
    o = next(o for o in doc['observations'] if o['name'] == 'search_N16#2')
    X = B.normalize(np.array(o['X'])); D = B.discrepancy(X)
    Y, D1 = follow_exits(X, D)
    assert D1 < D - 1e-5, f'exit not followed: {D} -> {D1}'
