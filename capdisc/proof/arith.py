"""proof.arith -- forward-mode automatic differentiation over Arb balls, points of a configuration in the
coordinates of the fundamental domain, and caps (TRUSTED CORE; see PROOF.md, Steps 2-5).

Coordinates q (2N-3):  x0 = (0,0,1),  x1 = (sin q0, 0, cos q0),
    x_i = (sin th cos ph, sin th sin ph, cos th),  th = q[2i-3], ph = q[2i-2]   (i >= 2).
A cap is given by its defining set I: |I| = 2 is the smallest cap through the pair; |I| = 3 is one of the
two caps bounded by the circle through the triple, side = +1 or -1.
"""
import itertools
import re
import flint
from flint import arb

ZERO = arb(0)
ONE = arb(1)


# ---------------------------------------------------------------- forward-mode AD over arb
class AD:
    __slots__ = ("v", "d")

    def __init__(self, v, d=None):
        self.v = v if isinstance(v, arb) else arb(v)
        self.d = d or {}

    def _lift(o):
        return o if isinstance(o, AD) else AD(o)

    def __add__(a, b):
        b = AD._lift(b); d = dict(a.d)
        for k, x in b.d.items(): d[k] = d[k] + x if k in d else x
        return AD(a.v + b.v, d)
    __radd__ = __add__

    def __neg__(a):
        return AD(-a.v, {k: -x for k, x in a.d.items()})

    def __sub__(a, b): return a + (-AD._lift(b))
    def __rsub__(a, b): return AD._lift(b) + (-a)

    def __mul__(a, b):
        b = AD._lift(b); d = {k: x * b.v for k, x in a.d.items()}
        for k, x in b.d.items(): d[k] = d[k] + a.v * x if k in d else a.v * x
        return AD(a.v * b.v, d)
    __rmul__ = __mul__

    def __truediv__(a, b):
        b = AD._lift(b); inv = 1 / b.v
        d = {k: x * inv for k, x in a.d.items()}
        q = a.v * inv
        for k, x in b.d.items():
            t = -q * x * inv
            d[k] = d[k] + t if k in d else t
        return AD(q, d)

    def sqrt(a):
        s = a.v.sqrt(); h = 1 / (2 * s)
        return AD(s, {k: x * h for k, x in a.d.items()})


def ad_sin_cos(x: AD):
    s, c = x.v.sin(), x.v.cos()
    return AD(s, {k: v * c for k, v in x.d.items()}), AD(c, {k: -v * s for k, v in x.d.items()})


def ball(lo, hi):
    return arb((lo + hi) / 2, (hi - lo) / 2)


def points_ad(qint, N):
    """qint: list of arb (point values or intervals).  Returns list of 3-lists of AD."""
    P = [[AD(0), AD(0), AD(1)]]
    q0 = AD(qint[0], {0: ONE})
    s0, c0 = ad_sin_cos(q0)
    P.append([s0, AD(0), c0])
    for i in range(2, N):
        th = AD(qint[2 * i - 3], {2 * i - 3: ONE}); ph = AD(qint[2 * i - 2], {2 * i - 2: ONE})
        st, ct = ad_sin_cos(th); sp, cp = ad_sin_cos(ph)
        P.append([st * cp, st * sp, ct])
    return P


def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


# ---------------------------------------------------------------- caps
class CapSpec:
    """A cap kind: (I, side).  |I|=2: smallest cap (side ignored); |I|=3: side = +1/-1."""
    __slots__ = ("I", "side")

    def __init__(self, I, side=1):
        self.I = tuple(I); self.side = side

    def __repr__(self): return f"Cap{self.I}{'+' if self.side > 0 else '-'}"


def cap_terms(P, cap):
    """Return (t, margin_fn) where t is the signed level (AD) and margin_fn(x) >= 0 iff x in cap."""
    I = cap.I
    if len(I) == 2:
        S = add(P[I[0]], P[I[1]]); SS = dot(S, S); nS = SS.sqrt()
        t = nS * 0.5
        return t, (lambda x: dot(S, x) * 2 - SS)
    a, b, c = (P[i] for i in I)
    n = cross(sub(b, a), sub(c, a)); nn = dot(n, n).sqrt(); tr = dot(n, a)
    s = cap.side
    t = tr * s / nn
    return t, (lambda x: (dot(n, x) - tr) * s)


def all_caps(N):
    caps = [CapSpec(I) for I in itertools.combinations(range(N), 2)]
    for I in itertools.combinations(range(N), 3):
        caps += [CapSpec(I, 1), CapSpec(I, -1)]
    return caps



def parse_cap(s):
    """inverse of repr(CapSpec): 'Cap(0, 1, 2)+' -> CapSpec((0, 1, 2), 1)."""
    m = re.match(r"Cap\(([\d, ]+)\)([+-])", s)
    return CapSpec(tuple(int(v) for v in m.group(1).split(',')), 1 if m.group(2) == '+' else -1)
