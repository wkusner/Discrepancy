"""Second-kernel re-run of the refinement stage (jobs/refine.py), in mpmath interval arithmetic.

Independent of Arb/python-flint and of sphcap.certify: cap geometry, forward AD over intervals,
robust counts, the single-cap and LP-combined first-order bounds and the domain tests are
re-implemented here on mpmath.iv.  The same survivor cells of the interval-verified mesh stage
(capcert_rig) are bisected until every piece is
  * inside a box B of the local verification (certified), or
  * provably outside the fundamental domain, or
  * shown to satisfy D >= T by
        D(Y) >= sum_C lam_C a_C - sum_j |sum_C lam_C m_Cj| w_j      (lam >= 0, sum lam = 1),
    with a_C = phi_C(c) - sum_j r_Cj w_j, G_C in m_C +- r_C enclosing grad phi_C over the cell.
usage: python jobs/xcheck_refine.py N PREFIX [--workers 16] [--min-width 1e-7]
writes PREFIX.xrefine.json
"""
import argparse, glob, json, os, sys, time, itertools, re
from multiprocessing import Pool
import numpy as np
from mpmath import iv, mp
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__)); CERT = os.path.join(HERE, '..', 'certificates')
iv.dps = 17          # 56-bit intervals; enough here, and fast


# ---------------------------------------------------------------- forward AD over mpmath.iv
class V:
    __slots__ = ('v', 'd')
    def __init__(s, v, d=None): s.v = v; s.d = d or {}
    @staticmethod
    def c(x): return x if isinstance(x, V) else V(iv.mpf(x))
    def __add__(a, b):
        b = V.c(b); d = dict(a.d)
        for k, x in b.d.items(): d[k] = d[k] + x if k in d else x
        return V(a.v + b.v, d)
    __radd__ = __add__
    def __neg__(a): return V(-a.v, {k: -x for k, x in a.d.items()})
    def __sub__(a, b): return a + (-V.c(b))
    def __rsub__(a, b): return V.c(b) + (-a)
    def __mul__(a, b):
        b = V.c(b); d = {k: x * b.v for k, x in a.d.items()}
        for k, x in b.d.items(): d[k] = d[k] + a.v * x if k in d else a.v * x
        return V(a.v * b.v, d)
    __rmul__ = __mul__
    def __truediv__(a, b):
        b = V.c(b); q = a.v / b.v; d = {k: x / b.v for k, x in a.d.items()}
        for k, x in b.d.items():
            t = -q * x / b.v; d[k] = d[k] + t if k in d else t
        return V(q, d)
    def sqrt(a):
        if not (a.v.a > 0):                 # radicand not certainly positive: the cap is unusable here
            raise ZeroDivisionError
        s = iv.sqrt(a.v); return V(s, {k: x / (2 * s) for k, x in a.d.items()})


def sc(x):
    s, c = iv.sin(x.v), iv.cos(x.v)
    return V(s, {k: v * c for k, v in x.d.items()}), V(c, {k: -v * s for k, v in x.d.items()})


G = {}


def points(q, N):
    nq = 2 * N - 3
    Q = [V(q[k], {k: iv.mpf(1)}) for k in range(nq)]
    P = [[V.c(0), V.c(0), V.c(1)]]
    s0, c0 = sc(Q[0]); P.append([s0, V.c(0), c0])
    for i in range(2, N):
        st, ct = sc(Q[2 * i - 3]); sp, cp = sc(Q[2 * i - 2])
        P.append([st * cp, st * sp, ct])
    return P


dot = lambda a, b: a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
sub = lambda a, b: [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
cross = lambda a, b: [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def cap(P, I, side):
    if len(I) == 2:
        S = [P[I[0]][k] + P[I[1]][k] for k in range(3)]; SS = dot(S, S)
        return SS.sqrt() * 0.5, (lambda x: dot(S, x) * 2 - SS)
    a, b, c = (P[i] for i in I)
    n = cross(sub(b, a), sub(c, a)); na = dot(n, a)
    return na * side / dot(n, n).sqrt(), (lambda x: (dot(n, x) - na) * side)


def mv(center, grad_box, w):
    """mean-value enclosure over the box: value at the centre + sum_j G_j [-w_j, w_j]."""
    out = center
    for k, g in grad_box.items():
        out = out + g * iv.mpf([-w[k], w[k]])
    return out


def init(N, T, boxes, minw):
    caps = [(I, 1) for I in itertools.combinations(range(N), 2)]
    caps += [(I, s) for I in itertools.combinations(range(N), 3) for s in (1, -1)]
    G.update(N=N, T=iv.mpf(T), Tf=T, boxes=boxes, minw=minw, caps=caps)


def inside_box(c, w):
    for (m, r) in G['boxes']:
        if all(iv.mpf(abs(iv.mpf(ci) - iv.mpf(mi)).b) + iv.mpf(wi) <= iv.mpf(r) for ci, mi, wi in zip(c, m, w)):
            # exact-ish check: |c - m| + w <= r evaluated with outward rounding
            return True
    return False


def misses_domain(c, w, N):
    for i in range(2, N - 1):
        a, b = 2 * i - 3, 2 * (i + 1) - 3
        if (iv.mpf(c[a]) - iv.mpf(w[a])).a > (iv.mpf(c[b]) + iv.mpf(w[b])).b:
            return True
    box = [iv.mpf([iv.mpf(ci) - iv.mpf(wi), iv.mpf(ci) + iv.mpf(wi)]) for ci, wi in zip(c, w)]
    box = [iv.mpf([x.a, x.b]) for x in box]
    P = points(box, N)
    c01 = iv.cos(box[0])
    for i, j in itertools.combinations(range(N), 2):
        if (i, j) == (0, 1): continue
        g = dot(P[i], P[j]).v
        if g.a > c01.b:
            return True
    return False


def prescreen(c, w, N, keep=18, margin=0.06):
    """Selection only (floating point): the caps whose value at the centre is within `margin`
    of the largest, at most `keep` of them.  Any subset of caps gives a valid bound."""
    X = [[0.0, 0.0, 1.0], [np.sin(c[0]), 0.0, np.cos(c[0])]]
    for i in range(2, N):
        th, ph = c[2 * i - 3], c[2 * i - 2]
        X.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    X = np.array(X); vals = []
    for (I, s) in G['caps']:
        if len(I) == 2:
            S = X[I[0]] + X[I[1]]; nS = np.linalg.norm(S)
            if nS < 1e-9: continue
            wv = S / nS; tt = nS / 2
        else:
            nv = np.cross(X[I[1]] - X[I[0]], X[I[2]] - X[I[0]]); nn = np.linalg.norm(nv)
            if nn < 1e-12: continue
            wv = s * nv / nn; tt = wv @ X[I[0]]
        cnt = int(np.sum(X @ wv >= tt - 1e-9))
        vals.append((cnt / N - (1 - tt) / 2, (I, s)))
    vals.sort(key=lambda v: -v[0])
    top = vals[0][0] if vals else 0
    return [cs for v, cs in vals[:keep] if v >= top - margin - 2 * float(np.max(w))]


def cap_plain(P, I, side):
    """cap level t and margin on plain intervals (no derivatives)."""
    if len(I) == 2:
        S = [P[I[0]][k] + P[I[1]][k] for k in range(3)]; SS = dot(S, S)
        if not (SS.a > 0): raise ZeroDivisionError
        return iv.sqrt(SS) / 2, (lambda x: dot(S, x) * 2 - SS)
    a, b, c = (P[i] for i in I)
    n = cross(sub(b, a), sub(c, a)); na = dot(n, a); nn = dot(n, n)
    if not (nn.a > 0): raise ZeroDivisionError
    return na * side / iv.sqrt(nn), (lambda x: (dot(n, x) - na) * side)


def cell_bound(c, w):
    N = G['N']; nq = 2 * N - 3
    cb = [iv.mpf(x) for x in c]
    box = [iv.mpf([iv.mpf(ci) - iv.mpf(wi), iv.mpf(ci) + iv.mpf(wi)]) for ci, wi in zip(c, w)]
    box = [iv.mpf([x.a, x.b]) for x in box]
    Pb = points(box, N)                                          # with derivatives (gradients over the box)
    Pcv = [[x.v for x in p] for p in points(cb, N)]               # values at the centre
    Pbv = [[x.v for x in p] for p in Pb]                          # value enclosures over the box
    best = iv.mpf(1) / N
    rows = []
    for (I, s) in prescreen(c, w, N):
        try:
            tc, _ = cap_plain(Pcv, I, s)
            tb, _ = cap(Pb, I, s)
            _, mbox = cap_plain(Pbv, I, s)
        except ZeroDivisionError:
            continue
        if not all(abs(float(x)) < 1e300 for x in (tc.a, tc.b, tb.v.a, tb.v.b)):
            continue
        # robust count: points (other than the defining ones) inside the cap for every Y in the cell
        n = len(I) + sum(1 for j in range(N) if j not in I and mbox(Pbv[j]).a > 0)
        tc = V(tc)
        phic = iv.mpf(n) / N - (1 - tc.v) / 2
        grad = {k: g / 2 for k, g in tb.d.items()}             # d phi / dq = (dt/dq)/2
        mid = np.zeros(nq); rad = np.zeros(nq)
        for k, g in grad.items():
            mid[k] = float(g.mid); rad[k] = float(g.delta) / 2
        # a_C = phi(c) - sum_j r_j w_j with r_j >= |G_j - mid_j| (outward)
        rsum = iv.mpf(0)
        for k, g in grad.items():
            mk = iv.mpf(mid[k])
            rk = max(abs(iv.mpf(g.a) - mk).b, abs(iv.mpf(g.b) - mk).b)      # upper bound of |G_k - mid_k|
            rsum = rsum + iv.mpf(rk) * iv.mpf(w[k])
        a = phic - rsum
        one = a - sum((abs(iv.mpf(mid[k])) * iv.mpf(w[k]) for k in grad), iv.mpf(0))
        if one.a > best.a: best = one
        rows.append((a, mid, grad))
    if best.a >= G['T'].b:
        return best
    # LP-combined bound: lam from a float LP, rigorous evaluation in intervals
    A = np.array([np.append(r[1], -1.0) for r in rows]); av = np.array([float(r[0].mid) for r in rows])
    res = linprog(np.append(np.zeros(nq), 1.0), A_ub=A, b_ub=-av, bounds=[(-wk, wk) for wk in w] + [(None, None)], method='highs')
    if res.status != 0:
        return best
    lam = np.maximum(-res.ineqlin.marginals, 0.0)
    if lam.sum() <= 0: return best
    L = [iv.mpf(float(x)) for x in lam]; S = sum(L, iv.mpf(0))
    acc = iv.mpf(0)
    for (a, mid, grad), l in zip(rows, L):
        if float(l.b) == 0: continue
        acc = acc + l * a
    pen = iv.mpf(0)
    for j in range(nq):
        gj = iv.mpf(0)
        for (a, mid, grad), l in zip(rows, L):
            if float(l.b) == 0: continue
            gj = gj + l * iv.mpf(mid[j])
        pen = pen + iv.mpf(max(abs(gj.a), abs(gj.b))) * iv.mpf(w[j])
    lb = (acc - pen) / S
    if lb.a > best.a: best = lb
    return best


def split_dim(c, w, N):
    s = w.copy()
    for i in range(2, N):
        s[2 * i - 2] = w[2 * i - 2] * (abs(np.sin(c[2 * i - 3])) + 1e-3)
    return int(np.argmax(s))


def process(cell):
    c0, w0 = cell; N = G['N']
    stack = [(np.array(c0), np.array(w0))]
    st = dict(cells=0, bounded=0, boxed=0, sym=0, failed=0, fails=[])
    while stack:
        c, w = stack.pop(); st['cells'] += 1
        if inside_box(c, w): st['boxed'] += 1; continue
        if misses_domain(c, w, N): st['sym'] += 1; continue
        lb = cell_bound(c, w)
        if lb.a >= G['T'].b: st['bounded'] += 1; continue
        if w.max() < G['minw']:
            st['failed'] += 1
            if len(st['fails']) < 20: st['fails'].append(dict(c=c.tolist(), w=w.tolist(), lb=float(lb.a)))
            continue
        j = split_dim(c, w, N); h = w.copy(); h[j] = w[j] / 2
        a = c.copy(); a[j] = c[j] - h[j]; b = c.copy(); b[j] = c[j] + h[j]
        h[j] = np.nextafter(h[j] + (abs(c[j]) + w[j]) * 2.0 ** -52, np.inf)   # cover rounding of the new centres
        stack += [(a, h), (b, h)]
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int); ap.add_argument('prefix')
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--min-width', type=float, default=1e-7)
    ap.add_argument('--limit', type=int, default=0, help='test: only the first LIMIT input cells')
    a = ap.parse_args()
    N, nq = a.N, 2 * a.N - 3
    name = f'N{N}_best'
    lc = json.load(open(os.path.join(CERT, f'localcert_{name}.json')))
    xl = json.load(open(os.path.join(CERT, f'xcheck_local_{name}.json')))
    assert lc.get('verified') and xl.get('verified'), "local verification not certified"
    boxes = [(im['q_center'], lc['box_radius']) for im in lc['images']]
    T = max(im['D_enclosure'][1] for im in lc['images'])
    stamp = dict(kv.split('=', 1) for kv in open(a.prefix + '.done').read().split())
    # survivors of either mesh kernel: 'arb' (capcert_arb, pipeline B) or 'rigorous' (capcert_rig)
    assert int(stamp['N']) == N and stamp['mode'] in ('arb', 'rigorous') and float.fromhex(stamp['keep']) == 1.0
    assert float.fromhex(stamp['T']) >= T
    rec = (2 * nq + 1) * 8
    files = sorted(f for f in glob.glob(a.prefix + '.surv.*') if f.rsplit('.', 1)[1].isdigit())
    for f in files: assert os.path.getsize(f) % rec == 0
    arr = np.concatenate([np.fromfile(f, dtype=np.float64) for f in files]).reshape(-1, 2 * nq + 1)
    assert len(arr) >= int(stamp['survivors'])
    if a.limit: arr = arr[:a.limit]
    cells = [(row[:nq], row[nq:2 * nq]) for row in arr]
    print(f"N={N}: {len(cells):,} cells, T={T!r}, kernel mpmath.iv", flush=True)
    t0 = time.time(); tot = dict(cells=0, bounded=0, boxed=0, sym=0, failed=0); fails = []
    with Pool(a.workers, initializer=init, initargs=(N, T, boxes, a.min_width)) as pool:
        for k, st in enumerate(pool.imap_unordered(process, cells, chunksize=16)):
            for key in tot: tot[key] += st[key]
            fails += st['fails'][:5]
            if (k + 1) % 2000 == 0 or k + 1 == len(cells):
                print(f"[{time.time()-t0:7.0f}s] {k+1:,}/{len(cells):,}; {tot}", flush=True)
    res = dict(N=N, T=T, mesh_stamp=stamp, mesh_kernel=stamp['mode'], input_cells=len(cells), complete=(not a.limit), **tot, fails=fails[:50],
               time=time.time() - t0, verified=(tot['failed'] == 0 and not a.limit), kernel='mpmath.iv', label='interval-verified (mpmath.iv), second kernel for the refinement')
    json.dump(res, open(a.prefix + '.xrefine.json', 'w'), indent=1)
    print("VERIFIED (mpmath.iv): no cell left" if res['verified'] else f"failed {tot['failed']} (limit={a.limit})")


if __name__ == '__main__':
    main()
