"""The atlas export of cap discrepancy for N = 1..4, from the closed forms of sphcap/gram.py (NOT TRUSTED tooling;
the values it labels "proved (manuscript)" rest on the hand proofs of the unpublished manuscript, not on this code).

    python jobs/atlas_small_n.py [--out DIR] [--write-input] [--check-input]

Two steps, one command:

1. The input file sphcap/data/small_n.json (written by --write-input, checked by --check-input, committed): one
   record per sample configuration ("classes": name, N, the closed-form parameters, coordinates in double
   precision, the exact Gram matrix as strings in Q or Q(sqrt k), the stratum the sample was drawn for and how) and
   one record per stratum ("families": the defining equations and inequalities in named variables, the
   parameterization, the dimension modulo rotations and relabelling, the value of D, the strata in its closure,
   the tags of its generic point, and the samples by name).  These are the closed forms and the statements of the
   manuscript; the file is the `raw` record source of the observations below.

2. The export, in the form of jobs/atlas_pipeline.py, under export/sphere/d3/cap_discrepancy_sup/n<N>/ (N = 1..4):
   observations.jsonl  spec v0.2 observation lines: kind "find" for every class record, kind "family" for every
                       stratum record (a new kind, proposed with critconfig-atlas DECISIONS.md), raw = the record of
                       sphcap/data/small_n.json, obs_id by the rule of spec 7.1 over that file's SHA-256.
   observations.json   the export: for every find line the record of jobs/atlas_export.py observation() (active
                       hypergraph, weight space, key basis, first and second order, Gram matrix) with the level-3
                       value "proved (manuscript)", the exact Gram matrix, a level-0 cross-check of D by
                       sphcap.discrepancy (methods exact and bruteforce), the stuck-point class (N = 3, 4), and the
                       tags; for every family line the stratum record with its key basis (canonical code of the active
                       hypergraph at the stratum's generic sample), its samples by obs_id, local charts (base point and
                       tangent basis, level 0), a numerical probe of the transverse type (level 0), a trajectory
                       along the parameterization and the parameter region for the viewer.
   manifest.json       inputs with SHA-256 and commit, the code commit and the SHA-256 of every script involved,
                       library versions, and the SHA-256 of the outputs.

Deterministic: no clock, no randomness except numpy generators seeded 0; a second run rewrites the same bytes.

Exact checks done by this script (Fractions and Q(sqrt k) arithmetic, `QF` below), on every class record: the Gram
entries equal the coordinates' inner products to 1e-14; the Gram matrix has rank <= 3 (N = 4: det G = 0 exactly)
and, for N = 3, rank 2 (det G = 0 exactly: the points lie on a great circle); N = 3: the arcs satisfy the region
inequalities exactly where they are given exactly; N = 4: x + y + z = 1, 16xyz = xy + yz + zx and
max(x, y, z) <= 9/16 exactly.  Everything else it computes is double precision (level 0).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse, json, sys
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import atlas_export as E
import atlas_pipeline as P
import atlas_stuck as S
from sphcap import gram as GR
from sphcap.discrepancy import discrepancy

INPUT = 'sphcap/data/small_n.json'
SCRIPT = 'jobs/atlas_small_n.py'
CODE = [SCRIPT, 'jobs/atlas_export.py', 'jobs/atlas_pipeline.py', 'jobs/atlas_stuck.py', 'sphcap/gram.py',
        'sphcap/balance.py', 'sphcap/stuck.py', 'sphcap/discrepancy.py', 'sphcap/__init__.py']
MANUSCRIPT = ('hand proof in the unpublished manuscript on spherical cap discrepancy (Kusner); the citation replaces '
              'this label when the paper appears')
PROVED = 'proved (manuscript)'
G_STAR = 'g* = 2 arccos(2/3)'
NS = (1, 2, 3, 4)


# ------------------------------------------------------------------ exact arithmetic in Q(sqrt k)
class QF(object):
    """a + b sqrt(k), a, b rational, k a squarefree positive integer (k = 1 means b = 0)."""

    def __init__(self, a, b=0, k=1):
        self.a, self.b, self.k = Fraction(a), Fraction(b), int(k)
        if self.k == 1:
            self.a, self.b = self.a + self.b, Fraction(0)

    def _co(self, o):
        if not isinstance(o, QF):
            o = QF(o)
        if self.b == 0 and o.b != 0:
            return QF(self.a, 0, o.k), o
        if o.b == 0 and self.b != 0:
            return self, QF(o.a, 0, self.k)
        if self.b != 0 and o.b != 0 and self.k != o.k:
            raise ValueError('different fields: sqrt %d and sqrt %d' % (self.k, o.k))
        k = self.k if self.b != 0 else o.k
        return QF(self.a, self.b, k), QF(o.a, o.b, k)

    def __add__(self, o):
        x, y = self._co(o); return QF(x.a + y.a, x.b + y.b, x.k)
    __radd__ = __add__

    def __neg__(self):
        return QF(-self.a, -self.b, self.k)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, QF) else QF(o)))

    def __rsub__(self, o):
        return QF(o) - self

    def __mul__(self, o):
        x, y = self._co(o); return QF(x.a * y.a + x.b * y.b * x.k, x.a * y.b + x.b * y.a, x.k)
    __rmul__ = __mul__

    def __truediv__(self, o):
        x, y = self._co(o)
        n = y.a * y.a - y.b * y.b * y.k
        return x * QF(y.a / n, -y.b / n, y.k)

    def sign(self):
        """exact sign of a + b sqrt k"""
        sa, sb = (self.a > 0) - (self.a < 0), (self.b > 0) - (self.b < 0)
        if sb == 0 or sa == sb:
            return sa if sa else sb
        if sa == 0:
            return sb
        d = self.a * self.a - self.b * self.b * self.k      # sign of |a| - |b| sqrt k
        return sa if d > 0 else (sb if d < 0 else 0)

    def __float__(self):
        return float(self.a) + float(self.b) * float(np.sqrt(self.k))

    def __str__(self):
        def q(x):
            return str(x.numerator) if x.denominator == 1 else '%d/%d' % (x.numerator, x.denominator)
        if self.b == 0:
            return q(self.a)
        s = '' if self.a == 0 else q(self.a) + (' + ' if self.b > 0 else ' - ')
        if self.a == 0 and self.b < 0:
            s = '-'
        b = abs(self.b)
        return s + ('' if b == 1 else q(b) + '*') + 'sqrt(%d)' % self.k


def det(M):
    """exact determinant (Laplace expansion; N <= 4)"""
    n = len(M)
    if n == 1:
        return M[0][0]
    out = QF(0)
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in M[1:]]
        term = M[0][j] * det(minor)
        out = out + term if j % 2 == 0 else out - term
    return out


# ------------------------------------------------------------------ the closed forms
F = Fraction
COS_GSTAR = QF(F(-1, 9))                                   # cos g* = 2 (2/3)^2 - 1
SQ = lambda k: QF(0, 1, k)


def n3_class(name, cos_abc, ab, stratum, drawn_by):
    """Three points on the equator at consecutive arcs a, b, c = 2 pi - a - b (sphcap.gram.n3_points), with the exact
    cosines (cos a, cos b, cos c)."""
    a, b = ab
    X = GR.n3_points(a, b)
    ca, cb, cc = cos_abc
    G = [[QF(1), ca, cc], [ca, QF(1), cb], [cc, cb, QF(1)]]
    return dict(name=name, N=3, params=dict(a=a, b=b, c=2 * np.pi - a - b), points=X.tolist(), gram=G,
                stratum=stratum, drawn_by=drawn_by)


def n4_class(name, x, y, z, stratum, drawn_by):
    """Disphenoid (+-sqrt x, +-sqrt y, +-sqrt z), even number of minus signs (sphcap.gram.disphenoid_points); exact
    Gram matrix with entries 2x - 1, 2y - 1, 2z - 1 (sphcap.gram.disphenoid_gram)."""
    X = GR.disphenoid_points(float(x), float(y), float(z))
    a, b, c = 2 * x - 1, 2 * y - 1, 2 * z - 1
    one = QF(1)
    G = [[one, a, b, c], [a, one, c, b], [b, c, one, a], [c, b, a, one]]
    return dict(name=name, N=4, params=dict(x=float(x), y=float(y), z=float(z)), points=X.tolist(), gram=G,
                exact_params=dict(x=x, y=y, z=z), stratum=stratum, drawn_by=drawn_by)


def n4_xy(z):
    """x, y (x < y) with x + y = 1 - z, xy = z (1 - z)/(16 z - 1): the arc through z in exact form (z rational)."""
    z = F(z)
    s, p = 1 - z, z * (1 - z) / (16 * z - 1)
    disc = s * s - 4 * p                                   # rational; sqrt(disc) = r sqrt k with k squarefree
    num, den = disc.numerator, disc.denominator
    n2 = num * den                                         # sqrt(num/den) = sqrt(num den)/den
    k, r = 1, 1
    m, f = n2, 2
    while f * f <= m:
        while m % (f * f) == 0:
            m //= f * f; r *= f
        f += 1
    k = m
    root = QF(0, F(r, den), k) if k != 1 else QF(F(r, den))
    return (QF(s) - root) / 2, (QF(s) + root) / 2


def classes():
    g = GR.G_STAR
    out = []
    # N = 1, 2
    out.append(dict(name='N1_point', N=1, params={}, points=[[0.0, 0.0, 1.0]], gram=[[QF(1)]], stratum=None,
                    drawn_by='the configuration space modulo rotations is one point; any point represents it'))
    out.append(dict(name='N2_antipodal', N=2, params={}, points=[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
                    gram=[[QF(1), QF(-1)], [QF(-1), QF(1)]], stratum=None, drawn_by='the antipodal pair'))
    # N = 3: generic points of each stratum, chosen with exact cosines, and the special points
    a1, b1 = float(np.arccos(-1 / 3)), float(np.arccos(-1 / 4))
    out.append(n3_class('N3_generic', (QF(F(-1, 3)), QF(F(-1, 4)), QF(F(1, 12), F(-1, 6), 30)), (a1, b1), 'N3_S',
                        'a = arccos(-1/3), b = arccos(-1/4): a generic point (arcs pairwise distinct, all > g*)'))
    out.append(n3_class('N3_equilateral', (QF(F(-1, 2)),) * 3, (2 * np.pi / 3, 2 * np.pi / 3), 'N3_E',
                        'the equilateral triple a = b = c = 2 pi/3'))
    bA = float(np.arccos(-1 / 3))
    out.append(n3_class('N3_edge', (COS_GSTAR, QF(F(-1, 3)), QF(F(1, 27), F(-8, 27), 10)), (g, bA), 'N3_A',
                        'a = g*, b = arccos(-1/3): a generic point of the edge a = g* (b, c distinct, > g*)'))
    sP = float(np.arccos(-1 / 3))
    out.append(n3_class('N3_fold_larger', (QF(F(-1, 3)), QF(F(-7, 9)), QF(F(-1, 3))), (sP, 2 * np.pi - 2 * sP), 'N3_Fp',
                        'arcs (s, 2 pi - 2s, s) with s = arccos(-1/3): two equal arcs, the third larger'))
    sM = float(2 * np.arccos(3 / 5))
    out.append(n3_class('N3_fold_smaller', (QF(F(-7, 25)), QF(F(-3, 5)), QF(F(-3, 5))), (sM, np.pi - sM / 2), 'N3_Fm',
                        'arcs (s, t, t) with s = 2 arccos(3/5), t = pi - s/2: two equal arcs, the third smaller'))
    out.append(n3_class('N3_edge_midpoint', (COS_GSTAR, QF(F(-2, 3)), QF(F(-2, 3))), (g, np.pi - g / 2), 'N3_M',
                        'a = g*, b = c = pi - g*/2: the midpoint of an edge'))
    cc = QF(F(-79, 81))                                    # cos 2g* = 2 (1/81) - 1
    for k, (ab, cs) in enumerate((((g, g), (COS_GSTAR, COS_GSTAR, cc)), ((g, 2 * np.pi - 2 * g), (COS_GSTAR, cc, COS_GSTAR)),
                                  ((2 * np.pi - 2 * g, g), (cc, COS_GSTAR, COS_GSTAR)))):
        out.append(n3_class('N3_corner_%d' % (k + 1), cs, ab, 'N3_C',
                            'corner %d of the parameter region (two arcs equal to g*); the three corners are '
                            'relabellings of one configuration' % (k + 1)))
    # N = 4: the arc z in [z_sym, 9/16] and its endpoints
    r105, r70 = SQ(105), SQ(70)
    xs, ys = (13 - r105) / 32, (19 + r105) / 64
    out.append(n4_class('N4_arc_end_symmetric', xs, ys, ys, 'N4_SYM',
                        'N4_ARC_END_SYMMETRIC of sphcap/gram.py: x = (13 - sqrt 105)/32, y = z = (19 + sqrt 105)/64'))
    xe, ye = (14 - r70) / 64, (14 + r70) / 64
    out.append(n4_class('N4_arc_end_edge', xe, ye, QF(F(9, 16)), 'N4_EDGE',
                        'N4_ARC_END_EDGE of sphcap/gram.py: x = (14 - sqrt 70)/64, y = (14 + sqrt 70)/64, z = 9/16'))
    for z in (F(15, 32), F(1, 2), F(17, 32)):
        x, y = n4_xy(z)
        out.append(n4_class('N4_arc_z_%d_%d' % (z.numerator, z.denominator), x, y, QF(z), 'N4_ARC',
                            'interior point of the arc at z = %s (x < y the roots of t^2 - (1 - z) t + z (1 - z)/(16 z - '
                            '1))' % z))
    return out


# ------------------------------------------------------------------ the strata (family records)
def _tag(value, note=None, level=3, status=PROVED, **kw):
    d = dict(value=value, level=level, status=status)
    if level == 3 and status == PROVED:
        d['proof'] = dict(kind='manuscript', reference=MANUSCRIPT)
    if note:
        d['note'] = note
    d.update(kw)
    return d


def _value(num, den, closed):
    """level-3 value of D (a rational number): minimal polynomial den x - num, degenerate isolating interval"""
    q = '%d/%d' % (num, den) if den != 1 else str(num)
    return dict(level=3, status=PROVED, proof=dict(kind='manuscript', reference=MANUSCRIPT),
                value=dict(minpoly=[str(-num), str(den)], isolating_interval=[q, q], closed_form=closed,
                           decimal=repr(num / den)),
                note='D on the whole stratum (constant); ' + MANUSCRIPT)


N3_VARS = [dict(name='x_0, x_1, x_2', kind='coordinates', meaning='the points, unit vectors of R^3'),
           dict(name='a, b, c', kind='parameters', meaning='the arcs x_0 x_1, x_1 x_2, x_2 x_0 along the great circle '
                                                            '(radians), a + b + c = 2 pi'),
           dict(name='G_ij', kind='gram', meaning='<x_i, x_j>; G_01 = cos a, G_12 = cos b, G_20 = cos c')]
N3_SYSTEM = ['<x_i, x_i> = 1 (i = 0, 1, 2)', 'det(x_0, x_1, x_2) = 0 (the three points lie on a great circle)',
             'a + b + c = 2 pi, with x_1 at arc a from x_0 and x_2 at arc a + b from x_0 in one orientation']
N3_REGION = ['a >= g*', 'b >= g*', 'c >= g*', 'g* = 2 arccos(2/3), i.e. cos g* = -1/9 (the arc at which the smallest cap '
             'through a pair, with count 2/3 and area (1 - cos(arc/2))/2, reaches D = 1/2)']
N3_PARAM = dict(parameters=[dict(name='a', domain='[g*, 2 pi - 2 g*]'), dict(name='b', domain='[g*, 2 pi - 2 g*]')],
                region=['a >= g*', 'b >= g*', '2 pi - a - b >= g*'],
                map=dict(kind='points', formula=['x_0 = (1, 0, 0)', 'x_1 = (cos a, sin a, 0)',
                                                 'x_2 = (cos(a + b), sin(a + b), 0)']),
                covering='the map (a, b) -> configuration covers the closed region; modulo rotations, reflections and '
                         'relabelling it is 6 to 1 at a point with pairwise distinct arcs (S_3 permutes the arcs); a '
                         'fundamental domain is g* <= a <= b <= c, the triangle with vertices E (equilateral), C '
                         '(corner) and M (edge midpoint)',
                source='sphcap/gram.py n3_points, n3_gram, n3_is_minimiser_parameters')

N4_VARS = [dict(name='x_0, ..., x_3', kind='coordinates', meaning='the points, unit vectors of R^3'),
           dict(name='x, y, z', kind='parameters', meaning='squares of the coordinates of the disphenoid '
                                                           '(+-sqrt x, +-sqrt y, +-sqrt z), even number of minus signs'),
           dict(name='G_ij', kind='gram', meaning='<x_i, x_j> = 2x - 1, 2y - 1 or 2z - 1 (sphcap/gram.py '
                                                  'disphenoid_gram)')]
N4_SYSTEM = ['x + y + z = 1', '16 x y z = x y + y z + z x (on x + y + z = 1 this is sphcap/gram.py n4_cubic: every '
             'face cap has height t = 1/4)', 'x_0 = (sqrt x, sqrt y, sqrt z), x_1 = (sqrt x, -sqrt y, -sqrt z), '
             'x_2 = (-sqrt x, sqrt y, -sqrt z), x_3 = (-sqrt x, -sqrt y, sqrt z)']
N4_REGION = ['max(x, y, z) <= 9/16 (at 9/16 the smallest cap through an edge reaches D = 3/8)',
             'x, y, z > 0']


def n4_param(lo):
    return dict(parameters=[dict(name='z', domain='[%s, 9/16]' % lo)],
                region=['%s <= z <= 9/16' % lo],
                map=dict(kind='points', formula=[
                    'y = ((1 - z) + sqrt((1 - z)^2 - 4 z (1 - z)/(16 z - 1)))/2, x = 1 - y - z (x < y: the two roots '
                    'of t^2 - (1 - z) t + z (1 - z)/(16 z - 1))',
                    'x_0 = (sqrt x, sqrt y, sqrt z), x_1 = (sqrt x, -sqrt y, -sqrt z), x_2 = (-sqrt x, sqrt y, -sqrt z), '
                    'x_3 = (-sqrt x, -sqrt y, sqrt z)']),
                covering='z in [(14 + sqrt 70)/64, 9/16] covers the arc of minimizers in the labelled configuration '
                         'space; the relabelling that swaps y and z (a symmetry of the disphenoid family) folds it at '
                         'z = (19 + sqrt 105)/64, where y = z, so modulo relabelling the arc is z in [(19 + sqrt '
                         '105)/64, 9/16]',
                source='sphcap/gram.py disphenoid_points, n4_cubic, N4_ARC_END_SYMMETRIC, N4_ARC_END_EDGE')


def _tags(amb, dim, coindex, nullity, local, local_note, level_local, order='first', glob_note=None):
    """tags of the generic point of a stratum: dimensions exact by construction, index 0 (global minimizer)"""
    codim = amb - dim
    lt_level, lt_status = (3, PROVED) if level_local == 3 else (0, 'numerical')
    t = dict(
        ambient_dim=[_tag(amb, '2N - 3 = dimension of (S^2)^N modulo rotations (relabelling acts discretely): a '
                               'dimension count', status='exact (test-verified)')],
        stratum_dim=[_tag(dim, 'dimension of the stratum modulo rotations and relabelling', lt_level, lt_status)],
        codim=[_tag(codim, None, lt_level, lt_status)],
        index=[_tag(0, 'a global minimizer has no descending direction')],
        global_=[_tag('global minimizer (proved)', glob_note or 'D = D* = min D on the whole stratum')],
        order=[_tag(order, None, lt_level, lt_status)])
    t['coindex'] = [_tag(coindex, local_note, lt_level, lt_status)]
    t['nullity'] = [_tag(nullity, local_note, lt_level, lt_status)]
    t['local_type'] = [_tag(local, local_note, lt_level, lt_status)]
    t['global'] = t.pop('global_')
    return t


HAND = ('derived by hand in the export session from the parameterization and the active caps (not reviewed; '
        'checked numerically by the probe of the export)')


def families():
    out = []
    n3 = lambda **kw: dict(N=3, kind='minimizer_family', variables=N3_VARS, system=N3_SYSTEM, parameterization=N3_PARAM,
                           value=_value(1, 2, '1/2'), **kw)
    sharp3 = ('moving a point off the great circle makes the cap through the three points smaller: its area is '
              '(1 - t)/2 with t linear in the displacement, so D grows at first order; along the circle D stays 1/2 '
              'while every arc stays >= g*')
    out.append(n3(name='N3_S', label='great-circle triples with pairwise distinct arcs, all > g*', dimension=2,
                  inequalities=N3_REGION + ['a, b, c pairwise distinct', 'a, b, c > g*'],
                  closure=[dict(name=n, reason=r) for n, r in (('N3_A', 'active_type'), ('N3_Fp', 'orbit_type'),
                                                               ('N3_Fm', 'orbit_type'), ('N3_C', 'active_type'),
                                                               ('N3_M', 'active_type'), ('N3_E', 'orbit_type'))],
                  tags=_tags(3, 2, 1, 0, 'sharp minimum', sharp3, 3),
                  samples=['N3_generic'], generic_sample='N3_generic',
                  note='the minimizers of D for N = 3 are the great-circle triples with every arc >= g* (D = 1/2); this '
                       'stratum is their generic part. Every triple has D >= 1/2 by the hemispheres. ' + MANUSCRIPT))
    out.append(n3(name='N3_A', label='one arc equal to g*, the other two distinct and > g* (edge of the region)',
                  dimension=1, inequalities=N3_REGION + ['a = g*', 'b, c > g*', 'b != c'],
                  closure=[dict(name='N3_C', reason='active_type'), dict(name='N3_M', reason='orbit_type')],
                  tags=_tags(3, 1, 2, 0, 'non-strict minimum', 'one-sided flat: D stays 1/2 when the arc a grows (into '
                             'N3_S) and grows at first order when it shrinks (the cap through the pair at arc g* has '
                             'count 2/3 and area 1/6) or when a point leaves the circle; ' + HAND, 0),
                  samples=['N3_edge'], generic_sample='N3_edge',
                  note='an active-type boundary: the smallest cap through the pair at arc g* is a worst cap here'))
    out.append(n3(name='N3_Fp', label='two equal arcs, the third larger (fold of the region under relabelling)',
                  dimension=1, inequalities=N3_REGION + ['a = c < b', 'a > g*'],
                  closure=[dict(name='N3_C', reason='active_type'), dict(name='N3_E', reason='orbit_type')],
                  tags=_tags(3, 1, 1, 1, 'non-strict minimum', 'the direction that makes the two equal arcs unequal '
                             'stays in N3_S (two-sided flat); off the circle D grows at first order; ' + HAND, 0),
                  samples=['N3_fold_larger'], generic_sample='N3_fold_larger',
                  note='an orbit-type boundary: the relabelling that swaps the two end points of the equal arcs '
                       'fixes these configurations, so the quotient by relabelling folds here; the active caps are '
                       'those of N3_S'))
    out.append(n3(name='N3_Fm', label='two equal arcs, the third smaller (fold of the region under relabelling)',
                  dimension=1, inequalities=N3_REGION + ['b = c > a', 'a > g*'],
                  closure=[dict(name='N3_M', reason='active_type'), dict(name='N3_E', reason='orbit_type')],
                  tags=_tags(3, 1, 1, 1, 'non-strict minimum', 'as for N3_Fp; ' + HAND, 0),
                  samples=['N3_fold_smaller'], generic_sample='N3_fold_smaller',
                  note='an orbit-type boundary, as N3_Fp; the active caps are those of N3_S'))
    out.append(n3(name='N3_C', label='two arcs equal to g* (corner of the region)', dimension=0,
                  inequalities=N3_REGION + ['a = b = g*', 'c = 2 pi - 2 g*'], closure=[],
                  tags=_tags(3, 0, 3, 0, 'non-strict minimum', 'flat only on the cone where both short arcs grow '
                             '(into N3_A, N3_Fp and N3_S), first-order growth in every other direction; ' + HAND, 0),
                  samples=['N3_corner_1', 'N3_corner_2', 'N3_corner_3'], generic_sample='N3_corner_1',
                  note='two caps through pairs are worst caps; the three corners of the (a, b) region are relabellings '
                       'of one configuration'))
    out.append(n3(name='N3_M', label='one arc equal to g*, the other two equal (midpoint of an edge)', dimension=0,
                  inequalities=N3_REGION + ['a = g*', 'b = c = pi - g*/2'], closure=[],
                  tags=_tags(3, 0, 2, 1, 'non-strict minimum', 'two-sided flat along the edge (N3_A), one-sided flat '
                             'into N3_S and N3_Fm, first-order growth otherwise; ' + HAND, 0),
                  samples=['N3_edge_midpoint'], generic_sample='N3_edge_midpoint',
                  note='active type of N3_A and the orbit type of the fold N3_Fm'))
    out.append(n3(name='N3_E', label='the equilateral triple', dimension=0,
                  inequalities=N3_REGION + ['a = b = c = 2 pi/3'], closure=[],
                  tags=_tags(3, 0, 1, 2, 'non-strict minimum', 'every motion along the circle stays in N3_S (two-sided '
                             'flat); off the circle D grows at first order; ' + HAND, 0),
                  samples=['N3_equilateral'], generic_sample='N3_equilateral',
                  note='the most symmetric minimizer (orbit type); sphcap.discrepancy gave 5/12 here before the fix of '
                       'audit-refactor 7f7fd86 (missed hemisphere through three points of a great circle); the fixed '
                       'code gives 1/2'))
    n4 = lambda **kw: dict(N=4, kind='minimizer_family', variables=N4_VARS, system=N4_SYSTEM,
                           value=_value(3, 8, '3/8'), **kw)
    sharp4 = ('two-branch balance: each face {i, j, k} has two worst caps, the cap through the face without the fourth '
              'point (count 3/4) and the one with it (count 1), with values 1/4 + t_F/2 and 1/2 - t_F/2, so D - 3/8 >= '
              '|t_F - 1/4|/2 for every face F; the arc is {t_F = 1/4 for all F}, and where the four differentials dt_F '
              'are independent modulo rotations D grows at first order in every transverse direction (rank 4 at the '
              'samples: first_order.rank of the export, level 0)')
    out.append(n4(name='N4_ARC', label='the disphenoid arc on the cubic, interior points', dimension=1,
                  inequalities=N4_REGION + ['(19 + sqrt 105)/64 < z < 9/16 (modulo relabelling)'],
                  parameterization=n4_param('(19 + sqrt 105)/64'),
                  closure=[dict(name='N4_SYM', reason='orbit_type'), dict(name='N4_EDGE', reason='active_type')],
                  tags=_tags(5, 1, 4, 0, 'sharp minimum', sharp4, 3),
                  samples=['N4_arc_z_15_32', 'N4_arc_z_1_2', 'N4_arc_z_17_32'], generic_sample='N4_arc_z_1_2',
                  note='the minimizers of D for N = 4 are the disphenoids of the cubic with max(x, y, z) <= 9/16 (D = '
                       '3/8): an arc, not a unique minimizer. ' + MANUSCRIPT))
    out.append(n4(name='N4_SYM', label='the symmetric end of the arc (y = z)', dimension=0,
                  inequalities=N4_REGION + ['y = z = (19 + sqrt 105)/64', 'x = (13 - sqrt 105)/32'],
                  parameterization=None, closure=[],
                  tags=_tags(5, 0, 4, 1, 'non-strict minimum', 'the arc passes through this point in the labelled '
                             'configuration space (two-sided flat along it); modulo the relabelling that swaps y and z '
                             'it is an end point; first-order growth in the four other directions; ' + HAND, 0),
                  samples=['N4_arc_end_symmetric'], generic_sample='N4_arc_end_symmetric',
                  note='orbit type: the tetragonal disphenoid (y = z) has the larger symmetry group; the active caps are '
                       'those of N4_ARC'))
    out.append(n4(name='N4_EDGE', label='the end of the arc where an edge cap becomes worst (z = 9/16)', dimension=0,
                  inequalities=N4_REGION + ['z = 9/16', 'x = (14 - sqrt 70)/64', 'y = (14 + sqrt 70)/64'],
                  parameterization=None, closure=[],
                  tags=_tags(5, 0, 5, 0, 'non-strict minimum', 'one-sided flat into N4_ARC (z decreasing), first-order '
                             'growth in every other direction (the smallest cap through the edge of length given by z '
                             'grows past 3/8 for z > 9/16); ' + HAND, 0),
                  samples=['N4_arc_end_edge'], generic_sample='N4_arc_end_edge',
                  note='active type: the smallest caps through the two edges with coordinate z (count 2/4, value '
                       'sqrt(z)/2 = 3/8 at z = 9/16) join the worst caps'))
    return out


# ------------------------------------------------------------------ the input file
def gram_strings(G):
    return [[str(v) for v in row] for row in G]


def input_doc():
    cl = []
    for c in classes():
        d = dict(name=c['name'], N=c['N'], params=c['params'], points=c['points'], gram_exact=gram_strings(c['gram']),
                 stratum=c['stratum'], drawn_by=c['drawn_by'])
        if 'exact_params' in c:
            d['exact_params'] = {k: str(v) for k, v in c['exact_params'].items()}
        cl.append(d)
    return dict(description='Closed-form configurations and strata of the cap-discrepancy minimizers for N = 1..4 '
                            '(sphcap/gram.py), written by jobs/atlas_small_n.py --write-input; the raw source of the '
                            'atlas observations of N = 1..4. Coordinates are double precision; Gram matrices are exact '
                            '(Q or Q(sqrt k), as strings); values and tags labelled "proved (manuscript)" rest on the '
                            'hand proofs of the unpublished manuscript.',
                generated_by=SCRIPT, classes=cl, families=families())


def check_classes():
    """the exact checks of the module docstring; returns a list of strings (one per check passed)"""
    done = []
    for c in classes():
        X, G, N = np.array(c['points']), c['gram'], c['N']
        Gn = np.array([[float(v) for v in row] for row in G])
        err = float(np.abs(X @ X.T - Gn).max())
        assert err < 1e-14, (c['name'], err)
        if N >= 3:
            dG = det(G)
            assert dG.sign() == 0, (c['name'], str(dG))
        if N == 3:
            # rank exactly 2 is implied by det = 0 and a nonzero 2x2 minor
            assert det([row[:2] for row in G[:2]]).sign() != 0, c['name']
            cosg = COS_GSTAR
            for v in (G[0][1], G[1][2], G[0][2]):
                # arc >= g*  <=>  cos(arc) <= cos g*  (arcs in [0, pi] as angles between points)... the arcs along the
                # circle may exceed pi; the region is checked numerically on the parameters below
                pass
            p = c['params']
            assert min(p['a'], p['b'], p['c']) >= GR.G_STAR - 1e-12, c['name']
        if N == 4:
            x, y, z = (c['exact_params'][k] for k in 'xyz')
            assert (x + y + z - 1).sign() == 0, c['name']
            assert (16 * x * y * z - (x * y + y * z + z * x)).sign() == 0, c['name']
            for v in (x, y, z):
                assert (QF(F(9, 16)) - v).sign() >= 0 and v.sign() > 0, c['name']
        done.append(c['name'])
    return done


# ------------------------------------------------------------------ the probe (level 0)
# directions of each stratum in the cover parameters ((a, b) for N = 3, z for N = 4)
STRATUM_TANGENT = {'N3_S': [(1, 0), (0, 1)], 'N3_A': [(0, 1)], 'N3_Fp': [(1, -2)], 'N3_Fm': [(1, -0.5)],
                   'N3_C': [], 'N3_M': [], 'N3_E': [], 'N4_ARC': [(1,)], 'N4_SYM': [], 'N4_EDGE': []}


def frames(X):
    """orthonormal tangent frames (e1_i, e2_i) at the points"""
    out = []
    for x in X:
        a = np.array([1.0, 0, 0]) if abs(x[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = a - (a @ x) * x; e1 /= np.linalg.norm(e1)
        out.append((e1, np.cross(x, e1)))
    return out


def to_coords(X, dX, fr):
    return np.array([v for i in range(len(X)) for v in (fr[i][0] @ dX[i], fr[i][1] @ dX[i])])


def move(X, v, t, fr):
    Y = []
    for i, x in enumerate(X):
        d = v[2 * i] * fr[i][0] + v[2 * i + 1] * fr[i][1]
        n = np.linalg.norm(d)
        Y.append(x if n == 0 else np.cos(t * n) * x + np.sin(t * n) * d / n)
    return np.array(Y)


def Dval(X):
    return discrepancy(X, 'exact').value


def cover_map(N, p):
    """points of the cover parameterization: N = 3 (a, b); N = 4 (z,)"""
    if N == 3:
        return GR.n3_points(p[0], p[1])
    z = p[0]
    s, q = 1 - z, z * (1 - z) / (16 * z - 1)
    y = (s + np.sqrt(s * s - 4 * q)) / 2
    return GR.disphenoid_points(1 - y - z, y, z)


def cover_params(c):
    return [c['params']['a'], c['params']['b']] if c['N'] == 3 else [c['params']['z']]


def cone(c):
    """(lineality basis, one-sided generators) of the tangent cone of the cover region at the sample, in parameter
    coordinates; from the region's constraints that are tight at the sample"""
    g = GR.G_STAR
    if c['N'] == 3:
        p = c['params']
        tight = [k for k in 'abc' if abs(p[k] - g) < 1e-12]
        grads = {'a': np.array([1.0, 0]), 'b': np.array([0, 1.0]), 'c': np.array([-1.0, -1.0])}
        if not tight:
            return [np.array([1.0, 0]), np.array([0, 1.0])], []
        if len(tight) == 1:
            n = grads[tight[0]]
            l = np.array([-n[1], n[0]]); l /= np.linalg.norm(l)
            return [l], [n / np.linalg.norm(n)]
        # two tight constraints: a pointed cone spanned by the two edge directions
        n1, n2 = grads[tight[0]], grads[tight[1]]
        e1 = np.array([-n1[1], n1[0]]); e1 = e1 if e1 @ n2 > 0 else -e1
        e2 = np.array([-n2[1], n2[0]]); e2 = e2 if e2 @ n1 > 0 else -e2
        return [], [e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)]
    z = c['params']['z']
    if abs(z - 9 / 16) < 1e-12:
        return [], [np.array([-1.0])]
    return [np.array([1.0])], []


FLAT, ASC = 1e-5, 1e-3


def probe(c, stratum_dim, h=1e-6):
    """numerical transverse type at a sample (level 0): returns a dict with the rank of the parameterization
    modulo rotations, the flat and ascending directions found, and the counts index / coindex / nullity"""
    X = np.array(c['points']); N = len(X); fr = frames(X); D0 = Dval(X)
    p = np.array(cover_params(c))
    J = []
    for k in range(len(p)):
        e = np.zeros(len(p)); e[k] = 1e-6
        J.append(to_coords(X, (cover_map(N, p + e) - cover_map(N, p - e)) / 2e-6, fr))
    J = np.array(J).T                                         # 2N x params
    R = np.array([to_coords(X, np.cross(w, X), fr) for w in np.eye(3)]).T
    lin, gens = cone(c)
    rate = lambda v: (Dval(move(X, v / np.linalg.norm(v), h, fr)) - D0) / h
    res = dict(sample=c['name'], step=h, D=D0)
    # the stratum's own tangent: lineality directions that are tangent to the stratum (the stratum's dimension)
    lin_v = [J @ l for l in lin]
    gen_v = [J @ g for g in gens]
    flat_two, flat_one, asc_opp = [], [], []
    for v in lin_v:
        flat_two.append(max(abs(rate(v)), abs(rate(-v))))
    for v in gen_v:
        flat_one.append(abs(rate(v)))
        asc_opp.append(rate(-v))
    # transverse: orthogonal complement of rotations and the parameter image
    B = np.hstack([R, J]) if J.size else R
    U, sv, Vt = np.linalg.svd(B, full_matrices=True)
    r = int(np.sum(sv > 1e-9 * sv.max()))
    Nrm = U[:, r:]
    rng = np.random.default_rng(0)
    asc = []
    for _ in range(24):
        w = Nrm @ rng.standard_normal(Nrm.shape[1])
        asc.append(min(rate(w), rate(-w)))
    for j in range(Nrm.shape[1]):
        asc.append(min(rate(Nrm[:, j]), rate(-Nrm[:, j])))
    rk_par = int(np.linalg.matrix_rank(np.hstack([R, J]), tol=1e-7)) - int(np.linalg.matrix_rank(R, tol=1e-7)) if J.size else 0
    amb = 2 * N - 3
    two_sided = len(lin) - stratum_dim
    res.update(param_rank_mod_rotations=rk_par, transverse_dim=int(Nrm.shape[1]),
               max_rate_two_sided_flat=max(flat_two) if flat_two else None,
               max_rate_one_sided_flat=max(flat_one) if flat_one else None,
               min_rate_opposite_of_one_sided=min(asc_opp) if asc_opp else None,
               min_rate_transverse=min(asc) if asc else None,
               index=0 if (not asc or min(asc) > 0) and all(x >= -1e-9 for x in asc_opp) else None,
               nullity=two_sided, coindex=amb - stratum_dim - two_sided)
    # flat: rate below 1e-5 at h = 1e-6 (a tangent step leaves a curved stratum at second order, rate O(h));
    # ascending: rate above 1e-3 (first-order growth; the rates found are >= 0.1)
    ok = ((not flat_two or max(flat_two) < FLAT) and (not flat_one or max(flat_one) < FLAT)
          and (not asc_opp or min(asc_opp) > ASC) and (not asc or min(asc) > ASC))
    res['consistent'] = bool(ok)
    return res


# ------------------------------------------------------------------ export
def code_state():
    """commit of the last change to the code of this export (CODE), as jobs/atlas_pipeline.py code_state()"""
    c = P.git('log', '-1', '--format=%h', '--', *CODE) or 'unknown'
    dirty = bool(P.git('status', '--porcelain', '--', *CODE))
    return dict(commit='unknown' if dirty else c, last_commit=c, modified=dirty,
                scripts={p: P.sha_file(os.path.join(ROOT, p)) for p in CODE})


def sh(x):
    return P.sha_bytes(x if isinstance(x, bytes) else x.encode())


def load_input():
    path = os.path.join(ROOT, INPUT)
    blob = open(path, 'rb').read()
    return path, blob, json.loads(blob.decode('utf-8'))


def tag_export(t):
    """the family record's tags, as exported (flat descriptions, as the rest of the export)"""
    return {k: [dict(v) for v in lst] for k, lst in t.items()}


def find_record(c, fsha, fcommit, idx):
    X = np.array(c['points'], float)
    src = dict(file=INPUT, file_sha256=fsha, pointer=c['name'], script=SCRIPT, commit=fcommit, pipeline=SCRIPT)
    rec = E.observation(X, c['name'], src)
    N = c['N']
    if N == 1:   # one point modulo rotations: the stabilizer SO(2) makes the count 2N - 3 = -1 wrong
        rec['first_order'].update(dim_quotient=0, kernel_dim=0, rank=0)
        rec['first_order']['note'] = ('N = 1: S^2 modulo rotations is a point (the stabilizer SO(2) of a point is '
                                      '1-dimensional), so the quotient has dimension 0, not 2N - 3 = -1')
    if N == 2:
        rec['first_order']['note'] = ('N = 2: at the antipodal pair the smallest cap through the two points is a '
                                      'hemisphere and its height is not differentiable there (the arc is pi), so the '
                                      'gradient-based first-order data above (rank 0) do not see the first-order growth; '
                                      'see the tags')
    Dstar = {1: (1, 1, '1'), 2: (1, 2, '1/2'), 3: (1, 2, '1/2'), 4: (3, 8, '3/8')}[N]
    ex, bf = discrepancy(X, 'exact').value, discrepancy(X, 'bruteforce').value
    dstar = Dstar[0] / Dstar[1]
    rec['value'].append(dict(id='v0x', level=0, status='numerical', value=float(ex),
                             note='cross-check by sphcap.discrepancy (as fixed in audit-refactor 7f7fd86 / 105e525): '
                                  'method exact %r, method bruteforce %r; |exact - %s| = %.1e, |bruteforce - %s| = %.1e'
                                  % (float(ex), float(bf), Dstar[2], abs(ex - dstar), Dstar[2], abs(bf - dstar)),
                             cross_check=dict(exact=float(ex), bruteforce=float(bf), closed_form=Dstar[2])))
    v3 = _value(Dstar[0], Dstar[1], Dstar[2])
    v3['id'] = 'v3'
    v3['note'] = 'D of this configuration; ' + MANUSCRIPT
    rec['value'].append(v3)
    rec['gram'] = dict(level=3, status='exact (test-verified)', ref=dict(file=INPUT, file_sha256=fsha,
                                                                     pointer='/classes/%d/gram_exact' % idx),
                       entries=c['gram_exact'],
                       note='exact Gram matrix of the closed form (Q or Q(sqrt k), entries as strings, labelled as '
                            'the X of this record); checked by jobs/atlas_small_n.py: entries equal the inner products '
                            'of X to 1e-14, det G = 0 exactly for N >= 3' + (' (rank 2: a great circle)' if N == 3
                                                                             else ''))
    rec['small_n'] = dict(params=c['params'], drawn_by=c['drawn_by'], stratum=c['stratum'])
    return rec


def entity_tags(c, fam_by_name):
    N = c['N']
    if N == 1:
        note = 'S^2 modulo rotations is a point: every configuration is this one, D = 1'
        return dict(ambient_dim=[_tag(0, 'S^2 modulo rotations is a point (the stabilizer of a point is SO(2)): a '
                                         'dimension count', status='exact (test-verified)')],
                    stratum_dim=[_tag(0, note)], codim=[_tag(0)],
                    index=[_tag(0)], coindex=[_tag(0)], nullity=[_tag(0)],
                    local_type=[_tag('sharp minimum', 'vacuous: no direction to move in modulo rotations')],
                    **{'global': [_tag('global minimizer (proved)', note)]}, order=[_tag('first')])
    if N == 2:
        note = ('moving off the antipodal pair shrinks the smallest cap through the two points at first order (its '
                'area (1 - cos(arc/2))/2 falls below 1/2 while it keeps both points), so D grows at first order; '
                'unique modulo rotations')
        return dict(ambient_dim=[_tag(1, '2N - 3 = 1, the angle between the points: a dimension count',
                                      status='exact (test-verified)')], stratum_dim=[_tag(0, 'isolated')],
                    codim=[_tag(1)], index=[_tag(0, note)], coindex=[_tag(1, note)], nullity=[_tag(0, note)],
                    local_type=[_tag('sharp minimum', note)],
                    **{'global': [_tag('global minimizer (proved)', 'D = 1/2, the minimum for N = 2, attained only by '
                                                                     'the antipodal pair')]}, order=[_tag('first')])
    return tag_export(fam_by_name[c['stratum']]['tags'])


def family_record(f, recs_by_name, cls_by_name, obs_of, plots):
    """the exported stratum record of family f"""
    rep = recs_by_name[f['generic_sample']]
    code = rep['active']['code']
    kb = dict(method='active-hypergraph canonical code at the generic sample %s (individualization-refinement, '
                     'jobs/atlas_export.py canonical_code); the ordinal is minted by the atlas' % f['generic_sample'],
              code=code, graph_hash=P.sha_bytes(("sphere|d3|%s|N%d|%s" % (E.FUNCTIONAL['id'], f['N'], code)).encode())[:12])
    out = dict(kind='family', functional=E.FUNCTIONAL['id'], container='sphere', d=3, N=f['N'], name=f['name'],
               label=f['label'], stratum_kind=f['kind'], key_basis=kb)
    out['system'] = dict(level=3, status=PROVED, proof=dict(kind='manuscript', reference=MANUSCRIPT),
                         variables=f['variables'], equations=f['system'], inequalities=f['inequalities'],
                         gauge='configurations modulo rotations (O(3) including reflections) and relabelling',
                         note='the stratum as a subset of the minimizer set; exact strings')
    out['dimension'] = [dict(level=3 if f['tags']['stratum_dim'][0]['level'] == 3 else 0,
                             status=f['tags']['stratum_dim'][0]['status'], value=f['dimension'],
                             note='modulo rotations and relabelling')]
    if 'proof' in f['tags']['stratum_dim'][0]:
        out['dimension'][0]['proof'] = f['tags']['stratum_dim'][0]['proof']
    out['parameterization'] = (dict(level=3, status=PROVED, proof=dict(kind='manuscript', reference=MANUSCRIPT),
                                    **f['parameterization']) if f.get('parameterization') else None)
    out['value'] = [dict(f['value'], id='v3')]
    out['closure'] = f['closure']
    out['tags'] = tag_export(f['tags'])
    # samples, charts and the probe (level 0)
    samples, charts, probes, Ds, ranks = [], [], [], [], []
    for s in f['samples']:
        c, r = cls_by_name[s], recs_by_name[s]
        samples.append(dict(name=s, obs_id=obs_of[s], params=c['params'], drawn_by=c['drawn_by']))
        X = np.array(c['points']); fr = frames(X)
        p = np.array(cover_params(c))
        lin, _ = cone(c)
        J = []
        for k in range(len(p)):
            e = np.zeros(len(p)); e[k] = 1e-6
            J.append(((cover_map(c['N'], p + e) - cover_map(c['N'], p - e)) / 2e-6).ravel())
        J = np.array(J).T
        tb = []
        T = [np.array(t, float) for t in STRATUM_TANGENT[f['name']]]
        if T:
            M = np.array([J @ t for t in T]).T
            Rg = np.array([np.cross(w, X).ravel() for w in np.eye(3)]).T
            Q, _ = np.linalg.qr(Rg)
            M = M - Q @ (Q.T @ M)               # orthogonal to the rotations
            U, sv, _ = np.linalg.svd(M, full_matrices=False)
            tb = [list(map(float, np.round(U[:, k], 12))) for k in range(len(T))]
            ranks.append(int(np.sum(sv > 1e-7 * sv.max())))
        else:
            ranks.append(0)
        charts.append(dict(level=0, status='numerical', sample=s, obs_id=obs_of[s], base_point=c['points'],
                           tangent_basis=tb, coordinates='displacements of x_0, ..., x_%d in R^3, flattened, orthogonal '
                                                         'to the rotations' % (c['N'] - 1),
                           note='tangent basis of the stratum at the sample from the derivative of the parameterization '
                                '(central differences, h = 1e-6), orthonormalized; double precision'))
        pr = probe(c, f['dimension'])
        probes.append(pr)
        Ds.append(abs(pr['D'] - float(F(f['value']['value']['closed_form']))))
    out['samples'] = samples
    out['sampling'] = dict(method='closed-form points chosen by hand: a generic point of the stratum with exact '
                                  'cosines or exact parameters, or the special point itself (sphcap/data/small_n.json, '
                                  'field drawn_by)', count=len(samples))
    out['charts'] = charts
    lv0 = lambda v, note: dict(level=0, status='numerical', value=v, note=note)
    pnote = ('numerical probe at the samples (jobs/atlas_small_n.py probe): D by sphcap.discrepancy exact at '
             'displacements of 1e-6 along great circles; flat directions from the tangent cone of the parameter region, '
             'ascending directions sampled in the complement of the rotations and the parameterization (24 random '
             'directions, both signs, and a basis); flat means a rate below %g, ascending above %g; consistent: %s'
             % (FLAT, ASC, ', '.join('%s %s' % (p['sample'], 'yes' if p['consistent'] else 'NO') for p in probes)))
    p0 = probes[0]
    out['probe'] = dict(level=0, status='numerical', results=probes, note=pnote)
    for k in ('coindex', 'nullity'):
        out['tags'][k].append(lv0(p0[k], pnote))
    out['tags']['index'].append(lv0(p0['index'], pnote))
    out['tags']['stratum_dim'].append(lv0(ranks[0], 'rank, modulo rotations, of the derivative of the '
                                          'parameterization along the stratum at %s (relative tolerance 1e-7)'
                                          % f['samples'][0]))
    out['value'].append(dict(id='v0', level=0, status='numerical', value=float(max(pr['D'] for pr in probes)),
                             note='largest D over the samples (sphcap.discrepancy exact); largest deviation from the '
                                  'closed form %.1e' % max(Ds)))
    if f['name'] in plots:
        out['trajectory'] = plots[f['name']]['trajectory']
    out['plot'] = plots.get('_N%d' % f['N'])
    out['note'] = f['note']
    return out


def trajectories():
    """sampled paths along the parameterized strata (level 0) and the parameter regions for plots"""
    g = GR.G_STAR
    out = {}
    def path(N, ps, note, par):
        pts = []
        for p in ps:
            X = cover_map(N, p)
            pts.append(dict(params={k: float(v) for k, v in zip(par, p)}, X=[list(map(float, x)) for x in X],
                            D=float(Dval(X))))
        return dict(level=0, status='numerical', parameters=par, points=pts, note=note)
    # N = 3 strata
    e = 2 * np.pi / 3
    inr = (e - g) / 2
    loop = [(e + inr * np.cos(t), e + inr * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 49)]
    out['N3_S'] = dict(trajectory=path(3, loop, 'a loop around the equilateral triple at half the inradius of the '
                                               'region, in the cover (a, b); it crosses the folds six times', ['a', 'b']))
    out['N3_A'] = dict(trajectory=path(3, [(g, b) for b in np.linspace(g, np.pi - g / 2, 25)],
                                       'a = g*, b from g* (corner) to pi - g*/2 (edge midpoint)', ['a', 'b']))
    out['N3_Fp'] = dict(trajectory=path(3, [(s, 2 * np.pi - 2 * s) for s in np.linspace(g, e, 25)],
                                        'arcs (s, 2 pi - 2s, s), s from g* (corner) to 2 pi/3 (equilateral)', ['a', 'b']))
    out['N3_Fm'] = dict(trajectory=path(3, [(s, np.pi - s / 2) for s in np.linspace(g, e, 25)],
                                        'arcs (s, pi - s/2, pi - s/2), s from g* (edge midpoint) to 2 pi/3 '
                                        '(equilateral)', ['a', 'b']))
    zs = float((19 + np.sqrt(105)) / 64)
    out['N4_ARC'] = dict(trajectory=path(4, [(z,) for z in np.linspace(zs, 9 / 16, 33)],
                                         'z from (19 + sqrt 105)/64 (symmetric end) to 9/16 (edge end)', ['z']))
    # plots
    c1, c2, c3 = (g, g), (g, 2 * np.pi - 2 * g), (2 * np.pi - 2 * g, g)
    mid = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    m12, m23, m13 = mid(c1, c2), mid(c2, c3), mid(c1, c3)
    E3 = (e, e)
    seg = lambda s, p, q: dict(stratum=s, kind='segment', points=[list(map(float, p)), list(map(float, q))])
    pt = lambda s, p: dict(stratum=s, kind='point', points=[list(map(float, p))])
    feats = [dict(stratum='N3_S', kind='polygon', points=[list(map(float, c)) for c in (c1, c2, c3)])]
    for p, q in ((c1, m12), (m12, c2), (c2, m23), (m23, c3), (c3, m13), (m13, c1)):
        feats.append(seg('N3_A', p, q))
    for cnr in (c1, c2, c3):
        feats.append(seg('N3_Fp', cnr, E3))
    for m in (m12, m23, m13):
        feats.append(seg('N3_Fm', E3, m))
    feats += [pt('N3_C', c) for c in (c1, c2, c3)] + [pt('N3_M', m) for m in (m12, m23, m13)] + [pt('N3_E', E3)]
    out['_N3'] = dict(level=0, status='numerical', coordinates=['a', 'b'], features=feats,
                      fundamental_domain=[list(map(float, p)) for p in (E3, c1, m13)],
                      note='the parameter region of N = 3 in the cover (a, b), double precision: the triangle a, b, c >= '
                           'g*; edges a = g* etc. (N3_A, split by their midpoints N3_M), half-medians from the corners '
                           '(N3_Fp) and to the edge midpoints (N3_Fm) are the folds of relabelling; the shaded '
                           'fundamental domain is g* <= a <= b <= c')
    zl = float((14 + np.sqrt(70)) / 64)
    curve = []
    for z in np.linspace(zl, 9 / 16, 61):
        s, q = 1 - z, z * (1 - z) / (16 * z - 1)
        y = (s + np.sqrt(s * s - 4 * q)) / 2
        curve.append([float(y), float(z)])
    out['_N4'] = dict(level=0, status='numerical', coordinates=['z'],
                      features=[dict(stratum='N4_ARC', kind='segment', points=[[zs], [9 / 16]]),
                                dict(stratum='N4_SYM', kind='point', points=[[zs]]),
                                dict(stratum='N4_EDGE', kind='point', points=[[9 / 16]])],
                      curve=dict(coordinates=['y', 'z'], points=curve,
                                 note='the arc in the (y, z) plane, cover z in [(14 + sqrt 70)/64, 9/16]; the fold '
                                      'y = z is at the symmetric end'),
                      note='the parameter interval of N = 4 modulo relabelling, z in [(19 + sqrt 105)/64, 9/16], '
                           'double precision')
    return out


def stuck_blocks(recs):
    out = {}
    for r in recs:
        if r['N'] >= 3:
            out[r['name']] = S.block(np.array(r['X']))[0]
    return out


def run(out_root=None, log=print):
    out_root = out_root or P.EXPORT
    path, blob, doc = load_input()
    want = input_doc()
    if P.canon(want) != P.canon(json.loads(blob.decode('utf-8'))):
        raise SystemExit('%s differs from what this script writes: run with --write-input and commit it' % INPUT)
    check_classes()
    fsha, fcommit = P.sha_bytes(blob), P.file_commit(path)
    code = code_state()
    plots = trajectories()
    cls_by_name = {c['name']: c for c in doc['classes']}
    fam_by_name = {f['name']: f for f in doc['families']}
    docs = {}
    for N in NS:
        outdir = os.path.join(out_root, 'n%d' % N)
        os.makedirs(outdir, exist_ok=True)
        lines, recs, fams, obs_of = [], [], [], {}
        idx_of = {c['name']: i for i, c in enumerate(doc['classes'])}
        for c in [c for c in doc['classes'] if c['N'] == N]:
            oid = P.obs_id(fsha, c['name'], c)
            obs_of[c['name']] = oid
            lines.append(dict(record='observation', schema_version=P.SCHEMA_VERSION, obs_id=oid, seq=len(lines) + 1,
                              kind='find', container=P.CONTAINER, N=N, raw=c,
                              source=dict(file_name=os.path.basename(INPUT), file_sha256=fsha, record_index=c['name'],
                                          format='other', commit=fcommit),
                              tolerances=P.TOLERANCES,
                              ingested_by=dict(script=SCRIPT, commit=code['commit'], inputs=[INPUT],
                                               note='closed-form record %s of %s' % (c['name'], INPUT))))
            r = find_record(c, fsha, fcommit, idx_of[c['name']])
            r['obs_id'], r['seq'] = oid, lines[-1]['seq']
            r['search'] = {'pass': None, 'hits': None, 'sources': None,
                           'note': 'closed form, not from a search'}
            r['tags'] = entity_tags(c, fam_by_name)
            recs.append(r)
        stuck = stuck_blocks(recs)
        for r in recs:
            if r['name'] in stuck:
                r['stuck_point'] = stuck[r['name']]
        rb = {r['name']: r for r in recs}
        for k, f in enumerate(doc['families']):
            if f['N'] != N:
                continue
            key = 'family:' + f['name']
            oid = P.obs_id(fsha, key, f)
            lines.append(dict(record='observation', schema_version=P.SCHEMA_VERSION, obs_id=oid, seq=len(lines) + 1,
                              kind='family', container=P.CONTAINER, N=N, raw=f,
                              source=dict(file_name=os.path.basename(INPUT), file_sha256=fsha, record_index=key,
                                          format='other', commit=fcommit),
                              tolerances=P.TOLERANCES,
                              ingested_by=dict(script=SCRIPT, commit=code['commit'], inputs=[INPUT],
                                               note='stratum record %s of %s' % (f['name'], INPUT))))
            fr = family_record(f, rb, cls_by_name, obs_of, plots)
            fr['obs_id'], fr['seq'] = oid, lines[-1]['seq']
            fr['source'] = dict(file=INPUT, file_sha256=fsha, pointer='/families/%d' % k, script=SCRIPT,
                                commit=fcommit)
            fams.append(fr)
        jl = ''.join(P.canon(l) + '\n' for l in lines)
        P.write_text(os.path.join(outdir, 'observations.jsonl'), jl)
        seen = {}
        for r in recs:
            seen.setdefault(r['key_basis']['graph_hash'], []).append(r['name'])
        d = dict(functional=E.FUNCTIONAL, N=N, observations=recs, families=fams,
                 shared_hashes={h: v for h, v in seen.items() if len(v) > 1}, search_passes=[], convergence=None,
                 label='Values and tags labelled "proved (manuscript)" (level 3) rest on the hand proofs of the '
                       'unpublished manuscript; everything else is numerical (level 0, double precision) unless its '
                       'level says otherwise. Nothing here is a proof input.')
        P.write_text(os.path.join(outdir, 'observations.json'), json.dumps(d, indent=1) + '\n')
        outs = {f: P.sha_file(os.path.join(outdir, f)) for f in ('observations.jsonl', 'observations.json')}
        man = dict(N=N, functional=E.FUNCTIONAL['id'], schema_version=P.SCHEMA_VERSION,
                   generated_by=dict(script=SCRIPT, commit=code['commit'], last_code_commit=code['last_commit'],
                                     code_modified=code['modified'], scripts=code['scripts'],
                                     classifier_version=S.classifier_version(), versions=P.versions()),
                   inputs=[dict(file=INPUT, sha256=fsha, commit=fcommit, format='small-n closed forms',
                                records=len(recs) + len(fams))],
                   observations=len(recs), families=len(fams), search_passes=0, outputs=outs,
                   command='python %s' % SCRIPT)
        P.write_text(os.path.join(outdir, 'manifest.json'), json.dumps(man, indent=1, sort_keys=True) + '\n')
        docs[N] = d
        log('N=%d: %d observations, %d strata; D = %s' % (N, len(recs), len(fams),
                                                          sorted({r['value'][-1]['value']['closed_form'] for r in recs})))
        for f in fams:
            log('  %s (%s): dim %d, samples %d, probe consistent: %s' % (
                f['name'], f['key_basis']['graph_hash'], f['dimension'][0]['value'], len(f['samples']),
                [p['consistent'] for p in f['probe']['results']]))
    return docs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None)
    ap.add_argument('--write-input', action='store_true', help='write %s and stop' % INPUT)
    ap.add_argument('--check-input', action='store_true', help='run the exact checks of the closed forms and stop')
    a = ap.parse_args(argv)
    if a.write_input:
        P.write_text(os.path.join(ROOT, INPUT), json.dumps(input_doc(), indent=1) + '\n')
        print('wrote %s' % INPUT)
        return
    if a.check_input:
        print('exact checks passed: %s' % ', '.join(check_classes()))
        return
    run(a.out)


if __name__ == '__main__':
    main()
