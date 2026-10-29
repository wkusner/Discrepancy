"""
sphcap.gram
===========

Gram-matrix coordinates and the exact (symbolic) descriptions of the minimal
discrepancy sets for N <= 5.

A configuration X = (x_1,...,x_N) in (S^2)^N is determined up to O(3) by its
Gram matrix G = X^T X:  symmetric, PSD, diag = 1, rank <= 3.  Every ingredient of
the cap discrepancy is an *algebraic function of G* (Tarski--Seidenberg):

    |I| = 3, det G_I > 0 :  t_I^2 = det G_I / (1^T adj(G_I) 1)
    |I| = 2              :  t_I^2 = (1 + G_ab)/2
    x_j in the closed cap of I   <=>   G_{jI} adj(G_I) 1 - det G_I >= 0 .

Hence D(G) is a finite max of terms  n/N -+ (1 -+ t_I)/2  with t_I the square
root of a rational function of G, and the minimiser set {D = D_min} is a
semialgebraic subset of the rank-<=3 PSD cone.
"""
from __future__ import annotations

import numpy as np


def gram(X):
    X = np.asarray(X, float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


# ---------------------------------------------------------------------------
# cap data from the Gram matrix
# ---------------------------------------------------------------------------
def t_squared(G, I):
    """t_I^2 as a rational function of the Gram entries."""
    I = list(I)
    if len(I) == 2:
        return (1 + G[I[0], I[1]]) / 2
    GI = G[np.ix_(I, I)]
    det = np.linalg.det(GI)
    adj = det * np.linalg.inv(GI)
    return det / adj.sum()


def membership(G, I, j):
    """>= 0  iff  x_j lies in the closed small cap of I (needs det G_I > 0, |I|=3)."""
    I = list(I)
    GI = G[np.ix_(I, I)]
    det = np.linalg.det(GI)
    adj = det * np.linalg.inv(GI)
    return G[j, I] @ adj @ np.ones(len(I)) - det


# ---------------------------------------------------------------------------
# N = 3 : minimisers are great-circle triples with all arcs >= g*
# ---------------------------------------------------------------------------
G_STAR = 2 * np.arccos(2.0 / 3.0)           # 96.379...  degrees = 1.68231 rad


def n3_gram(a, b):
    """Gram matrix of three points on a great circle with consecutive arcs a, b, c = 2pi-a-b."""
    c = 2 * np.pi - a - b
    return np.array([[1, np.cos(a), np.cos(c)], [np.cos(a), 1, np.cos(b)], [np.cos(c), np.cos(b), 1]])


def n3_points(a, b):
    c = 2 * np.pi - a - b
    return np.array([[1, 0, 0], [np.cos(a), np.sin(a), 0], [np.cos(a + b), np.sin(a + b), 0]])


def n3_is_minimiser_parameters(a, b, tol=0.0):
    c = 2 * np.pi - a - b
    return min(a, b, c) >= G_STAR - tol


# ---------------------------------------------------------------------------
# N = 4 : disphenoids, the plane cubic and the arc of minimisers
# ---------------------------------------------------------------------------
def disphenoid_points(x, y, z):
    """Vertices (+-p,+-q,+-r) with even number of minus signs, (p,q,r) = sqrt(x,y,z), x+y+z=1."""
    p, q, r = np.sqrt([x, y, z])
    return np.array([[p, q, r], [p, -q, -r], [-p, q, -r], [-p, -q, r]])


def disphenoid_gram(x, y, z):
    a, b, c = 2 * x - 1, 2 * y - 1, 2 * z - 1
    return np.array([[1, a, b, c], [a, 1, c, b], [b, c, 1, a], [c, b, a, 1]])


def n4_cubic(x, y, z):
    """F(x,y,z) = 16xyz - xy - xz - 3x - yz - 3y - 3z + 3  (= (8abc + 7 e2 - 1)/4, a=2x-1,...).

    On x+y+z = 1 all four faces of the disphenoid have t = 1/4 iff F = 0; the
    minimisers (D = 3/8) are the points of this cubic with max(x,y,z) <= 9/16.
    """
    return 16 * x * y * z - x * y - x * z - 3 * x - y * z - 3 * y - 3 * z + 3


def disphenoid_face_t(x, y, z):
    """Common face distance t of the equifacial tetrahedron: t^2 = (abc + e2) / (2 (1 + e2))."""
    a, b, c = 2 * x - 1, 2 * y - 1, 2 * z - 1
    e2 = a * b + b * c + c * a
    return np.sqrt(max((a * b * c + e2) / (2 * (1 + e2)), 0.0))


def disphenoid_discrepancy(x, y, z):
    """Closed form of D on the disphenoid family (verified against the exact method).

    D = max{ 3/4 - V, V, sqrt(u)/2, (1 - sqrt(u))/2  (u = x, y, z), 1/4 },  V = (1 - t)/2.
    """
    t = disphenoid_face_t(x, y, z)
    V = (1 - t) / 2
    r = np.sqrt([x, y, z])
    return max(0.75 - V, V, *(r / 2), *((1 - r) / 2), 0.25)


N4_ARC_END_SYMMETRIC = ((13 - 105 ** 0.5) / 32, (19 + 105 ** 0.5) / 64, (19 + 105 ** 0.5) / 64)
N4_ARC_END_EDGE = ((14 - 70 ** 0.5) / 64, (14 + 70 ** 0.5) / 64, 9.0 / 16)


def n4_cubic_curve_points(num=4000):
    """Points (x,y,z) of the cubic in the open simplex, solved from a quadratic in y."""
    pts = []
    for x in np.linspace(1e-4, 1 - 1e-4, num):
        A = 16 * x - 1
        Bc = 16 * x * x - 17 * x + 1
        C = x - x * x
        if abs(A) < 1e-12:
            ys = [-C / Bc]
        else:
            disc = Bc * Bc - 4 * A * C
            if disc < 0:
                continue
            ys = [(-Bc + s * disc ** 0.5) / (2 * A) for s in (1, -1)]
        for y in ys:
            if y > 0 and x + y < 1:
                pts.append((x, y, 1 - x - y))
    return np.array(pts)


# ---------------------------------------------------------------------------
# N = 5 : the C_2v-symmetric ansatz and its exact elimination
# ---------------------------------------------------------------------------
def n5_symmetric_points(c, e):
    """P3 = pole; P0,P2 = (0,+-b,c); P1,P4 = (+-a,0,e);  b = sqrt(1-c^2), a = sqrt(1-e^2).

    Order of the returned rows is the labelling used in the note: x_0..x_4.
    """
    a, b = np.sqrt(1 - e * e), np.sqrt(1 - c * c)
    return np.array([[0, b, c], [a, 0, e], [0, -b, c], [0, 0, 1.0], [-a, 0, e]])


def n5_symmetric_equations():
    """Area-balance equations for the three orbits of active caps (sympy).

    unknowns (c, e, D);  tau = 2D - 1/5.
      E1: cap through x0,x2,x1 :  t^2 = (3/5 - 2D)^2
      E2: cap through x0,x1,x4 :  t^2 = tau^2
      E3: cap through x0,x1,x3 :  t^2 = tau^2        (simplified by (1-e)(1-c))
    """
    import sympy as sp
    c, e, D = sp.symbols("c e D")
    tau = 2 * D - sp.Rational(1, 5)
    s = sp.Rational(3, 5) - 2 * D
    E1 = sp.expand((1 - e ** 2) * c ** 2 - s ** 2 * ((c - e) ** 2 + 1 - e ** 2))
    E2 = sp.expand((1 - c ** 2) * e ** 2 - tau ** 2 * ((e - c) ** 2 + 1 - c ** 2))
    E3 = sp.expand((1 + e) * (1 + c) - tau ** 2 * (3 + e + c - e * c))
    return (c, e, D), (E1, E2, E3)


def n5_eliminate():
    """Eliminate (c, e): returns the factor list of the resultant in D (sympy)."""
    import sympy as sp
    (c, e, D), (E1, E2, E3) = n5_symmetric_equations()
    R12 = sp.factor_list(sp.resultant(E1, E2, e))[1]
    R13 = sp.factor_list(sp.resultant(E1, E3, e))[1]
    big12 = [f for f, m in R12 if sp.Poly(f, c, D).total_degree() > 1][0]
    big13 = [f for f, m in R13 if sp.Poly(f, c, D).total_degree() > 1][0]
    R = sp.resultant(big12, big13, c)
    return sp.factor_list(sp.Poly(R, D).as_expr())[1]


# minimal polynomials (coefficients of x^0 .. x^k) --------------------------------------------
# verified to > 1400 digits against 1500-digit Newton solutions (see data/critical_configurations.json)
MINPOLY = {
    "N5_best":   [173504, 2726960, -16763225, -87317250, 351851875, 762400000, -1882875000,
                  -1826875000, -8985937500, 29968750000, -3437500000, -31250000000, 15625000000],
    "N5_valley": [71, 940, -3200, -2000, 2500],
    "N6_best":   [1600, 16320, -65952, -2799144, 22584177, -28320192, -298446768, 1223320320,
                  -556162848, -5744286720, 13317675264, -11246708736, 3401222400],
}
