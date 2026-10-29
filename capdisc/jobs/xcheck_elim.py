"""Second-kernel check of the elimination: msolve (Groebner bases, F4) on the symmetry-reduced
systems msolve/N5_C2v.ms and msolve/N6_D3.ms, eliminating everything but D; factor the eliminant
(sympy) and check that P_N(N D) is a factor and the only one with a root in the enclosure of D*.
The first kernel is jobs/sym_cert.py (sympy resultants from the geometric orbit equations).
usage: python jobs/xcheck_elim.py [MSOLVE]       writes certificates/xcheck_elim.json"""
import sys, os, re, json, subprocess, tempfile, sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); CERT = os.path.join(ROOT, 'certificates')
MS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('MSOLVE', 'msolve')
D, u = sp.symbols('D u')
P = {5: [64, -640, -352, 15344, -23004, -23384, -120504, 243968, 562963, -698538, -670529, 545392, 173504],
     6: [25, -496, 3524, -9120, -5298, 69920, -102348, -58272, 278817, -207344, -29312, 43520, 25600]}
# the interval of Theorem 10 (in u = N D), which contains N D* by xcheck_sym
THM = {5: ('1.616628254990254', '1.616628254990268'), 6: ('1.672072891057447', '1.672072891057457')}
ENC = {N: (sp.Rational(a) / N, sp.Rational(b) / N) for N, (a, b) in THM.items()}
out = {}
for N, f in ((5, 'N5_C2v.ms'), (6, 'N6_D3.ms')):
    src = os.path.join(ROOT, 'msolve', f)
    nvars = len(open(src).readline().split(','))
    o = tempfile.mktemp()
    subprocess.run([MS, '-f', src, '-e', str(nvars - 1), '-g', '2', '-o', o, '-t', '4'], check=True, capture_output=True)
    txt = open(o).read()
    body = txt[txt.index('['):txt.rindex(']') + 1].strip('[]:\n ')
    elims = [sp.sympify(s.replace('^', '**')) for s in body.split(',')]
    elims = [e for e in elims if e.free_symbols <= {D}]
    assert len(elims) == 1, "expected one univariate eliminant"
    E = sp.Poly(elims[0], D)
    PD = sp.Poly(sum(c * (N * D) ** (12 - i) for i, c in enumerate(P[N])), D).primitive()[1]
    facs = [sp.Poly(g, D) for g, _ in sp.factor_list(E.as_expr())[1] if sp.Poly(g, D).degree() > 0]
    lo, hi = ENC[N]
    hits = [g for g in facs if g.count_roots(lo, hi) > 0]
    same = len(hits) == 1 and (hits[0].primitive()[1] == PD or hits[0].primitive()[1] == -PD)
    out[N] = dict(eliminant_degree=E.degree(), factor_degrees=[g.degree() for g in facs],
                  factor_meeting_enclosure_is_P_N=bool(same), verified=bool(same),
                  note='msolve rational reconstruction is multi-modular (probabilistic): a cross-check of the deterministic sympy resultants', kernel='msolve (F4 Groebner basis, elimination order) + sympy factorization')
    print(f"N={N}: msolve eliminant degree {E.degree()}, factors {[g.degree() for g in facs]}; the only factor with a root in the enclosure is P_{N}(N D): {same}")
json.dump(out, open(os.path.join(CERT, 'xcheck_elim.json'), 'w'), indent=1)
