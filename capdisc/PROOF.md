# The proof for N = 5 and N = 6: what the code proves, and why that suffices

This file is the written argument behind the computer-assisted result for five and six points
("Verified Result" in the Monthly article, Theorem `thm:computer` in the Experimental Mathematics
paper). It is meant to be read next to the code. Every step names the lemma it uses, the script that
checks it, the arithmetic it trusts, and the certificate file it writes. The words *verified*,
*cross-verified* etc. are used as defined in [VERIFICATION.md](VERIFICATION.md).

> **Status (draft, 2026-10-30).** This document describes the code as it stands at commit
> `d8831ce`. Section 6 lists the places where the code or its records fall short of what is
> described here; they are being fixed in the refactor described in
> [REFACTOR_PLAN.md](REFACTOR_PLAN.md).

## 0. The claim

For N points x₁,…,x_N on the unit sphere S², the spherical cap discrepancy is

  D(X) = sup over closed spherical caps C of | #(X ∩ C)/N − σ(C) |,

with σ the normalized area. Let P₅ and P₆ be the integer polynomials of degree 12 in the variable u = N·D printed in both
papers and in `jobs/xcheck_roots.py` (coefficients from u¹² down).

**Claim.** For N = 5 and N = 6:

1. min D over all N-point configurations equals D_N = u_N / N, where u_N is the unique root of P_N
   in the isolating interval of `certificates/xcheck_roots.json`
   (D₅ = 0.32332565099805…, D₆ = 0.27867881517624…);
2. P_N is irreducible over ℚ, so D_N is algebraic of degree 12;
3. the minimum is attained by exactly one configuration up to rotations, reflections and
   relabelings: Y(s*) of the symmetric families in Section 5 of the Exp. Math. paper, with symmetry
   group C₂ᵥ (N = 5) and D₃ (N = 6);
4. the Gram matrix of the minimizer has entries in ℚ(D_N), given in `sphcap/data/gram_exact.json`.

## 1. The shape of the argument

The configurations, modulo rotations, reflections and relabelings, are parametrized by a
fundamental domain F in a box Q ⊂ ℝ^{2N−3} (Step 1). The argument is:

* **Away from the minimizer** (Steps 2–3): every point of F outside a few small boxes B₁,…,B_m
  has D ≥ T, where T is a dyadic rational slightly above D_N.
* **Inside each box** (Step 4): D ≥ D*_k on B_k, with equality only at one point q*_k, and
  D*_k ≤ T.
* **The boxes hold one configuration** (Steps 5–6): every q*_k is the same configuration Y(s*),
  up to isometry, and its value is the root of P_N.
* **Assembly** (Step 7).

The computation never needs to be trusted for anything except the inequalities and identities it
finally checks. Choices (axes, LP weights, splitting order, approximate inverses) are made in
floating point, and a poor choice can only make a check fail, never make a false check pass.

## 2. The steps

### Step 1. Fundamental domain

*Mathematics.* Lemma `lem:domain` (Exp. Math. §6.1): every configuration has a representative q
in F ⊂ Q, where x₁ is the north pole, x₂ lies in the half-plane y = 0, x ≥ 0, the pair (x₁, x₂) is a
closest pair, θ₃ ≤ … ≤ θ_N, and 0 ≤ φ₃ ≤ π.

*What the code does.* Cells are discarded when they provably miss F: θ_i > θ_{i+1} throughout,
or some pair closer than (x₁, x₂) throughout.

| | pipeline A | pipeline B |
|---|---|---|
| mesh | `csrc/capcert.c` (`-DRIGOROUS`), IEEE intervals `csrc/ia.h`, `csrc/rig.h` | `csrc/capcert_arb.c` (Arb) |
| refinement | `sphcap/certify.py::misses_domain` (Arb) | `jobs/xcheck_refine.py::misses_domain` (mpmath.iv) |

*Covering.* The initial box is widened by one ulp beyond [0, fl(π)] etc., and every bisection
enlarges the children's half-widths to cover the rounding of their centers (Exp. Math. §6.3,
"Covering"; `jobs/refine.py::process`).

### Step 2. The mesh: D ≥ T on every cell but the survivors

*Mathematics.* Lemma `lem:cell` (cell bound). For a cell B with center configuration X, per-point
motion bounds E_i (equation `eq:motion`) and any axis w, D(Y) ≥ k/N − (1 − c_(k))/2 and
D(Y) ≥ (1 − C_(k))/2 − (k−1)/N for all Y in B. Axes are those of the capture caps of X
(Lemma `lem:capture`), both orientations.

*What the code does.* `capcert_rig N T MIN_EPS OUT` bisects F and proves D ≥ T on every cell except
the *survivors*, which it writes to `OUT.surv.*` once they are smaller than MIN_EPS.

*Certificate.* `certificates/proof_N{5,6}.json` (`capcert_result`, `capcert_stamp`).
Pipeline B: `capcert_arb`, records in `certificates/pipelineB_N{5,6}.json`.

### Step 3. Refinement: D ≥ T on every survivor, outside the boxes

*Mathematics.* The first-order bound `eq:lpbound` (Exp. Math. §6.4): with counts frozen on the cell
and gradient enclosures m_C ± r_C, for any weights λ ≥ 0 with Σλ = 1,
D(Y) ≥ Σ λ_C a_C − Σ_j |Σ λ_C m_Cj| h_j. The weights come from a floating-point LP; only the
right-hand side is evaluated rigorously.

*What the code does.* `jobs/refine.py N PREFIX LOCALCERT` takes every survivor and closes it when
(a) it lies inside one of the boxes B_k of Step 4, (b) it misses F, or (c) the bound above is ≥ T
(`sphcap/certify.py::cell_lower_bound`, Arb). Otherwise it bisects, down to a minimum width;
a cell that still fails is reported, and the proof fails.

*Certificate.* `certificates/proof_N{5,6}.json`, field `refine`: `failed = 0`. For N = 5: 590,986
survivors refined into 846,246 cells, 10 of them inside boxes. For N = 6: 6,509,863 survivors into
6,531,051 cells, 8 inside boxes. Pipeline B: `jobs/xcheck_refine.py` (mpmath.iv),
`certificates/pipelineB_N{5,6}.json`.

### Step 4. Local verification at each image of the minimizer

A configuration on the boundary of F has several representatives ("images") in F. The optimum has
3 images for N = 5 and 2 for N = 6 (`sphcap/certify.py::domain_images`); each gets its own box.

*Mathematics.* For image k, let A be its 2N−2 worst caps and Φ(q, D) = (φ_C(q) − D)_{C∈A}.

1. *Krawczyk test.* On a box Z of radius 10⁻¹² around the numerical solution, K(Z) ⊂ int Z, so Φ has
   exactly one zero (q*_k, D*_k) in Z. This encloses D*_k in an interval of width about 10⁻¹⁵.
2. *Frozen counts.* On the larger box B_k = q*_k + [−r, r]^{2N−3} (r = 3·10⁻⁴ for N = 5,
   10⁻⁴ for N = 6) no point crosses the boundary of a cap in A, so D ≥ φ_C on B_k for C ∈ A.
3. *Orthant test.* On each of the 2^{2N−3} orthants of B_k there are weights μ ≥ 0 that make the
   μ-combination of the interval gradients strictly positive in the direction of the orthant. Hence
   D > D*_k on B_k \ {q*_k}.
4. *Other caps.* Every cap of the capture list outside A is below D*_k on Z, so D(q*_k) = D*_k.

*What the code does.* `jobs/local_cert.py NAME r` (Arb). Pipeline B: `jobs/xcheck_local.py`
(mpmath.iv, independently written).

*Certificate.* `certificates/localcert_N{5,6}_best.json`: per image `krawczyk`, `D_enclosure`,
`counts_frozen_on_box`, `orthants_failed = 0`, `others_below`. Pipeline B:
`certificates/xcheck_local_N{5,6}_best.json`.

*Hand-off to Step 3.* T is the largest upper end of the D*_k enclosures (`run_proof.sh`), and the
boxes B_k are the exclusion boxes of the refinement.

### Step 5. Symmetry and the exact value, per image

*Mathematics.* In the symmetric coordinates s ((c, e) for N = 5, (a, b, h) for N = 6) the family
Y(s) is invariant under its group by construction, and the worst caps fall into three orbits. A
second Krawczyk test (128-bit balls, radius 10⁻³⁰) gives a unique solution s* of the orbit
equations φ_R = D. By symmetry every worst cap has value D at Y(s*); rotated into the normal form,
Y(s*) solves Φ = 0 inside Z, so it *is* q*_k. Iterated resultants of the squared orbit equations
give a polynomial in D vanishing at D*; interval evaluation selects the factor that can vanish
there, and that factor is P_N(N·D) up to a constant.

*What the code does.* `jobs/sym_cert.py NAME k` (Arb + sympy resultants + FLINT factorization).
Pipeline B: `jobs/xcheck_sym.py` (mpmath.iv); elimination cross-checked by msolve Gröbner bases
(`jobs/xcheck_elim.py`; probabilistic rational reconstruction, so a check, not the proof).

*Certificate.* `certificates/symcert_N{5,6}_best_img{k}.json`: `krawczyk_symmetric`,
`gauge_in_local_box`, `D_in_local_enclosure`, `P_divides_eliminant`, `s_box_mid`,
`s_box_radius`, `s_enclosure`. Pipeline B: `certificates/xcheck_sym_*.json`,
`certificates/xcheck_elim.json`.

### Step 6. The images are one configuration

*Mathematics.* Step 5 shows that each box holds a symmetric configuration Y(s*_k), but the
symmetric system could have several solutions (for N = 5, P₅ has a second root 1.61681… very close
to u₅). For each image k there is a map σ of the symmetric parameters that is an isometry of the
family (N = 5: the identity; N = 6: sign changes, which are the reflections in the coordinate
planes) with σ(enclosure of s*_k) inside the symmetric Krawczyk box of image 0. Since the solution
there is unique, σ(s*_k) = s*₀, so Y(s*_k) is congruent to Y(s*₀).

*What the code does.* `jobs/congruence_cert.py NAME` (Arb). The choice of σ is numerical; the
containment is checked in Arb, against a box shrunk by the factor 1 − 10⁻⁶.

*Certificate.* `certificates/congruence_N{5,6}_best.json`: `congruent = true`.
Pipeline B: **not yet implemented** (Section 6).

### Step 7. Exact data

* *Root isolation.* P_N is irreducible over ℚ and has exactly one root in the isolating interval
  (`jobs/xcheck_roots.py`, sympy; FLINT in `jobs/sym_cert.py`). Certificate
  `certificates/xcheck_roots.json`.
* *Gram matrix.* `jobs/gram_verify.py` proves that the integer-relation guess G_e in
  `sphcap/data/gram_exact.json` is the Gram matrix of q*: rank 3 and every worst-cap equation hold
  exactly in ℚ(D*) (sympy and FLINT); positive definiteness, the signs lost by squaring, and
  membership of the realization in B are checked in Arb and mpmath.iv. Certificate
  `certificates/gramcert_N{5,6}_best.json`.

## 3. Assembly

Let X be any configuration of N points (N = 5 or 6) and q a representative of X in F (Step 1).

* If q lies in no box B_k, then D(X) ≥ T by Steps 2 and 3. Since T is a dyadic rational and D_N
  is irrational (Step 7: P_N is irreducible of degree 12), and D_N ≤ T (Step 4), we have
  D(X) ≥ T > D_N.
* If q lies in B_k, then D(X) ≥ D*_k, with equality only at q = q*_k (Step 4).
* All q*_k are congruent to Y(s*) (Steps 5–6), so all D*_k are equal to D(Y(s*)), and
  N·D(Y(s*)) is the root of P_N in the isolating interval (Steps 5 and 7).

Hence min D = D_N, attained exactly by the configurations congruent to Y(s*). The symmetry group
and the Gram matrix are those of Y(s*) (Steps 5 and 7). ∎

## 4. What is proved by hand, not by code

The code checks inequalities; the lemmas that make those inequalities sufficient are proved in
the papers and are not checked by any program:

| lemma | where |
|---|---|
| Capture lemma (finitely many caps suffice) | Monthly `lem:capture`; Exp. Math. Appendix A |
| Lipschitz bound | Monthly `prop:lip`; Exp. Math. Appendix A |
| Frozen counts, balance, sharpness | Monthly `lem:frozen`, `thm:balance`, `thm:sharp` |
| Fundamental domain | Exp. Math. `lem:domain` |
| Cell bound | Exp. Math. `lem:cell` |
| First-order (LP) bound | Exp. Math. `eq:lpbound` |
| Krawczyk test; orthant test | Exp. Math. §6.4 (Krawczyk 1969; Rump 2010) |
| Interval arithmetic with correct rounding | Exp. Math. §6.3 (IEEE 754) |

Both pipelines implement the same lemmas, so a misapplied lemma would not be caught by the
cross-check. That is why the lemmas are written out and proved by hand.

## 5. Trusted base (summary; details to go in TRUST.md)

* IEEE 754 correct rounding of +, −, ×, ÷, √ on the build machine, with `-ffp-contract=off
  -fno-fast-math` (pipeline A mesh). sin and cos are *not* trusted: they are Taylor polynomials
  with a Lagrange remainder.
* Arb / FLINT (via python-flint and from C), mpmath's `iv` module, sympy's exact arithmetic and
  real-root isolation.
* The C compiler, the Python interpreter, numpy (only where floating-point values are converted to
  balls; see Section 6), and the operating system.
* Not trusted: the search, scipy's LP solver, every floating-point result, msolve (a check only).

## 6. Known gaps in the current code and records (to be fixed in the refactor)

These do not, as far as we know, make any step false, but they make the steps harder to audit or
leave a record incomplete.

1. **N = 6, pipeline A record.** `certificates/proof_N6.json` was transcribed from the server log;
   it has no `capcert_stamp`, and the local certificate used on the server is not stored. Re-run
   `run_proof.sh` for N = 6 with `UPDATE_CERTS=1` and store all outputs.
2. **Congruence in pipeline B.** Step 6 has no second-kernel version yet.
3. **Floating point inside the trusted core.** Some rigorous bounds are assembled from
   floating-point quantities with explicit slack factors, e.g. `np.dot(rad, w) * (1 + 1e-12)` in
   `sphcap/certify.py::cap_data`, and the box test `np.abs(c - m) + w <= r * (1 - 1e-12)` in
   `jobs/refine.py::inside_box` (correct because c − m is exact by Sterbenz's lemma when c is
   close to m, but that takes an argument). Every such inequality should be done in Arb, so that an
   auditor needs to trust only the ball arithmetic.
4. **Decimal values in JSON.** Box centers and enclosures are stored as decimal strings; they
   should be stored as exact binary values (hex floats or Arb ball strings with outward rounding),
   so that reading a certificate back cannot change it.
5. **Versions.** Library versions, compiler flags and the commit are not recorded in every
   certificate, and `flint.ctx.prec` is set implicitly in some scripts.
6. **Resume.** `run_proof.sh` allows resuming a mesh run from a progress file; in proof mode a run
   should start clean, and record per-root survivor counts.
7. **Chain checking.** Nothing yet checks mechanically that the certificates fit together: that
   the boxes excluded in Step 3 are exactly the boxes B_k of Step 4, that T is the value Step 4
   produced, that every image of `domain_images` has a local, symmetric and congruence
   certificate, and that all files come from the same commit.
8. **The polynomials are not in the certificates.** P₅ and P₆ appear only as literals in the
   scripts; the root and elimination certificates should record them (and their hash), so that the
   chain checker can compare them with the papers.
9. **Tests.** The unit tests for the interval kernel, the cell bound and the Krawczyk test exist
   only as scratch scripts.

## 7. How to re-run (current state)

```
bash build.sh                       # capcert (float) and capcert_rig (interval), with self-tests
THREADS=16 NS="5 6" bash run_proof.sh      # pipeline A: Steps 2-5
python jobs/congruence_cert.py N5_best     # Step 6 (and N6_best)
python jobs/gram_verify.py N5_best         # Step 7 (and N6_best)
THREADS=32 NS="5 6" bash run_xcheck.sh     # pipeline B: Steps 2-5 and 7, then a verdict
```

N = 5 takes about an hour on 4 threads; N = 6 about 75 minutes on 32 threads. The refactor replaces
this with one command per N and a checker for the certificate chain.
