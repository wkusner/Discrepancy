# The proof for N = 5 and N = 6: what the code proves, and why that suffices

This file is the written argument behind the computer-assisted result for five and six points
("Verified Result" in the Monthly article, Theorem `thm:computer` in the Experimental Mathematics
paper). It is meant to be read next to the code. Every step names the lemma it uses, the script that
checks it, the arithmetic it trusts, and the certificate file it writes. The words *verified*,
*cross-verified* etc. are used as defined in [VERIFICATION.md](VERIFICATION.md).

> **Status (draft, 2026-10-31).** Pipeline A now runs from the trusted core `proof/` (branch
> `audit-refactor`), driven by `verify.sh N` and checked by `check/check_chain.py`. The trusted base is in
> [TRUST.md](TRUST.md). Section 6 lists what is fixed and what is still open; the certificates in
> `certificates/` are those of the reference runs at commit `d8831ce` until the new reference runs are stored
> in `certificates/verify_N{5,6}/`.

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
| refinement | `proof/domain.py::misses_domain` (Arb) | `jobs/xcheck_refine.py::misses_domain` (mpmath.iv) |

*Covering.* The initial box is widened by one ulp beyond [0, fl(π)] etc., and every bisection
enlarges the children's half-widths to cover the rounding of their centers (Exp. Math. §6.3,
"Covering"; in the refinement the enlarged half-width is computed in Arb, `proof/refine.py::process`).

### Step 2. The mesh: D ≥ T on every cell but the survivors

*Mathematics.* Lemma `lem:cell` (cell bound). For a cell B with center configuration X, per-point
motion bounds E_i (equation `eq:motion`) and any axis w, D(Y) ≥ k/N − (1 − c_(k))/2 and
D(Y) ≥ (1 − C_(k))/2 − (k−1)/N for all Y in B. Axes are those of the capture caps of X
(Lemma `lem:capture`), both orientations.

*What the code does.* `capcert_rig N T MIN_EPS OUT` bisects F and proves D ≥ T on every cell except
the *survivors*, which it writes to `OUT.surv.*` once they are smaller than MIN_EPS.

*Certificate.* `mesh.out` and the completion stamp `mesh.done` (N, T as a hex float, mode, keep fraction,
exclusion balls, survivor count), in the output directory of `verify.sh`; formerly
`certificates/proof_N{5,6}.json`. Pipeline B: `capcert_arb`, records in `certificates/pipelineB_N{5,6}.json`.

### Step 3. Refinement: D ≥ T on every survivor, outside the boxes

*Mathematics.* The first-order bound `eq:lpbound` (Exp. Math. §6.4): with counts frozen on the cell
and gradient enclosures m_C ± r_C, for any weights λ ≥ 0 with Σλ = 1,
D(Y) ≥ Σ λ_C a_C − Σ_j |Σ λ_C m_Cj| h_j. The weights come from a floating-point LP; only the
right-hand side is evaluated rigorously.

*What the code does.* `python -m proof.refine N PREFIX LOCALCERT` takes every survivor and closes it when
(a) it lies inside one of the boxes B_k of Step 4 (decided in Arb), (b) it misses F, or (c) the bound above
is ≥ T (`proof/cellbound.py::cell_lower_bound`; every quantity in the bound is an Arb ball). Otherwise it bisects, down to a minimum width;
a cell that still fails is reported, and the proof fails.

*Certificate.* `mesh.refine.json`: `failed = 0`, `T_hex`, `exclusion_boxes`, and the SHA-256 of the local
certificate, the mesh stamp and every survivor file it read. Reference runs at `d8831ce`
(`certificates/proof_N{5,6}.json`): for N = 5, 590,986
survivors refined into 846,246 cells, 10 of them inside boxes. For N = 6: 6,509,863 survivors into
6,531,051 cells, 8 inside boxes. Pipeline B: `jobs/xcheck_refine.py` (mpmath.iv),
`certificates/pipelineB_N{5,6}.json`.

### Step 4. Local verification at each image of the minimizer

A configuration on the boundary of F has several representatives ("images") in F. The optimum has
3 images for N = 5 and 2 for N = 6; each gets its own box. Finding the images, and the worst caps of
each, is a *choice* made by untrusted code (`explore/make_hints.py` → `proof/hints/<NAME>.json`): a missed
image would leave cells that the refinement cannot close, and a wrong cap or count makes Step 4 fail.

*Mathematics.* For image k, let A be its 2N−2 worst caps and Φ(q, D) = (φ_C(q) − D)_{C∈A}.

1. *Krawczyk test.* On a box Z of radius 10⁻¹² around the numerical solution, K(Z) ⊂ int Z, so Φ has
   exactly one zero (q*_k, D*_k) in Z. This encloses D*_k in an interval of width about 10⁻¹⁵.
2. *Frozen counts.* On the larger box B_k = q*_k + [−r, r]^{2N−3} (r = 3·10⁻⁴ for N = 5,
   10⁻⁴ for N = 6) no point crosses the boundary of a cap in A, so D ≥ φ_C on B_k for C ∈ A.
3. *Orthant test.* On each of the 2^{2N−3} orthants of B_k there are weights μ ≥ 0 that make the
   μ-combination of the interval gradients strictly positive in the direction of the orthant. Hence
   D > D*_k on B_k \ {q*_k}.
4. *Other caps.* Every cap of the capture list outside A is below D*_k on Z, so D(q*_k) = D*_k.

*What the code does.* `python -m proof.local NAME r` (Arb; the orthant test uses the exact ends of the
gradient balls, and "other caps below D*" is decided in Arb). Pipeline B: `jobs/xcheck_local.py` (mpmath.iv,
independently written).

*Certificate.* `localcert_<NAME>.json`: per image `krawczyk`, `D_enclosure` (doubles rounded outward;
also `D_enclosure_hex` and the Arb ball `D_ball`),
`counts_frozen_on_box`, `orthants_failed = 0`, `others_below`. Pipeline B:
`certificates/xcheck_local_N{5,6}_best.json`.

*Hand-off to Step 3.* T is the largest upper end of the D*_k enclosures (`verify.sh`), and the
boxes B_k are the exclusion boxes of the refinement; `check/check_chain.py` checks both (C4, C5).

### Step 5. Symmetry and the exact value, per image

*Mathematics.* In the symmetric coordinates s ((c, e) for N = 5, (a, b, h) for N = 6) the family
Y(s) is invariant under its group by construction, and the worst caps fall into three orbits. A
second Krawczyk test (128-bit balls, radius 10⁻³⁰) gives a unique solution s* of the orbit
equations φ_R = D. By symmetry every worst cap has value D at Y(s*); rotated into the normal form,
Y(s*) solves Φ = 0 inside Z, so it *is* q*_k. Iterated resultants of the squared orbit equations
give a polynomial in D vanishing at D*; interval evaluation selects the factor that can vanish
there, and that factor is P_N(N·D) up to a constant.

*What the code does.* `python -m proof.symmetric NAME k` (Arb + sympy resultants + FLINT factorization).
Pipeline B: `jobs/xcheck_sym.py` (mpmath.iv); elimination cross-checked by msolve Gröbner bases
(`jobs/xcheck_elim.py`; probabilistic rational reconstruction, so a check, not the proof).

*Certificate.* `symcert_<NAME>_img<k>.json`: `krawczyk_symmetric`, `gauge_in_local_box`,
`D_in_local_enclosure`, `P_divides_eliminant`, `P_N`, `P_N_irreducible`, `P_N_is_the_factor`, `s_box_mid`,
`s_box_radius`, `s_enclosure`. Pipeline B: `certificates/xcheck_sym_*.json`,
`certificates/xcheck_elim.json`.

### Step 6. The images are one configuration

*Mathematics.* Step 5 shows that each box holds a symmetric configuration Y(s*_k), but the
symmetric system could have several solutions (for N = 5, P₅ has a second root 1.61681… very close
to u₅). For each image k there is a map σ of the symmetric parameters that is an isometry of the
family (N = 5: the identity; N = 6: sign changes, which are the reflections in the coordinate
planes) with σ(enclosure of s*_k) inside the symmetric Krawczyk box of image 0. Since the solution
there is unique, σ(s*_k) = s*₀, so Y(s*_k) is congruent to Y(s*₀).

*What the code does.* `python -m proof.congruence NAME` (Arb). The choice of σ is numerical; the
containment is checked in Arb, against a box shrunk by the factor 1 − 10⁻⁶.

*Certificate.* `congruence_<NAME>.json`: `congruent = true`.
Pipeline B: **not yet implemented** (Section 6).

### Step 7. Exact data

* *Root isolation.* P_N is irreducible over ℚ and has exactly one root in the isolating interval
  (`jobs/xcheck_roots.py`, sympy; FLINT in `proof/symmetric.py`). Certificate
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

## 5. Trusted base

See [TRUST.md](TRUST.md). In short: IEEE 754 correct rounding and the C compiler (mesh of pipeline A); Arb
(python-flint and FLINT from C); mpmath's `iv`; sympy's exact arithmetic. Not trusted: the search, the hints,
scipy's LP solver, every floating-point result, msolve (a check only).

## 6. Status of the gaps found in the code audit

*Fixed on `audit-refactor`:*

* **Floating point inside the trusted core** (old item 3): every inequality of Steps 3–6 is decided in Arb;
  no slack factors, no Sterbenz arguments (`proof/cellbound.py`, `proof/local.py`, `proof/refine.py`).
* **Exact values in certificates** (old item 4): doubles are written rounded outward and as hex floats; balls
  as Arb strings (`proof/certio.py`).
* **Versions and stamps** (old item 5): every certificate carries a stamp (step, commit, versions, precision,
  command line, machine type, date, run time, inputs by SHA-256); `flint.ctx.prec` is set explicitly.
* **Resume** (old item 6): `verify.sh` deletes its output directory and runs every step from scratch.
* **Chain checking** (old item 7): `check/check_chain.py` (standard library only) checks C1–C8: images,
  symmetric and congruence certificates for every image, proof-mode mesh, T and the exclusion boxes, input
  hashes, D* < T strictly, agreement with the papers' P_N, interval and digits, one commit.
* **Polynomials in the certificates** (old item 8): the symmetric certificates record P_N and its
  irreducibility; `check/paper_values.json` holds the values printed in the papers.
* **Tests** (old item 9): `tests/test_proof_core.py` (cell bound against the exact discrepancy, bisection
  covering, outward rounding, domain test, and mutation tests that must make Step 4 fail).

*Still open:*

1. **N = 6 reference run of pipeline A** with the new core, on the server (`verify.sh 6`), and storing the
   reference outputs of both N in `certificates/verify_N{5,6}/`.
2. **Congruence in pipeline B.** Step 6 has no second-kernel version yet; pipeline B is not yet driven by a
   `verify.sh`-style script or checked by the chain checker.
3. **The mesh source** `csrc/capcert.c` still contains the floating-point exploration mode behind
   `#ifndef RIGOROUS`; the rigorous mesh should be its own file.
4. **Step 7** (`jobs/gram_verify.py`, `jobs/xcheck_roots.py`) is not yet in `proof/` and not stamped.

## 7. How to re-run

```
bash build.sh                                 # capcert_rig (interval mesh), with self-tests
python -m pytest tests/test_proof_core.py -q  # tests of the trusted core (a few seconds)
THREADS=16 bash verify.sh 5                   # pipeline A, Steps 4, 5, 6, 2, 3 -> results/verify_N5/
python3 check/check_chain.py 5 results/verify_N5          # chain checker, C1-C8
python jobs/gram_verify.py N5_best            # Step 7 (and N6_best)
THREADS=32 NS="5 6" bash run_xcheck.sh        # pipeline B: Steps 2-5 and 7, then a verdict
```

N = 5 takes about 2 hours on 2 threads; N = 6 about 75 minutes on 32 threads.
