# Verification vocabulary and status

A rigorous computation done in one kernel is called **verified**, and the label names the method
(interval-verified, Arb-verified, Krawczyk-verified, exact (kernel)). A result that a second,
independent kernel has also checked is called **cross-verified**, and the label names both kernels,
e.g. "cross-verified (Arb + mpmath.iv)". Paper text, code output, `STATUS.md`, JSON files and
commit messages all follow this file.

The word **certified** is not used for computations. In critconfig-atlas (data spec v0.2) it is
reserved for level 4, a certificate accepted by a named checker; nothing here is at level 4.
Files written before 2026-10-29 may still contain the old key `certified`; readers accept it as
a synonym of `verified`.

## Vocabulary

| label | meaning | example |
|---|---|---|
| **cross-verified (K1 + K2)** | rigorous, and confirmed by a second, independent kernel K2 (different arithmetic library and different code path) | the local verification at the minimizer: cross-verified (Arb + mpmath.iv) |
| **interval-verified** | rigorous single-kernel run in our IEEE interval arithmetic (`csrc/ia.h`, `csrc/rig.h`): one-ulp outward rounding, Taylor sin/cos with remainder | the mesh stage (`capcert_rig`) |
| **Arb-verified** | rigorous single-kernel run in Arb ball arithmetic (python-flint, or FLINT from C) | the refinement stage (`jobs/refine.py`); the mesh stage of pipeline B (`capcert_arb`) |
| **Krawczyk-verified** | existence and uniqueness of a zero by the Krawczyk test, single kernel (Arb) | the local and symmetric checks at the minimizer |
| **exact (kernel)** | exact rational/integer or algebraic-number computation in a computer-algebra kernel | resultants (sympy), Gröbner bases (msolve), root isolation (FLINT), arithmetic in Q(D) (sympy, FLINT) |
| **numerically identified** | integer-relation detection (LLL) at high precision, checked to many more digits, but not proved | minimal polynomials from `identify.py` for N ≥ 7 |
| **floating-point checked** | ordinary floating point with safety margins; evidence only | the `ladder_*` / `early_*` runs of the non-rigorous `capcert` |
| **proved** | a mathematical statement whose proof is complete; when computation is involved, the theorem says which labels its steps carry | "Theorem 10 (proved with interval-, Arb- and Krawczyk-verified computations)" |

Rules:
* A step that only *selects* something (axes, LP weights, splitting order) needs no
  verification. Only the inequality or identity that is finally checked does.
* Two runs of the same code on different machines, or with different settings, are a
  *re-run*, not a second kernel.
* In output strings and JSON keys use `verified` plus a `kernel` field; a cross-verified result
  gets `label: "cross-verified (K1 + K2)"`.
* Atlas levels (critconfig-atlas spec v0.2): verified and cross-verified interval results are
  level 2 (`interval-verified`); exact results accepted by an exact test are level 3
  (`exact (test-verified)`); the second kernel goes into a `cross_checks` list, not into the level.

## Status (2026-10-29)

| claim | step | label now | kernels |
|---|---|---|---|
| D_N ≥ T outside the boxes | pipeline A: mesh `capcert_rig` (IEEE intervals, C) + refinement `refine.py` (Arb) | interval-verified + Arb-verified | |
| | pipeline B: mesh `capcert_arb` (Arb, C) + refinement `xcheck_refine.py` (mpmath.iv) | N = 5, 6: **cross-verified** (pipelines A and B agree; `certificates/pipelineB_N5_local.json`, `certificates/pipelineB_N6.json`: 52,698,474 cells, 446,327 survivors refined into 483,847 cells, 0 failed) | IEEE intervals + Arb; Arb + mpmath.iv |
| minimizer unique in its box, D(q*) = D* | `local_cert.py` / `xcheck_local.py` | **cross-verified (Arb + mpmath.iv)** | independent code (`certificates/xcheck_local_*.json`) |
| minimizer symmetric, (q*, D*) = (q(Y(s*)), D_s) | `sym_cert.py` / `xcheck_sym.py` | **cross-verified (Arb + mpmath.iv)** | independent code (`certificates/xcheck_sym_*.json`); the orbit structure is exact combinatorics |
| N·D* is a root of P_N | elimination | **cross-verified (sympy + msolve)** | sympy resultants of the geometric orbit equations (`sym_cert.py`) + msolve Gröbner elimination of the paper's equations `msolve/*.ms` (`xcheck_elim.py`, `certificates/xcheck_elim.json`) |
| P_N irreducible, one root in the interval | root isolation | **cross-verified (FLINT + sympy)** | `certificates/xcheck_roots.json` |
| Gram matrix of the minimizer = G_e, entries in Q(D*) (N = 5, 6) | `gram_verify.py` | **cross-verified (exact: sympy + FLINT; intervals: Arb + mpmath.iv)**; atlas level 3 | `certificates/gramcert_N{5,6}_best.json` |
| exact Gram matrices of the other critical configurations; minimal polynomials for N ≥ 7 | `gram_exact.py`, `identify.py` | numerically identified | — |
| D_N ≥ D_best − η (ladder) | `capcert` (float) | floating-point checked | superseded by the proof for N = 5, 6 |

**The exact Gram matrices** (`jobs/gram_verify.py`). The entries in `sphcap/data/gram_exact.json`
were found by LLL. They are now proved to be the Gram matrix of the minimizer: (E1) the minimal
polynomial of D is P_N(N x) up to a constant, so its root in the theorem interval is D*; (E2) the
matrix has rank 3, exactly in Q(D*); (E3) every worst cap of the local verification satisfies its
equation φ_b = D* exactly (in squared form, in Q(D*)); (I1–I3) the leading 3×3 block is positive
definite, the signs of the squared equations are right, and the realization in the gauge lies in
the box B of the local verification. By the orthant test there, the minimizer is the only point of
B where all these caps take the value D*. A mutation test (one coefficient changed by 1; box B
shrunk to 10⁻¹⁹) makes the script fail, as it should.

Pipeline B (second kernels for every step) is run by `run_xcheck.sh`; `jobs/xcheck_verdict.py` states
per N whether it proves the theorem on its own (`certificates/xcheck_verdict.json`). Its code was
audited independently; the audit's findings (outward box B, D* < T and N·D* in the theorem interval
checked in B, exact symmetry group and orbits of all worst caps checked in B, exact Krawczyk box
hand-off, stale-file and failure handling) are fixed.

The theorem for N = 5, 6 is **proved**, with interval-verified, Arb-verified and Krawczyk-verified
steps, and every step is cross-verified for N = 5 and N = 6: pipeline B proves the theorem on its own
(`certificates/xcheck_verdict.json`).
