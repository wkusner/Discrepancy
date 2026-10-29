# Verification vocabulary and status

This project uses the same convention as the Twelve Spheres work. **A computational result is
called *certified* only after two independent kernels have checked it.** A rigorous computation
done in a single kernel is called *verified*, and the label names the method. Paper text, code
output, `STATUS.md`, JSON files and commit messages all follow this file.

## Vocabulary

| label | meaning | example |
|---|---|---|
| **certified** | rigorous, and confirmed by a second, independent kernel (different arithmetic library and different code path) | the local verification at the minimizer (Arb + mpmath.iv) |
| **interval-verified** | rigorous single-kernel run in our IEEE interval arithmetic (`csrc/ia.h`, `csrc/rig.h`): one-ulp outward rounding, Taylor sin/cos with remainder | the mesh stage (`capcert_rig`) |
| **Arb-verified** | rigorous single-kernel run in Arb ball arithmetic (python-flint) | the refinement stage (`jobs/refine.py`) |
| **Krawczyk-verified** | existence and uniqueness of a zero by the Krawczyk test, single kernel (Arb) | the local and symmetric checks at the minimizer |
| **exact (kernel)** | exact rational/integer computation in a computer-algebra kernel | resultants (sympy), Gröbner bases (msolve), root isolation (FLINT) |
| **numerically identified** | integer-relation detection (LLL) at high precision, checked to many more digits, but not proved | exact Gram matrices in `sphcap/data/gram_exact.json`; minimal polynomials from `identify.py` |
| **floating-point checked** | ordinary floating point with safety margins; evidence only | the `ladder_*` / `early_*` runs of the non-rigorous `capcert` |
| **proved** | a mathematical statement whose proof is complete; when computation is involved, the theorem says which labels its steps carry | "Theorem 10 (proved with interval- and Arb-verified computations)" |

Rules:
* A step that only *selects* something (axes, LP weights, splitting order) needs no
  verification. Only the inequality or identity that is finally checked does.
* Two runs of the same code on different machines, or with different settings, are a
  *re-run*, not a second kernel.
* Use the words above literally in output strings and JSON keys: `verified` plus a `kernel`
  field, and `certified` only for cross-checked results.

## Status (2026-10-29, updated)

| claim | step | label now | kernels |
|---|---|---|---|
| D_N ≥ T outside the boxes, N = 5, 6 | mesh stage `capcert_rig` | interval-verified | ours (IEEE, C). Second kernel planned: mesh re-implemented on Arb's C library (FLINT), about one night on w227 for N = 6 |
| | refinement `refine.py` | Arb-verified | Arb. Second kernel planned: numpy IEEE outward-rounded intervals |
| minimizer unique in its box, D(q*) = D* | `local_cert.py` / `xcheck_local.py` | **certified** | Arb (python-flint) + mpmath.iv, independent code (`certificates/xcheck_local_*.json`) |
| minimizer symmetric, (q*, D*) = (q(Y(s*)), D_s) | `sym_cert.py` / `xcheck_sym.py` | **certified** | Arb + mpmath.iv, independent code (`certificates/xcheck_sym_*.json`); the orbit structure is exact combinatorics |
| N·D* is a root of P_N | elimination | **certified** | sympy resultants of the geometric orbit equations (`sym_cert.py`) + msolve Gröbner elimination of the paper's equations `msolve/*.ms` (`xcheck_elim.py`, `certificates/xcheck_elim.json`) |
| P_N irreducible, one root in the interval | root isolation | **certified** | FLINT + sympy (`certificates/xcheck_roots.json`) |
| exact Gram matrices N = 5, 6 | `gram_exact.py` | numerically identified | — |
| D_N ≥ D_best − η (ladder) | `capcert` (float) | floating-point checked | superseded by the proof |

The theorem for N = 5, 6 is **proved**, with interval-verified, Arb-verified and certified steps.
It becomes *certified* once the mesh and refinement rows have their second kernels.
