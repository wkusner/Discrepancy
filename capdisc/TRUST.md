# Trusted base of the N = 5, 6 proof

What a reader has to believe, besides the hand-proved lemmas listed in [PROOF.md](PROOF.md) §4, for the
computer-assisted steps to prove what they claim. The versions are those of the reference runs; every
certificate records the versions it was produced with (`stamp.versions`).

## Pipeline A

| component | used for | trusted for | version (reference run) |
|---|---|---|---|
| IEEE 754 binary64 hardware, round-to-nearest | mesh `capcert_rig` (`csrc/capcert.c` built with `-DRIGOROUS`, `csrc/ia.h`, `csrc/rig.h`) | correct rounding of +, −, ×, ÷, √ | x86-64 |
| C compiler, flags `-O3 -march=native -fopenmp -DRIGOROUS -ffp-contract=off -fno-fast-math` | building `capcert_rig` | compiling the source as written: no contraction into FMA, no value-changing optimization | gcc 13.3 |
| python-flint / FLINT (Arb) | Steps 3–6 (`proof/`) | ball arithmetic: every operation returns a ball containing the exact result; comparisons are true only when certain; `sqrt`, `sin`, `cos`, `acos`, `atan2` enclose the exact value | python-flint 0.9.0, FLINT 3.6.0 |
| sympy | Step 5 (resultants, factorization of the eliminant), Step 7 | exact rational and polynomial arithmetic; real-root counting | 1.14.0 |
| CPython, numpy | glue; reading survivor files (`numpy.fromfile`, binary64) | moving doubles without changing them | CPython 3.13, numpy 2.5 |

The mesh does **not** trust `libm`: sin and cos are Taylor polynomials with a Lagrange remainder in
interval arithmetic (`csrc/rig.h`). Arb's transcendental functions *are* trusted (Steps 3–6).

## Pipeline B (second kernels)

| component | used for | version |
|---|---|---|
| FLINT/Arb from C | mesh `capcert_arb` (`csrc/capcert_arb.c`) | FLINT 3.0.1 (static build in `bin/`) or the system FLINT |
| mpmath `iv` | refinement, local and symmetric Krawczyk tests (`jobs/xcheck_*.py`) | 1.3.0 |
| sympy | root isolation (`jobs/xcheck_roots.py`) | 1.14.0 |
| msolve | Gröbner elimination (`jobs/xcheck_elim.py`): a *check*, not a proof (probabilistic rational reconstruction) | version recorded in the run log (to be stamped) |

Pipeline B shares Arb with pipeline A in one stage (its mesh), and shares sympy in the exact steps.

## Not trusted

Everything that only *chooses*: the search (`sphcap/`, `explore/`), the hints in `proof/hints/` (images,
worst caps, square subsets), floating-point Newton steps and approximate inverses in the Krawczyk tests,
scipy's LP solver (weights λ and μ), splitting order, axes. A bad choice makes a check fail; it cannot make a
false check pass. Floating-point output is never used as a bound.

## Rules for the trusted core (`proof/`)

* `proof/` imports nothing from `sphcap/` or `explore/`.
* Every inequality used in PROOF.md is decided in Arb (or in the IEEE interval arithmetic of the mesh).
  Doubles enter only as exact values (inputs, cell centers and half-widths, chosen weights).
* Doubles written to certificates are rounded outward (`certio.float_up` / `float_down`), and also stored as
  hex floats; Arb balls are stored as Arb strings, which read back outward.
* Every certificate carries a stamp: step, commit, versions, precision, command line, machine type, date,
  run time, and the SHA-256 of its inputs.
* Proof mode starts clean (`verify.sh` deletes its output directory first); nothing is resumed.

## Size of the trusted code

```
proof/*.py                    ~1,050 lines   (Steps 1, 3–6, bookkeeping)
csrc/capcert.c (rigorous path) + ia.h + rig.h   ~700 lines   (Step 2)
```

(`csrc/capcert.c` still contains the floating-point exploration mode behind `#ifndef RIGOROUS`; splitting it
into its own file is on the refactor list.)
