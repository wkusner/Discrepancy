# Trusted base of the N = 5, 6 proof

What a reader has to believe, besides the hand-proved lemmas listed in [PROOF.md](PROOF.md) §4, for the
computer-assisted steps to prove what they claim. The versions are those of the reference runs; every
certificate records the versions it was produced with (`stamp.versions`).

## Pipeline A

| component | used for | trusted for | version (reference run) |
|---|---|---|---|
| IEEE 754 binary64 hardware, round-to-nearest | mesh `capcert_rig` (`csrc/capcert.c` built with `-DRIGOROUS`, `csrc/ia.h`, `csrc/rig.h`, `csrc/fo.h`) | correct rounding of +, −, ×, ÷, √ | x86-64 |
| C compiler, flags `-O3 -march=native -fopenmp -DRIGOROUS -ffp-contract=off -fno-fast-math` | building `capcert_rig` | compiling the source as written: no contraction into FMA, no value-changing optimization | gcc 13.3 (`__VERSION__` and the flags are in every mesh stamp; the chain checker requires the three proof flags) |
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
* Every certificate carries a stamp: step, commit, the tree state of the proof code (`tree_dirty`, over
  `proof/`, `csrc/`, `verify.sh`, `build.sh`, untracked files included; `tree_checked` names those paths), versions,
  precision, command line, machine type, date, run time, and the SHA-256 of its inputs (inputs outside the
  repository are recorded by file name only).
* Every mesh stamp (`mesh.done`, `mesh2.done`) carries the SHA-256 of each survivor file and the provenance of the
  binary: commit, tree state of `csrc/` and `build.sh` at build time, SHA-256 of the C sources, `__VERSION__`, flags
  (compiled in by `build.sh`). The chain checker recomputes the source hash from the recorded commit and requires the
  commit's proof code to be that of the Python certificates (C8). A second pass records the SHA-256 of the first
  pass's stamp and of every input file.
* Proof mode starts clean: `verify.sh` refuses to run when the proof code (or `check/`) differs from the commit,
  rebuilds `capcert_rig`, checks that the binary is from this commit, and deletes its output directory; nothing is
  resumed. (`capcert_rig` itself can resume an interrupted pass; the stamp then records `resumed=` and the hashes
  cover everything in the files, so a resumed run is still checkable. Duplicated cells only cost time.)
* The rigorous build refuses `--keep-prob`, `--max-cells-per-root` and `--balls`; the stamps record the keep
  fraction, the exclusion balls and the cell cap actually used.

## Bookkeeping (what binds the certificates together)

Not part of any bound, but part of what a reader relies on when the chain checker says "chain complete": SHA-256
(Python's `hashlib` in the checker and the proof scripts; `csrc/sha256.h` in the mesh, tested against `hashlib`),
and `git` (the commits recorded in the stamps, the diff of the proof code between them, the C sources at a commit).
A wrong hash or a failing `git` call makes the check fail; it cannot make a false bound pass. The checker needs the
history of every commit it compares (a shallow clone fails C8 with a message saying so).

Stamps written before 2026-11-04 (the N = 7 run started at `768cefc`) have no survivor hashes and no mesh
provenance, and their `tree_dirty` is not evidence (the old pathspecs matched nothing). The checker accepts them only
with a `<P>.surv.sha256` sidecar that names the stamp's own SHA-256 (PROOF.md Section 3a), and records that route.

## Size of the trusted code

```
proof/*.py                    ~1,100 lines   (Steps 1, 3–6, bookkeeping)
csrc/capcert.c (rigorous path) + ia.h + rig.h   ~800 lines   (Step 2; includes the stamps)
csrc/fo.h (first-order bound in the mesh)      ~230 lines   (Step 2, with --first-order; tests/test_capcert_fo.py)
csrc/sha256.h                                  ~80 lines    (survivor hashes; bookkeeping, not a bound)
check/check_chain.py                          ~400 lines   (chain checker; standard library and git only)
```

(`csrc/capcert.c` still contains the floating-point exploration mode behind `#ifndef RIGOROUS`; splitting it
into its own file is on the refactor list.)
