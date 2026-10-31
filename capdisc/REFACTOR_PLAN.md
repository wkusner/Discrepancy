# Refactor plan: making the N = 5, 6 proof auditable

Goal: a reader who trusts only the arithmetic libraries in the trusted base can check the proof by
reading a small amount of code, running one command per N, and running an independent checker over
the certificates. [PROOF.md](PROOF.md) is the argument this code must carry.

## Principles

1. **A small trusted core, separated from everything else.** Only code that checks an inequality or
   identity used in PROOF.md lives in `proof/`. Search, atlas, identification, figures and the
   floating-point mesh live elsewhere and are not trusted. The core's line count is reported.
2. **No floating-point reasoning in the core.** Every inequality the proof uses is decided in ball or
   interval arithmetic. Floating point is used only to *choose* things (axes, weights, splits,
   approximate inverses). No `(1 + 1e-12)` slack factors, no Sterbenz arguments.
3. **Certificates are exact and self-describing.** Binary values are stored exactly (hex floats or
   outward-rounded Arb ball strings). Every certificate records: the claim it supports (the Step of
   PROOF.md), its inputs by hash, the commit, library versions, precision, compiler flags, machine,
   date, and run time.
4. **Proof mode runs clean.** No resume, no stale files, no partial outputs accepted. A run either
   writes a complete certificate or none.
5. **The chain is checked by a separate, short program** that does no numerics: it reads the
   certificates and checks that they fit together (Section "Chain checker").
6. **Each pipeline is a second kernel, not a re-run.** Pipeline B keeps its own code path for every
   step, including congruence.

## Proposed layout

```
capdisc/
  PROOF.md            the argument, step by step, with the file that checks each step
  TRUST.md            trusted base: libraries, versions, build flags; what is not trusted
  VERIFICATION.md     vocabulary (unchanged)
  proof/              TRUSTED CORE
    domain.py           Step 1: fundamental domain, images, cell-misses-domain test
    cellbound.py        Steps 2-3: zeroth-order and LP bounds on a cell (Arb)
    local.py            Step 4: Krawczyk + frozen counts + orthant test (Arb)
    symmetric.py        Step 5: symmetric Krawczyk test and elimination
    congruence.py       Step 6
    exact.py            Step 7: root isolation, Gram matrix in Q(D)
    refine.py           Step 3 driver
    mesh/               Step 2: capcert_rig (C, IEEE intervals): rig.h, ia.h, mesh.c
    certio.py           exact (de)serialization of balls and boxes; stamps
  proof_b/            SECOND KERNEL (mpmath.iv, Arb-in-C mesh), same step structure
  check/
    check_chain.py      chain checker (no numerics)
  verify_N5.sh, verify_N6.sh    one command each: build, run A and B, check chain, write a bundle
  tests/proof/        unit tests for the core (see below)
  explore/            NOT TRUSTED: search, atlas, identify, msolve generators, float capcert
  sphcap/             library for exploration (not imported by proof/)
```

`proof/` must not import from `sphcap/` or `explore/`. The C mesh is split so that the rigorous
build is its own file, with no `#ifdef RIGOROUS` branches.

## Chain checker

`check/check_chain.py certificates/` checks, without doing any numerics beyond comparing exact
rationals:

* every image listed by Step 1 has a Step 4, a Step 5 and a Step 6 certificate, for both pipelines;
* the T used by the mesh and the refinement equals the largest upper end of the Step 4 enclosures,
  and the exclusion boxes of the refinement are exactly the boxes B_k of Step 4 (by hash);
* the refinement read every survivor the mesh wrote (counts and hashes per root cell);
* the polynomial and isolating interval in Steps 5 and 7 agree with each other and with the values
  printed in the papers (`check/paper_values.json`);
* every certificate comes from the same commit, the same library versions, and proof mode;
* every `verified` flag is true, and `failed` counts are 0.

It writes `certificates/bundle_N{5,6}.json`: the claim, the list of certificates with hashes, and
the verdict.

## Tests

* interval kernel (`ia.h`, `rig.h`): outward rounding on random operands against MPFR; Taylor
  sin/cos enclosures against Arb;
* cell bound: rigorous bound ≤ exact discrepancy of random configurations in random cells; and the
  rigorous bound ≤ the floating-point bound plus a tolerance;
* Krawczyk: succeeds on known systems; fails on a perturbed system (mutation test);
* orthant test: fails when one worst cap is removed;
* congruence: fails when the second image is perturbed outside its box;
* chain checker: fails on each kind of mismatch (a deleted certificate, a changed T, a changed box,
  a different commit).

## Progress (2026-10-31)

Done on `audit-refactor`: steps 1, 2 and 3 below for pipeline A (TRUST.md, `proof/certio.py`, the core moved into
`proof/` and checked bit for bit against the old scripts, every inequality in Arb); `verify.sh`, the chain
checker, and the first tests (part of 5 and 6). Open: the reference runs (7), pipeline B (4), the rigorous mesh
as its own file, Step 7 in `proof/`, and the independent audit (8).

## Order of work

1. TRUST.md; `certio.py` (exact serialization) and stamps.
2. Move the trusted code into `proof/` without changing behavior; re-run N = 5 and compare
   certificates bit for bit with the current ones.
3. Replace the float-with-slack steps by Arb (PROOF.md §6, item 3); re-run N = 5.
4. Pipeline B congruence; `proof_b/` restructure.
5. Chain checker and `verify_N{5,6}.sh`; proof-mode guards (no resume).
6. Tests.
7. Full runs of both pipelines for N = 5 and N = 6 on the server, with stored certificates.
8. Independent audit of the refactored repository; then a tagged release and a Zenodo DOI.
