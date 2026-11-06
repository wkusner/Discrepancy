# designs: spherical t-designs on S^2 for the atlas

A new functional for the atlas, `spherical_design` (spec v0.2 layout, same record shape as
`export/sphere/d3/cap_discrepancy_sup`).  A spherical t-design is an N-point set X with
(1/N) sum p(x_i) = mean of p over the sphere for every polynomial p of degree <= t.  For fixed N the
interesting number is the largest t (Hardin-Sloane's tau(N)).

Status of this folder: **local working copy, not committed, not pushed.**

## Contents

| file | what |
|---|---|
| `sphdesign.py` | moment residuals (exact sphere means), rotation groups C_n, D_n, T, O, I, symmetric-stratum search, mpmath refinement |
| `run_batch.py` | for each N: search at the listed strength tau(N), refine one solution at 40 digits (N <= 24), also try tau+1, write atlas records |
| `verify_records.py` | independent re-computation of every exported record (double precision and 60-digit residual from the stored strings) |
| `literature/hs_tau_table.json` | the Hardin-Sloane table, N = 1..100, attributed (transcription unverified, see below) |
| `literature/spherical_designs_sources.json` | source keys with a per-source verification status |
| `search_log.json` | per N: groups tried, found / not found at tau and tau+1, seconds |
| `../export/sphere/d3/spherical_design/n<N>/observations.json` | the records (66 records for N = 1..40) |

Dependencies: numpy, scipy, mpmath.  Run: `python designs/run_batch.py 1 40 --hpmax 24`, then
`python designs/verify_records.py`.

## Method

For a rotation group G, a choice of special orbits (points on rotation axes, no free parameters) and g generic
orbits (two parameters each) with |G| g + (sizes of special orbits) = N, minimize the vector of moment
residuals by random-start least squares.  Groups are tried from the largest down, so the recorded group is the
largest symmetry this search found (the set may have more).  Solutions are deduplicated by the sorted Gram
inner products.  One solution per N (N <= 24) is refined with Gauss-Newton at 40 digits.

## Labels (follow the project's rules)

* level 0, "numerical": max moment residual in double precision (all records).
* level 1, "high-precision": residual at 40 digits after refinement (N <= 24, first record only).
  An approximation of a design, **not a proof**.
* Nothing here is "verified", "cross-verified" or "certified".  `verify_records.py` is a re-computation.
* "not found" is a statement about this search only (strata, 12 starts, time budget).  It is not evidence of
  non-existence.

## Results (computed)

* Strength tau(N) from the table was found for every N in 1..40 except **N = 29 (t = 6)** and **N = 37 (t = 7)**,
  which need a larger search (their listed groups [2]+ and [3]+ have many generic orbits).
* A design of strength tau(N)+1 was found for no N in 1..40.  This agrees with the table's conjecture but is a
  limited search (see above).
* N = 9, t = 2 was found; t = 3 was not found (the open case in the survey, arXiv:2211.01890).
* Recovered symmetric designs include N = 12 (I), 20 (I), 16 (T), 24 (O, t = 7), 36 (T, t = 8), 40 (T, t = 8).

## Attribution and licensing policy

* **Coordinates are computed in this project.**  No coordinate file from the Hardin-Sloane site or any other
  source was downloaded or copied.  Each record says so under `provenance.coordinates`.
* **Existence and strength claims are attributed** to the literature: each record carries
  `provenance.literature_claim` (source key, tau(N), the proof code and the symmetry group listed in the table)
  and `provenance.equivalence_to_literature = "not compared"`.  Whether our set is congruent to the listed one is
  open until someone compares them (a Gram-signature comparison is easy once the source file is available;
  ask the table's authors for redistribution terms before adding their files).
* `literature/hs_tau_table.json` was read through an automated summary of
  https://neilsloane.com/sphdesigns/ on 2026-11-03.  **Transcription is unverified**: compare with the source
  before relying on any entry.  tau(N) is the authors' conjectured maximum except where a proof is indicated.
* In `spherical_designs_sources.json` a source is marked "confirmed" only if its title/arXiv number was seen in a
  search result; otherwise "from memory" (verify before citing).  The codes M90, B93, R95 still need full references.
* Suggested licence line for the records: data CC-BY-4.0, code MIT (as for the rest of the atlas); credit
  "Kusner, critconfig-atlas".  Maintainer to confirm.

## Next (not done)

* N = 41..100 (the table's t up to 13), with a better search for the strata with many generic orbits.
* Exact coordinates by symmetric strata and msolve: the square systems are N = 72 (I, t = 11, 2 equations),
  N = 36 (T, t = 8, 6 equations), N = 14 (D3, t = 4), N = 16 (T, t = 5), N = 24 (O, t = 7).
* Other families from the literature, added the same way (recompute, attribute, label): Tammes optima,
  sum-of-distances / energy minimizers, cap-discrepancy minimizers.
