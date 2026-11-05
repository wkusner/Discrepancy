# Decisions: the atlas pipeline of the cap-discrepancy export (2026-11-04)

The atlas data specification v0.2 is a draft, not adopted. Where the pipeline (`jobs/atlas_pipeline.py`) had to
extend it or choose between options, the choice and the reason are written here. None of them settles an open
question of the spec (its section 10.3); they are choices for this export, to be revisited when the spec is adopted.

## Observation log (spec section 7.1, `observation.schema.json`)

1. **Location.** The log is `export/sphere/d3/cap_discrepancy_sup/n<N>/observations.jsonl`, next to the export,
   not `log/observations/<container>-d<d>-n<N>.jsonl` of spec section 1.1. Reason: this repository holds the
   source data of the atlas, not the atlas; the path carries the functional; the atlas importer can copy the lines.
2. **Functional.** The schema has no field for the functional (the spec was written for the kissing problem). It is
   put in `container.params` as `{"functional": "cap_discrepancy_sup"}`, which the schema allows. Open: whether the
   spec gets a `functional` field (as the mapping report proposed) or the functional stays in the path and params.
3. **`source.file_name`.** The pattern of the schema (`^[^/]+$|^(results|data|scripts)/[^ ]+$`) does not admit
   `sphcap/data/search/search_N21.json`. The line stores the bare file name (allowed for run output, which these
   files are: copies of `results/search_N<N>.json` of the search) and the repository path in
   `ingested_by.inputs`. Proposal for the spec: admit any repository-relative path without spaces.
4. **`source.format`.** `other` for the search files, `critical_configurations.json` and `atlas_N5_levels.json`:
   the enum lists kisscrit formats only.
5. **`source.commit`** is the commit that last changed the input file (`git log -1 -- file`), or `unknown` when the
   file is modified or outside the repository. Reason: the commit at which the file was read must not change when
   the export itself is committed, or a re-run would not be idempotent. The capdisc `observations.json` uses the
   same value in `source.commit`; before, it was the commit of the working tree at export time (e.g.
   `6f8a45d+dirty`). This is the one value of the earlier export that changes in every record.
6. **Tolerances.** The schema's slots have names from the kissing problem. Mapping: `contact` = incidence of a point
   on a cap circle (1e-9, `atlas_export.TOL`); `stress_support` = support threshold of the weight space (1e-9);
   `eigenvalue` = relative threshold of the second-order counts (1e-5). No `symmetry` slot is filled.
7. **`kind`.** `find` for every record, the curated classes of `critical_configurations.json` included. The level-2
   and level-3 descriptions from the certificates are attached when the export is rebuilt; they are not written as
   `import` observations yet (open: whether the atlas wants them as observations of their own).
8. **`record_index`** is the class name for `critical_configurations.json` (a string, which the schema allows) and
   the list position for the other files. `obs_id` follows the spec: `o-` + 16 hex of SHA-256 of the canonical
   JSON of `{file_sha256, record_index, raw}`.
9. **Search pass provenance.** `prov` in the schema has `seed` and `run_id` but no rounds, workers or runs. The line
   gets `prov = {run_id: "search_N<N>-pass<k>", seed}` (seed only when known) and a sentence in
   `ingested_by.note`; the structured record is `passes.jsonl` (one line per pass). Proposal for the spec: `prov`
   fields `rounds`, `runs`, `pass`. No machine names are stored anywhere.
10. **No time stamps** (`utc`, `first_found_utc` are left out): the search files do not record them, and a run time
    would break idempotency.
11. **No entity layer.** The pipeline writes observations only, no events and no keys; the export's `key_basis`
    gives the graph hash and the atlas mints ordinals (as before).

## Search passes, hits and convergence

12. **Provenance of the files already in the repository** is taken from the commit messages and scripts that added
    them, keyed by the file's SHA-256 (`KNOWN_PASSES` in `jobs/atlas_pipeline.py`), so that one command rebuilds the
    export from scratch. Fields not recorded there are null: the seed of N = 21–25, whether the N = 9–12 run
    resumed an older `.pkl`. A new file needs `--pass-info`; without it the pass is stored with null fields and the
    note "not recorded".
13. **Hits.** `search_parallel.py` resumes `results/search_N<N>.pkl` and adds hits, so a resumed pass reports
    cumulative hits. `search.hits` is stored as in the file; `hits_in_pass` is the difference to the observation of
    the same configuration in the previous pass, and null when that pass did not export the configuration.
14. **Same configuration** (to link a later observation to an earlier one): value to 1e-8 and the sorted pairwise
    inner products to 1e-6, the rule `atlas_stuck.py` already used for exits. This is a level-0 link, not the merge
    test of spec section 7.3 (alignment under O(3) × S_N), and it merges nothing: both observations stay. Spec open
    question 5 (level-0 merge rule) is not decided here.
15. **Names.** Pass 1 keeps the names of the earlier export (`search_N21#3`); a later pass is named
    `search_N21.pass2#3`, so names stay unique and old links (`same_as`) stay valid.
16. **Convergence note** on the best configuration (lowest value; all its observations): when its total hits are 1,
    or the pass that last found it was still improving in its last round (from `--search-log`, or recorded by hand
    from the commit message for N = 23), or it was first found by a later pass. The note follows the best: when a
    later pass finds a better configuration, the note is on the new best and no longer on the old one (the old
    observation is otherwise unchanged).

## Export additions

17. **Cap geometry.** Each hyperedge of the active hypergraph gets `cap = {center, height, angular_radius,
    angular_radius_deg}` of the over-full closed cap {y : ⟨y, center⟩ ≥ height} (for a `low` branch: the complement
    of the open cap, center −w, height −t). Both height and angular radius are given, so the viewer can use either.
18. **Gram matrix.** `gram_numerical` (level 0, N×N doubles, X Xᵀ of the stored X) on every observation. The proved
    minimizers keep their level-3 `gram` (by reference); the two are separate fields because the earlier `gram`
    field is a single object, not a list of descriptions.
19. **N = 7, 8.** `value` of N7_best and N8_best gets a level-2 description: the enclosure of the value of the local
    minimum from `certificates/localcert_N{7,8}_best.json`, noted "verified (Krawczyk, Arb) ... a local statement, no
    global claim". The active hypergraph of these two stays level 0, although the local verification also checks
    frozen counts and the other caps on the box (as it does for N = 5, 6, where the active level is 2): there is no
    second kernel for N = 7, 8, and the upgrade is left to the maintainer (open question).

## Reproducibility

20. **Idempotency.** Every output is written in a fixed order with fixed formatting (the log in the canonical JSON
    of spec section 1.5), the manifest has no time stamps, and the code commit in the manifest is the last commit
    that changed a pipeline file (`unknown` if one is modified), not the commit of the working tree.
21. **Stuck-point cache.** `stuck.jsonl` is keyed by the SHA-256 of the stored coordinates and a classifier version
    (SHA-256 of `sphcap/stuck.py`, `sphcap/balance.py`, `BLOCK_VERSION`). A change of the classifier recomputes all
    classes. The cache is compacted (current version, sorted) at the end of a complete run, so its bytes are
    deterministic although the workers finish in any order.
22. **`jobs/atlas_export.py` run directly** refuses to rewrite an export that has a log (it would drop the records
    of earlier passes); `--force` overrides.
