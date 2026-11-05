# Atlas data spec v0.2 schemas used by the tests

Copied from the companion files of the atlas data specification, draft v0.2 of 2026-10-27 (not adopted):
`observation.schema.json` and `common.schema.json` verbatim; `entity.schema.json` is an excerpt with only
`$defs/tolerances`, the one definition `observation.schema.json` refers to. `tests/test_atlas_pipeline.py` validates
every line of `export/sphere/d3/cap_discrepancy_sup/n<N>/observations.jsonl` against them. Replace these files when
the spec is adopted or revised.
