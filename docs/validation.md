# Artifact validation

The source was installed into a fresh Python 3.10.12 environment using the `dev`
extra, with no external project source trees on `PYTHONPATH`.

- 109 tests passed and four inherited compatibility checks were expected failures.
- Python syntax compilation, shell syntax checks, CLI help and input-fingerprint
  checks passed.
- Source distribution and wheel builds succeeded.
- All five registered cold-start source fingerprints were verified.
- The review-file scan found no direct personal identifiers, host home paths,
  personal repository/preprint links or recognized credential patterns.
- The review ZIP passed its integrity check and uses a neutral root, normalized
  timestamps and no ZIP comment or Git metadata.

The four expected failures are retained historical assertions about retired textual
strategy-arm routing and an earlier KuaiRec jump expectation. They have explicit
`xfail` markers in the compatibility tests. The current six-arm, jump and evidence
memory regression tests pass, including synthetic end-to-end ledger/rollback flow.

These checks use synthetic inputs and mocked LLM/training workers. They do not
rerun paper-scale GPU experiments, verify reported effectiveness or certify a real
provider/GPU preflight. Use the separate `preflight` command on the target hardware
before formal runs.
