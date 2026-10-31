# Artifact validation

The default suite was installed into a fresh Python 3.10.12 environment using the
`dev` extra. It passed 119 tests, skipped the optional PyTorch test module and had
four inherited compatibility checks marked as expected failures.

A full CPU run with the available PyTorch 2.5.1 runtime passed 122 tests with the
same four expected failures. It includes real tiny-model training and frozen-test
inference for GR, D2Q and TPM. Those tests retain the candidate artifact, delete its
original workspace and training/validation files, then evaluate the retained weights
and verify that the checkpoint does not change. No GPU or provider calls are used.

Additional checks:

- Python syntax compilation, shell syntax checks, CLI help and input-fingerprint
  checks passed.
- Source distribution and wheel builds succeeded.
- All five registered Amazon cold-start source fingerprints were verified.
- KuaiRec tests cover deterministic interaction-group partitioning, test-overlap
  removal, training-only statistics, array column ordering, manifest/cache checks,
  required independent validation and interaction-overlap rejection.
- Controller checks cover checkpoint retention before trial cleanup, matching
  source/weight promotion, rejecting changed artifacts, a single final-test attempt
  and closing search after test starts.
- The review-file scan found no direct personal identifiers, host home paths,
  personal repository/preprint links or recognized credential patterns.
- The review ZIP passed its integrity check and uses a neutral root, normalized
  timestamps and no ZIP comment or Git metadata.

The four expected failures are retained historical assertions about retired textual
strategy-arm routing and an earlier KuaiRec jump expectation. They have explicit
`xfail` markers in the compatibility tests. The current six-arm, jump and evidence
memory regression tests pass, including synthetic end-to-end ledger/rollback flow.

These checks do not rerun paper-scale experiments, verify reported effectiveness or
certify a real provider/GPU preflight. Regenerate older KuaiRec arrays and rerun its
experiments under the documented three-way protocol before reporting those scores.
