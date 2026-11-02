# KuaiRec train/validation/test protocol

The KuaiRec workflow uses independent training, validation and test files.

## Prepare data

```bash
python prepare_gr_data.py \
  --raw-dir /path/to/kuairec --output-dir ./input/kuairec \
  --validation-fraction 0.1 --seed 2024
```

The preprocessing command writes `train_data.npy`, `val_data.npy`, `test_data.npy`
and `splits.json`:

- The complete `small_matrix` is retained for held-out testing.
- Every `big_matrix` row whose `(user_id, item_id)` occurs in the small matrix is
  excluded before splitting.
- The remaining unique user/item pairs are sorted and permuted with the fixed
  seed. Ten percent of these groups form validation; the rest form training.
  All repeated observations of a pair stay in its group. This is a random
  interaction-group split, not a temporal split or a user-disjoint split.
- Duration and target-normalization statistics fit training rows only. GR
  vocabulary and feature embedding bounds, D2Q quantiles, and TPM bins/dense
  statistics also fit training rows only. Static item/user metadata is supplied
  independently of interaction targets. GR maps unseen categorical values to
  its reserved unknown embedding.
- `splits.json` records the seed, validation fraction, schema, input/output hashes,
  row counts and the number of overlapping big-matrix rows removed.

The input consists of `big_matrix.csv` and `small_matrix.csv`, or processed CSV
files containing raw watch-ratio and duration columns. Normalization statistics
are fitted on training rows. Cached arrays are reused when the manifest, options,
inputs and output hashes match.

The 53-column arrays use column 48 for play duration in seconds, column 49 for
video duration in seconds, and column 50 for raw watch ratio.

## Search and final evaluation

Install the template dependencies and authenticate the Codex executable outside
the repository, then run:

```bash
pip install -e '.[templates]'
bash gr.sh run \
  --train-data ./input/kuairec/train_data.npy \
  --val-data ./input/kuairec/val_data.npy \
  --test-data ./input/kuairec/test_data.npy \
  --output ./runs/kuairec-gr \
  --gpus 0,1,2,3 --budget 43200 --cold-start gr \
  --llm-model gpt-6.1-sol --llm-effort medium
```

Use `check` to validate inputs without starting a run. Use `preflight` with a
separate new output directory to run the baseline and four real LLM/GPU mutations
without evaluating test data. Four GPU slots and 64 available CPU cores are required;
`--cpu-cores` supplies explicit core IDs. The Python executable that launches this
entrypoint must also have the template dependencies installed.

Hypothesis generation and code implementation use the same Codex model, reasoning
effort and transport as the Amazon entrypoint. `--print-llm-config` displays effective
settings without starting a run. See
[LLM configuration by entrypoint](reproduction.md#llm-configuration-by-entrypoint).
The host applies structured edits to isolated candidate source files, executes the
native KuaiRec routing/promotion/state tools and charges every training attempt to
the shared budget. LLM time and final test inference are accounted for separately.
The default trial reference is 6400 seconds and the trial ceiling is 9600 seconds;
select overrides with `--trial-secs` and `--trial-timeout`.

Other registered templates are `d2q`, `ks_d2q` (the D2Q alias) and `tpm`. Every
invocation requires explicit training, validation and test paths and a new output
directory. The input checks reject missing validation, aliased/copied files,
malformed arrays and user/item pair overlap across the three splits. The held-out
path is omitted from LLM context and search-worker variables. During search,
`GR_TEST_DATA` contains the validation path. The held-out test path is provided for
final evaluation.

Each template saves `gr_checkpoint.pt` at the epoch with highest validation
WT-XAUC, breaking ties by lower WT-MAE. The host retains its exact source, model
support files and weights before cleaning an isolated trial slot. Promotion checks
that the selected source matches the trained artifact. Artifacts live under
`<output>/recharness/native/logs/kuairec_artifacts/`, with selection in
`kuairec_selection.json` in the same logs directory.

After search, `evaluate_final_incumbent` verifies those hashes and runs the retained
program in `GR_RUN_MODE=evaluate`, loading frozen weights and training-derived
statistics. It does not train, refit statistics or select epochs on test data. An
exclusive attempt marker permits one held-out evaluation attempt and closes the
search/update/promotion tools. A changed checkpoint or missing trained artifact is
rejected before testing. The host runs final evaluation after search finishes and
includes the held-out report in its summary.

The isolation is a protocol for trusted local code, not an OS filesystem sandbox.
Checkpoint loading uses trusted, hashed local model objects. Do not load checkpoints
from an untrusted external source.

## Regression checks

The default `dev` suite covers reproducible grouping, pairwise disjointness,
training-only normalization, the column schema, cache verification, validation/test
separation, artifact retention/promotion and test-once enforcement. With PyTorch
installed, `tests/test_kuairec_templates.py` additionally trains tiny CPU versions
of all three templates, deletes the training/validation files, then verifies frozen
test inference without changing the checkpoint. No provider calls are needed.
