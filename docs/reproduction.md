# Reproduction

## Environment

Use Linux, Python 3.10+ (3.11+ recommended), four CUDA GPU slots and 64 CPU cores for the registered
experiment. Install the source checkout with `pip install -e '.[templates,dev]'`.
Install and authenticate the LLM command-line executable separately; never store
credentials in this tree. The standalone runner defaults to `gpt-6.1-sol` with
`medium` effort. Select the model and effort with `--llm-model` and `--llm-effort`.
The shell entrypoints use the same Codex transport and defaults.

Dependencies are specified in `pyproject.toml`. Python, CUDA and provider runtimes
are installed separately.

## LLM configuration by entrypoint

| Entrypoint | Workload | Transport | Model | Reasoning effort |
| --- | --- | --- | --- | --- |
| `python -m recharness`, `bash run2.sh`, `recharness` | Amazon; SASRec, GRU4Rec, BERT4Rec, NextItNet, HSTU | Codex CLI | `gpt-6.1-sol` | `medium` |
| `python -m recharness.kuai_cli`, `bash gr.sh`, `recharness-kuai` | KuaiRec; GR, D2Q, TPM | Codex CLI | `gpt-6.1-sol` | `medium` |

Every entrypoint uses one configuration for both hypothesis generation and code
implementation. `--llm-model`, `--llm-effort` and `--llm-timeout` (default 1200
seconds per attempt) apply to both workloads. No temperature override is supplied.
Model settings are explicit CLI arguments; provider-specific environment variables
do not override them. Authenticate the Codex executable separately.

Inspect effective settings without data files, GPU allocation or an LLM call:

```bash
python -m recharness --print-llm-config
bash run2.sh --print-llm-config
bash gr.sh --print-llm-config
bash gr.sh --llm-model gpt-6.1-sol --llm-effort medium --print-llm-config
```

Each run saves the settings in `config.json` and records model/effort in per-call
receipts. Calls share the same concurrency locks, bounded retries, structured-response
validation and request cache within that run. Training workers do not receive LLM
credentials. The shell wrappers accept the same arguments as their Python modules;
`RECHARNESS_PYTHON` can select the Python executable used by either wrapper.

## Amazon Reviews data

Supply the Amazon Reviews 2023 benchmark splits locally:

```text
<raw-data>/
  benchmark/5core/last_out/
    <category>.train.csv
    <category>.valid.csv
    <category>.test.csv
```

The categories are `Movies_and_TV`, `Industrial_and_Scientific`, `Electronics` and
`CDs_and_Vinyl`. Obtain the public dataset under its own terms; raw user histories
are not distributed in this artifact.

```bash
for category in Movies_and_TV Industrial_and_Scientific Electronics CDs_and_Vinyl; do
  python -m gagc.data_preprocess \
    --dataset "$category" --source 2023 --local-dir /path/to/raw-data \
    --data_dir ./input/trainval --test_dir ./input/test --seed 0
done
python -m gagc.compute_dataset_stats \
  --data_dir ./input/trainval --test_dir ./input/test --output_dir ./input/trainval
```

The prepared directory contains:

```text
input/
  trainval/
    <category>_train.txt
    <category>_valid.txt
    dataset_stats.json
  test/
    <category>_test.txt
```

The baseline uses fresh weights, training seed 42, 200 epochs and 200 batches per
epoch. Validation evaluates at most 200 users per category, held-out test evaluates
at most 10,000, with 99 unique negative candidates and evaluation seed 42. The
primary score is macro validation HR@10. See the benchmark code for short-history
and candidate availability handling.

## Execution

Follow the `check`, `preflight`, and `run` examples in the README. Use a separate new
output directory per family, repetition, preflight and ablation. The runtime refuses
to silently resume or overwrite an existing directory. If interrupted, preserve its
ledger and workers for inspection. Archive restoration does not authorize automatic
experiment continuation.

The default total budget is 43,200 summed worker seconds, per-trial upper limit
16,200 seconds, and minimum dispatch 300 seconds. Jump caps can be smaller than the
per-trial upper limit. Record explicit overrides and report the same overall budget
for comparisons. `--cpu-cores` orders the four 16-core CPU slots independently of
the physical GPU IDs, avoiding machine-specific numbering assumptions.

## Outputs

```text
runs/<experiment>/
  config.json                 # actual paths, resources, model and protocol
  frozen.json                 # code and prepared-split fingerprints
  status.json
  recharness/
    budget.json               # reservations, settlements and total consumption
    events.jsonl
    calls/                    # requests, structured responses and failures
    trials/                   # source, checkpoints, worker logs and receipts
    native/
      thompson_state.json
      execution/
      memory/                 # observations, programs, events, contexts
      final-selection.json
    single-model-selection.json
    selection.json
    final-test/
    summary.json
```

Runtime records can contain local paths and provider metadata. They are not included
in the review ZIP. Audit any additional experiment evidence before distributing it.

## KuaiRec entrypoint

`bash gr.sh --help` describes the KuaiRec check/preflight/run commands. Preprocessing
uses `prepare_gr_data.py` and writes independent train, validation and test arrays.
See the [KuaiRec protocol](kuairec.md) for partitioning and required inputs.
Both tasks use the same LLM defaults; their data, metrics and search-arm definitions
follow their respective benchmark protocols.
