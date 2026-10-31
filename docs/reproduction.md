# Reproduction

## Environment

Use Linux, Python 3.10+ (3.11+ recommended), four CUDA GPU slots and 64 CPU cores for the registered
experiment. Install the source checkout with `pip install -e '.[templates,dev]'`.
Install and authenticate the LLM command-line executable separately; never store
credentials in this tree. Provider choice is a transport setting. The recorded
standalone configuration uses `gpt-6.1-sol` with `medium` effort, while the earlier
compatibility scripts use a provider API. A replacement model is a different
experimental configuration and should be reported explicitly.

The package uses lower dependency bounds and does not claim a fully locked
historical environment. Record Python, CUDA, PyTorch, provider/CLI versions and
hardware alongside each new run. This artifact includes no vendored runtime.

## Amazon Reviews data

Supply the Amazon Reviews 2023 benchmark splits and category metadata locally:

```text
<raw-data>/
  benchmark/5core/last_out/
    <category>.train.csv
    <category>.valid.csv
    <category>.test.csv
  raw/meta_categories/
    meta_<category>.jsonl
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
    <category>_text_name_dict.json.gz
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

## Compatibility entrypoints

`bash run2.sh --help` and `bash gr.sh --help` describe the earlier Amazon and KuaiRec
entrypoints. KuaiRec preprocessing uses `prepare_gr_data.py`. These scripts require
the optional `legacy` dependencies and provider environment variables; their agent
workflow differs from the current standalone runner. Do not report the two execution
paths as an identical method configuration.
