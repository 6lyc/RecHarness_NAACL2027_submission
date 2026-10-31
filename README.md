# RecHarness

Anonymous code artifact for review. The primary implementation is the six-arm,
family-preserving RecHarness method with budgeted combination jumps and within-run
evidence memory. No author list, affiliations, paper link, experiment logs, data,
checkpoints, or prior search memory are included.

## Method

1. Train and evaluate the unmodified cold start using validation HR@10.
2. Use Thompson sampling over six directions: feature construction, embedding
   representation, architecture/interaction, sequence/interest, learning objective,
   and joint hyperparameter tuning. Select up to four distinct directions per group.
3. Ask the LLM for a hypothesis and structured source edits for each selected scope.
4. Evaluate isolated candidates and update the incumbent and posterior from validation.
5. When the stagnation/ceiling or reserved-exploration gate opens, test distinct
   combinations of two to four directions spanning at least two parent categories.
   Only the best valid jump enters a provisional basin. Reserve the jump and four
   local retuning groups together; accept an improved basin or restore the incumbent.
6. Retain immutable observations, exact programs, costs and jump lifecycle events.
   Retrieve bounded, relevant evidence from this independent run for subsequent edits.
7. Resolve unfinished jumps, verify the selected predictor/checkpoints and evaluate
   the held-out test set once after selection.

Details and defaults: [method](docs/method.md), [reproduction](docs/reproduction.md),
[jump configuration](configs/JUMP_POLICY.json), [memory design](configs/MEMORY_DESIGN.json).

## Install and check

Use Linux and Python 3.10+ (3.11+ recommended). A CUDA-enabled PyTorch environment is required for
training. The primary runtime calls an independently authenticated `codex`
executable on `PATH`; it accepts `--llm-model` and `--llm-effort` explicitly.
The experiment default is `gpt-6.1-sol` with `medium` reasoning effort.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[templates,dev]'
python -m recharness --help
python -m pytest -q
```

No real GPU training or LLM calls are made by the test suite. Packaging validation
and the inherited compatibility checks are documented in [validation](docs/validation.md).

## Main experiment

Prepare the four Amazon Reviews categories as described in
[reproduction](docs/reproduction.md), then validate paths and resources without
starting an experiment:

```bash
python -m recharness check \
  --data-dir ./input --output ./runs/sasrec \
  --family sasrec --gpus 0,1,2,3
```

Run a small, real GPU/LLM preflight in a separate, new directory:

```bash
python -m recharness preflight \
  --data-dir ./input --output ./runs/sasrec-preflight \
  --family sasrec --gpus 0,1,2,3
```

Run the formal search with fresh weights and an independent budget:

```bash
python -m recharness run \
  --data-dir ./input --output ./runs/sasrec \
  --family sasrec --gpus 0,1,2,3 --budget 43200
```

The other registered families are `gru4rec`, `bert4rec`, `nextitnet` and `hstu`.
Each invocation requires a new output directory and runs one family. Four GPUs and
64 available CPU cores are required by the registered protocol. Override
`--cpu-cores` with 64 comma-separated IDs when necessary; each GPU receives a
16-core slot. `--worker-python` can select a separate CUDA Python environment.
The command does not reserve GPUs against other jobs; allocate idle devices before
starting. Preflight is a separate validation step and does not certify a later run.

## Repository map

| Path | Purpose |
| --- | --- |
| `recharness/cli.py` | Portable check/preflight/run entrypoint and input fingerprints |
| `recharness/runner.py` | Latest family-preserving search and evidence integration |
| `recharness/controller.py`, `recharness/core.py` | Candidate execution, shared budget ledger and final selection |
| `recharness/transport.py` | Structured LLM requests, retries and within-run request identity |
| `recharness/worker.py`, `recharness/single_model.py` | External evaluation and model/checkpoint checks |
| `gagc/semantic_arms.py` | Six directions and combination-jump selection |
| `gagc/jump_policy.py`, `gagc/tools.py` | Budget protection, promotion, retuning and rollback |
| `gagc/experiment_memory.py` | Immutable evidence archive, retrieval and duplicate checks |
| `gagc/benchmarks/` | Evaluation contracts and metrics |
| `gagc/templates/` | Cold-start training programs and predictor writers |
| `tests/` | Synthetic regression and controller integration tests |

`gagc` is retained as an internal compatibility package. `run2.sh`, `gr.sh` and
`prepare_gr_data.py` preserve the earlier provider-based Amazon/KuaiRec entrypoints;
install `pip install -e '.[legacy,templates]'` to use those scripts. The standalone
`python -m recharness` entrypoint is the reproduction path for the latest method.
The earlier scripts do not provide the standalone runner's evidence archive and
single-model checks.

## Review artifact

```bash
python scripts/review_artifact.py
python scripts/review_artifact.py --archive ../recharness-review.zip
```

The archive contains a neutral `code/` directory without Git history or local
runtime outputs. Use the ZIP as supplementary material, or mirror its contents
through an anonymous code host. A personal-account repository URL exposes its owner
and must not be used as the anonymous review link. See [review packaging](docs/review.md).

The artifact does not contain aggregate experimental results or claim that the
paper-scale experiments were rerun during packaging. Execution uses trusted local
subprocesses; withholding test paths from search context is a protocol boundary,
not an operating-system security sandbox.

## License

MIT; see [LICENSE](LICENSE).
