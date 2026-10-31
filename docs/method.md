# Method implementation

The current policy is `semantic_v4`; evidence memory has schema version 2.
`recharness.runner.FamilyJumpController` combines the native routing/promotion state
machine with a matched external worker and a structured LLM transport.

## Local search

Six regular arms share four parent categories. Each decision makes one independent
Beta draw per direction and ranks the directions without replacement. A regular
candidate follows one direction. The hyperparameter arm can jointly adjust several
compatible settings of the existing model. Source edits must cover the selected
categories exactly and change executable syntax; semantic scope remains enforced
through the prompt and reviewable source/rationale, rather than a complete program
semantics proof.

Routing and promotion use the macro-average validation HR@10 across the four
Amazon categories. A regular candidate must strictly improve its current basin
incumbent. Secondary metrics are reporting outputs. Failures and invalid trials
cannot be promoted. The global incumbent is retained across provisional basins.

## Combination jumps

A jump combines two to four distinct registered directions across at least two
parent categories. It preserves the registered model family and trains one model
per dataset. There are six regular arms and one delayed jump-outcome counter;
combinations are not additional independent bandit arms. A batch contains at most
four distinct combinations, and only its best valid candidate enters a new basin.

The default policy waits for two stagnant rounds and a recent improvement gap below
0.03, or triggers protected exploration when the remaining-budget gate opens. At
most one jump window is attempted. Up to 35% of the initial search budget protects
the jump batch and four retuning groups. The cap uses recent measured worker costs
(the 75th percentile times 1.25, with a five-second margin and minimum-trial floor).
Widths shrink to fit that envelope. Exact conditions are implemented in
`gagc/jump_policy.py`; `configs/JUMP_POLICY.json` documents their defaults.

Delayed jump credit uses the full retuning window. An accepted basin must beat the
pre-jump global incumbent. At budget exhaustion, an incomplete window restores the
previous incumbent unless the provisional basin already improved it; it receives
no delayed posterior credit. Failed restoration stops final evaluation.

## Evidence memory

Every completed candidate records its hypothesis/proposal history, exact source
and AST-normalized hashes, source-expression changes, input-relative and global
validation deltas, per-category metrics, attempts, costs and failure status. Atomic,
checksummed observations remain immutable. Exact programs are content addressed;
separate events reconcile accepted, rejected and incomplete jump windows.

The hot state retains 50 observations and eight textual lessons, while the full
archive remains available for retrieval. Each request receives at most 12 relevant
records and 18,000 Unicode characters, balanced across source, direction, basin,
best validation, negative evidence and recency. Training facts use training files
only. Successful AST-identical programs from the same run are rejected before GPU
dispatch; failed or timed-out programs remain repairable. No cross-run memory or
held-out feedback is imported. A configuration's validation gain is an observation,
not causal attribution to individual edits.

## Accounting and selection

All worker attempts, including failed training, validation, audit and timeout work,
consume the same sum-of-worker-seconds search ledger. Concurrent ceilings are
reserved before dispatch. LLM wall time and final held-out inference are recorded
separately. Smoke preflight uses reduced training in its own output directory and
reports its cost outside the formal search budget.

The final program must match an actual trained candidate receipt. Before test
inference, the selected predictor and checkpoints must match their single-model
audit hashes. The test set is evaluated after jump resolution, never used for
routing, memory retrieval or promotion. The local audit is an explicit protocol
check, not a proof against arbitrary adversarial candidate code.
