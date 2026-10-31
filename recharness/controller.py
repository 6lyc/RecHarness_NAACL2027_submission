"""Native RecHarness routing/promotion/memory with the matched Codex executor."""
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
import sys
import threading

from .core import BudgetExhausted, Controller, TEXT, read, save, schema
from .paths import REPO

sys.path.insert(0, str(REPO))
from gagc import tools as native
from gagc.state import ArmState, ThompsonState
from gagc.schemas import MutationSpec
from gagc.claude_code_backend import _build_prompt
from gagc.prompts import GAGC_ORCHESTRATOR_PROMPT


class StateStore:
    """The native tools' store interface, persisted without changing state logic."""
    def __init__(self, path, initial):
        self.path = Path(path)
        if self.path.exists():
            raise RuntimeError('Existing native state must be explicitly reviewed before resume')
        self.value = {'content': initial.to_json()}
        save(self.path, self.value)

    def get(self, namespace, key):
        if tuple(namespace) != ('gagc', 'thompson') or key != '/state.json':
            raise ValueError('Unexpected state key')
        return SimpleNamespace(value=deepcopy(self.value))

    def put(self, namespace, key, value):
        if tuple(namespace) != ('gagc', 'thompson') or key != '/state.json':
            raise ValueError('Unexpected state key')
        self.value = deepcopy(value)
        save(self.path, value)

    def state(self):
        return ThompsonState.from_json(self.value['content'])


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def matched_group_cost(results):
    return sum(float(r.get('wall_time_secs', 0.) or 0.) for r in results), 'matched_sum_gpu_worker_seconds'


class RecHarnessController(Controller):
    def __init__(self, config, smoke=False):
        super().__init__(config, 'recharness')
        self.smoke = smoke
        self.workspace = self.root / 'native/workspace'
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.promotion_receipts = {}
        self.group_receipts = {}
        self.receipt_lock = threading.Lock()
        self.group_allowance = 0.
        self.mutation_specs = []

    def setup(self):
        random.seed(42)
        os.environ['GAGC_ENABLE_CLAUDE_BACKEND'] = '1'
        os.environ['GAGC_TRIAL_SECS'] = str(self.c['trial_floor_reference_seconds'])
        os.environ['GAGC_COLD_START'] = 'sasrec_perdataset'
        native._BENCHMARK_MODE = 'amazon_reviews'
        native._ROUTING_POLICY = 'thompson'
        native.BASIN_TRANSFER_RHO = 0.25
        native._SELECTION_METRIC = 'val_score'
        native._DATA_DIR = str(Path(self.c['data_dir']) / 'trainval')
        native._TEST_DIR = ''
        native._WORKSPACE_ROOT = str(self.workspace)
        native._LOGS_ROOT = str(self.root / 'native/logs')
        native._SERVER_CONFIG = native.ServerConfig(num_gpus=4, num_cpus=64, cpus_per_trial=16,
             gpu_ids=self.c['gpus'], numa_map={0: self.c['cpu_cores'][:32], 1: self.c['cpu_cores'][32:]})
        native._LAST_PROPOSED_CANDIDATES = []
        native._LAST_TRIAL_RESULTS = []
        native._LAST_SELECTION_LOG = {}
        native._LAST_PROMOTION_RESULT = {}
        native._LAST_TRIAL_GROUP_ID = ''
        # A code-generation failure consumes zero GPU seconds. Do not let the
        # native legacy zero-duration fallback charge LLM wall time as compute.
        native._trial_group_wall_time_secs = matched_group_cost
        initial = ThompsonState(arms={arm: ArmState(name=arm) for arm in native._active_arms()},
             global_budget=self.c['search_budget_gpu_seconds'],
             skill_notes='Cold start: no prior knowledge.', experiment_skill=native.default_experiment_skill_for_policy())
        self.store = StateStore(self.root / 'native/thompson_state.json', initial)
        native._STORE = self.store
        for name in ['best.py', 'train.py']:
            (self.workspace / name).write_text(self.baseline())
        (self.workspace / 'predict.py').write_text((REPO / 'gagc/templates/perdataset/predict.py').read_text())
        # Only training execution/LLM transport changes. Native group preparation,
        # trial isolation, Thompson sampling, promotion, jumps and memory stay native.
        native.execute_trial = self.execute

    def hypotheses(self, specs, diagnostics):
        properties = {str(i): TEXT for i in range(len(specs))}
        value = self.llm(GAGC_ORCHESTRATOR_PROMPT + '\nTRANSPORT TASK: Execute ONLY workflow step 3. '
             'Routing has already been performed by the native Thompson tool. Write one short hypothesis '
             'per selected candidate, under 800 characters each. Do not change candidate arms or write code.\n'
             + json.dumps({'selected': specs, 'diagnostics': diagnostics,
                 'state': json.loads(self.store.state().to_json())}), schema(properties),
             key=f'hypotheses-{self.store.state().round_idx:04d}')
        hypotheses = {}
        for i, spec in enumerate(specs):
            note = value[str(i)]
            if len(note) > 800:
                raise ValueError('Native hypothesis limit exceeded')
            hypotheses[spec.get('arm', spec['dimension'])] = note
        return hypotheses

    def implement(self, source, spec, feedback, attempt, key):
        request = _build_prompt(trial_dir='isolated-host-trial', train_path='train.py', predict_path='predict.py',
            spec_data=spec, benchmark_mode='amazon_reviews', allowed_files=['train.py'],
            env={'SASREC_EPOCHS': '200', 'CUDA_VISIBLE_DEVICES': 'assigned-by-host'},
            feedback=feedback, attempt_idx=attempt, max_attempts=3)
        request += ('\nLLM TRANSPORT ADAPTATION: The host applies exact find/replace edits and runs training. '
            'Return edits for train.py only. Do not run shell commands or inspect files. train.py contains '
            'the predictor writer; update that writer when needed to preserve trained/inference consistency. '
            'Preserve the current model family for architecture edits, as required by the native orchestrator. '
            'Follow the selected low-level arm and avoid unrelated rewrites. '
            'Metrics are supplied by the external host, so never invent them.\nCURRENT TRAIN.PY:\n' + source)
        reply = self.llm(request, schema({'edits': {'type': 'array', 'items': schema({'find': TEXT, 'replace': TEXT})},
                                         'change_summary': TEXT}), key=key)
        if not reply['edits']:
            raise ValueError('A mutation must make an edit')
        changed, error = native._apply_edits(source, reply['edits'])
        if error:
            raise ValueError(error)
        ast.parse(changed)
        if changed == source:
            raise ValueError('Mutation left the source unchanged')
        return changed, reply

    def execute(self, spec_json, script_path='/workspace/best.py', timeout_multiplier=1.5, slot_id=0):
        spec = json.loads(spec_json)
        spec.setdefault('code_diff', '')
        baseline = bool(spec.get('is_baseline'))
        gpu = self.c['gpus'][slot_id]
        path = Path(script_path)
        current = path.read_text()
        group = spec['trial_group_id']
        trial_id = spec['trial_id']
        key = f'round-{self.store.state().round_idx:04d}-slot-{trial_id}'
        spent, feedback, final, attempts = 0., '', None, []
        for attempt in range(1, 2 if baseline else 4):
            remaining = self.group_allowance - spent
            if remaining < self.c['minimum_trial_seconds']:
                feedback = feedback or 'Candidate GPU allowance exhausted'
                break
            try:
                if not baseline:
                    current, reply = self.implement(current, spec, feedback, attempt,
                                                    f'code-{key}-attempt-{attempt}')
                    attempts.append({'attempt': attempt, 'proposal': reply})
                path.write_text(current)
                final = self.trial(key + f'-attempt-{attempt}', current, gpu, requested=remaining, smoke=self.smoke)
            except BudgetExhausted as exc:
                feedback = str(exc)
                break
            except Exception as exc:
                feedback = str(exc)
                attempts.append({'attempt': attempt, 'error': feedback})
                continue
            spent += final['allocated_gpu_seconds']
            feedback = final.get('error', '')
            if final['status'] == 'success':
                predictor = Path(final['output']) / 'working/predict.py'
                (path.parent / 'predict.py').write_text(predictor.read_text())
                break
        ok = final is not None and final['status'] == 'success'
        timed_out = final is not None and final['status'] == 'timeout'
        oom = 'out of memory' in feedback.lower()
        train_log = (Path(final['output']) / 'training.log') if final else None
        stdout = train_log.read_text() if train_log and train_log.exists() else ''
        metric = final['score'] if ok else -1.0 if timed_out else -0.5 if oom else 0.0
        result = json.loads(native._make_trial_result(MutationSpec(**spec), wall_time=spent,
            timed_out=timed_out, oom=oom, val_score=metric,
            convergence_trace=native._parse_convergence_trace(stdout),
            val_metrics={k: v['metrics'] for k, v in final['per_dataset'].items()} if ok else None,
            error=None if ok else feedback or 'Candidate execution failed', stdout_tail=stdout[-3000:],
            stderr_tail=feedback[-2000:]))
        if ok:
            with self.receipt_lock:
                self.group_receipts[(group, trial_id)] = final
        save(self.root / 'native/execution' / (key + '.json'), {'attempts': attempts, 'receipt': final,
             'native_result': result, 'gpu_seconds_all_attempts': spent})
        return json.dumps(result)

    def group(self, specs, diagnostics=None):
        self.group_allowance = min(self.c['trial_timeout_seconds'],
             (600. if self.smoke else self.ledger.available() / len(specs) - 3.))
        if self.group_allowance < self.c['minimum_trial_seconds']:
            raise BudgetExhausted('Insufficient aggregate budget for the native group')
        self.mutation_specs = specs
        hypotheses = {} if specs[0].get('is_baseline') else self.hypotheses(specs, diagnostics or {})
        results = json.loads(native.execute_trial_group('__LAST_PROPOSED__',
             script_path=str(self.workspace / 'best.py'), hypotheses_json=json.dumps(hypotheses)))
        if not results or any(r.get('tool_error') for r in results):
            raise RuntimeError('Native trial-group preparation failed: ' + json.dumps(results))
        promotion = json.loads(native.promote_winner(script_path=str(self.workspace / 'best.py')))
        if promotion.get('promoted'):
            i = promotion['winner_idx']
            result = results[i]
            receipt = self.group_receipts[(result['trial_group_id'], result['trial_id'])]
            self.promotion_receipts[digest((self.workspace / 'best.py').read_text())] = receipt
        native.update_thompson_state('__LAST_RESULTS__', trial_group_id=results[0]['trial_group_id'])
        if not self.smoke:
            ledger = read(self.ledger.path)
            expected = max(0., ledger['total_gpu_seconds'] - ledger['spent_gpu_seconds'])
            if ledger['reservations'] or not math.isclose(self.store.state().global_budget, expected, abs_tol=1e-5):
                raise RuntimeError('Native and shared GPU budgets diverged')
        save(self.root / 'progress.json', {'round': self.store.state().round_idx,
             'available_gpu_seconds': self.ledger.available(),
             'native_remaining_gpu_seconds': self.store.state().global_budget,
             'current_basin': self.store.state().current_basin_id,
             'pending_jumps': self.store.state().pending_queue,
             'global_best_validation': self.brief(self.best) if self.best else None,
             'promotion': promotion})
        return results



def run(config):
    return RecHarnessController(config).run()
