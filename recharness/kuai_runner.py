"""KuaiRec host loop using the shared Codex transport and native trial tools."""
import ast
from contextlib import contextmanager
import contextvars
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from gagc import tools as native
from gagc.prompts import GR_ORCHESTRATOR_PROMPT
from gagc.schemas import MutationSpec
from gagc.state import ArmState, ThompsonState
from gagc.templates import get_template_dir

from .controller import StateStore, matched_group_cost
from .core import BudgetExhausted, Controller, TEXT, read, save, schema
from .transport import LLMTransport, TransportUnavailable


TASK = '''RecHarness KuaiRec task: maximize validation watch-time xAUC, using lower
watch-time MAE to break ties. Monitor watch-ratio xAUC and MAE. GR_TRAIN_DATA and
GR_VAL_DATA are supplied by the host; the compatibility GR_TEST_DATA variable is
validation only during search. Preserve the 53-column schema, training-only fitted
statistics, validation-selected gr_checkpoint.pt, frozen evaluation entrypoint and
metric printing contract. Do not read test files, fetch external data or pretrained
weights, alter split selection, or invent metrics. The host executes isolated
single-GPU training jobs and charges every attempt to one shared compute budget.
LLM time and final held-out inference are recorded separately. Return structured
code edits only; do not inspect files, invoke tools or run commands.
'''


class KuaiController(LLMTransport, Controller):
    task = TASK

    def __init__(self, config, smoke=False):
        Controller.__init__(self, config, 'recharness')
        self.smoke = smoke
        self.scope = contextvars.ContextVar('kuairec_request_scope', default='baseline')
        self.workspace = self.root / 'native/workspace'
        self.group_allowance = 0.
        self.native_execute = native.execute_trial

    @contextmanager
    def request_scope(self, value):
        token = self.scope.set(value)
        try:
            yield
        finally:
            self.scope.reset(token)

    def run_attempt(self, *args):
        try:
            return LLMTransport.run_attempt(self, *args)
        except OSError as exc:
            raise RuntimeError('LLM CLI launch failed: ' + str(exc)) from exc

    def llm(self, instruction, output_schema=None, search=False, key=None):
        with self.request_scope(key or 'unspecified'):
            while True:
                try:
                    return LLMTransport.llm(self, instruction, output_schema, search, key)
                except TransportUnavailable as exc:
                    self.event('llm_cooldown', key=key, error=str(exc), seconds=60)
                    if self.smoke:
                        raise
                    time.sleep(60)

    def setup(self):
        random.seed(42)
        os.environ['GAGC_TRIAL_SECS'] = str(self.c['trial_floor_reference_seconds'])
        os.environ['GAGC_COLD_START'] = self.c['cold_start']
        native._BENCHMARK_MODE = 'kuairec'
        native._ROUTING_POLICY = 'thompson'
        native.BASIN_TRANSFER_RHO = self.c['basin_transfer_rho']
        native._SELECTION_METRIC = 'val_score'
        native._GR_TRAIN_DATA = self.c['train_data']
        native._GR_VAL_DATA = self.c['val_data']
        native._GR_TEST_DATA = self.c['test_data']
        native._DATA_DIR = native._TEST_DIR = ''
        native._WORKSPACE_ROOT = str(self.workspace)
        native._LOGS_ROOT = str(self.root / 'native/logs')
        native._SERVER_CONFIG = native.ServerConfig(
            num_gpus=4, num_cpus=64, cpus_per_trial=16, gpu_ids=self.c['gpus'],
            numa_map={0: self.c['cpu_cores'][:32], 1: self.c['cpu_cores'][32:]})
        native._LAST_PROPOSED_CANDIDATES = []
        native._LAST_TRIAL_RESULTS = []
        native._LAST_SELECTION_LOG = {}
        native._LAST_PROMOTION_RESULT = {}
        native._LAST_TRIAL_GROUP_ID = ''
        native._trial_group_wall_time_secs = matched_group_cost
        initial = ThompsonState(
            arms={arm: ArmState(name=arm) for arm in native._active_arms()},
            global_budget=self.c['search_budget_gpu_seconds'],
            skill_notes='Cold start: no prior knowledge.',
            experiment_skill=native.default_experiment_skill_for_policy())
        self.store = StateStore(self.root / 'native/thompson_state.json', initial)
        native._STORE = self.store
        shutil.copytree(get_template_dir(self.c['cold_start']), self.workspace,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(self.workspace / 'train.py', self.workspace / 'best.py')
        native.execute_trial = self.execute

    def hypotheses(self, specs, diagnostics):
        reply = self.llm(
            GR_ORCHESTRATOR_PROMPT + '\nHOST TASK: Routing is complete. Write one hypothesis '
            'under 800 characters for each selected candidate. Preserve its arm and do not write code.\n'
            + json.dumps({'selected': specs, 'diagnostics': diagnostics,
                          'state': json.loads(self.store.state().to_json())}),
            schema({str(i): TEXT for i in range(len(specs))}),
            key=f'hypotheses-{self.store.state().round_idx:04d}')
        if any(len(note) > 800 for note in reply.values()):
            raise ValueError('Native hypothesis limit exceeded')
        return {spec.get('arm', spec['dimension']): reply[str(i)] for i, spec in enumerate(specs)}

    def implement(self, path, spec, feedback, attempt, key):
        allowed = spec.get('allowed_files') or ['train.py']
        available = {'train.py'} | {p.relative_to(path.parent).as_posix()
                                   for p in (path.parent / 'model').rglob('*.py')}
        if not set(allowed) <= available:
            raise ValueError('Mutation requests a file outside the template')
        sources = {name: (path.parent / name).read_text() for name in allowed}
        reply = self.llm(
            'Implement the selected KuaiRec candidate with exact find/replace edits. '
            'Change only the selected arm, retain training and frozen inference consistency, '
            'and preserve validation-selected checkpoint saving. The host applies edits and '
            'runs training; no commands or tool calls are needed.\n'
            + json.dumps({'spec': spec, 'sources': sources, 'feedback': feedback,
                          'attempt': attempt, 'maximum_attempts': 3}),
            schema({'files': {'type': 'array', 'items': schema({
                'path': {'type': 'string', 'enum': list(allowed)},
                'edits': {'type': 'array', 'items': schema({'find': TEXT, 'replace': TEXT})}})},
                'change_summary': TEXT}), key=key)
        changes = {}
        for item in reply['files']:
            name = item['path']
            if name not in sources or name in changes or not item['edits']:
                raise ValueError('Invalid or duplicate candidate file edits')
            changed, error = native._apply_edits(sources[name], item['edits'])
            if error:
                raise ValueError(error)
            ast.parse(changed)
            if changed == sources[name]:
                raise ValueError('Mutation left the source unchanged')
            changes[name] = changed
        if not changes:
            raise ValueError('A mutation must make an edit')
        for name, changed in changes.items():
            (path.parent / name).write_text(changed)
        return reply

    def execute(self, spec_json, script_path='/workspace/best.py', timeout_multiplier=1.5, slot_id=0):
        spec = json.loads(spec_json)
        path = Path(script_path)
        key = f'round-{self.store.state().round_idx:04d}-slot-{spec["trial_id"]}'
        baseline = bool(spec.get('is_baseline'))
        spent, feedback, final, attempts = 0., '', None, []
        for attempt in range(1, 2 if baseline else 4):
            remaining = self.group_allowance - spent
            if remaining < self.c['minimum_trial_seconds']:
                feedback = feedback or 'Candidate GPU allowance exhausted'
                break
            try:
                if not baseline:
                    proposal = self.implement(path, spec, feedback, attempt,
                                              f'code-{key}-attempt-{attempt}')
                    attempts.append({'attempt': attempt, 'proposal': proposal})
                training_spec = {**spec, 'implementation_backend': 'host_applied',
                                 'code_diff': '', 'code_edits': [], 'code_content': '',
                                 'predict_code_edits': [], 'predict_code_content': ''}
                allocation_key = key + f'-attempt-{attempt}'
                timeout = self.ledger.reserve(allocation_key, remaining, self.c['minimum_trial_seconds'])
            except BudgetExhausted as exc:
                feedback = str(exc)
                break
            except Exception as exc:
                feedback = str(exc)
                attempts.append({'attempt': attempt, 'error': feedback})
                continue
            training_spec['max_trial_gpu_seconds'] = timeout
            start = time.monotonic()
            try:
                final = json.loads(self.native_execute(json.dumps(training_spec), str(path),
                                                       timeout_multiplier, slot_id))
                feedback = final.get('error_message') or ''
            except Exception as exc:
                final = None
                feedback = str(exc)
                attempts.append({'attempt': attempt, 'error': feedback})
            finally:
                elapsed = time.monotonic() - start
                self.ledger.settle(allocation_key, elapsed)
                spent += elapsed
            if final is not None and native._is_valid_trial(final):
                break
        if final is None:
            final = json.loads(native._make_trial_result(
                MutationSpec(**spec), wall_time=spent, timed_out=False, oom=False,
                val_score=0., convergence_trace=[], error=feedback or 'Candidate execution failed'))
        final['wall_time_secs'] = spent
        save(self.root / 'native/execution' / (key + '.json'),
             {'attempts': attempts, 'native_result': final, 'gpu_seconds_all_attempts': spent})
        return json.dumps(final)

    def group(self, specs, diagnostics=None):
        self.group_allowance = min(self.c['trial_timeout_seconds'],
                                   self.ledger.available() / len(specs) - 3.)
        if self.group_allowance < self.c['minimum_trial_seconds']:
            raise BudgetExhausted('Insufficient aggregate budget for the native group')
        hypotheses = {} if specs[0].get('is_baseline') else self.hypotheses(specs, diagnostics or {})
        results = json.loads(native.execute_trial_group(
            '__LAST_PROPOSED__', script_path=str(self.workspace / 'best.py'),
            hypotheses_json=json.dumps(hypotheses)))
        if not results or any(r.get('tool_error') for r in results):
            raise RuntimeError('Native trial-group preparation failed: ' + json.dumps(results))
        promotion = json.loads(native.promote_winner(script_path=str(self.workspace / 'best.py')))
        native.update_thompson_state('__LAST_RESULTS__', trial_group_id=results[0]['trial_group_id'])
        ledger = read(self.ledger.path)
        expected = max(0., ledger['total_gpu_seconds'] - ledger['spent_gpu_seconds'])
        if ledger['reservations'] or not math.isclose(self.store.state().global_budget, expected, abs_tol=1e-5):
            raise RuntimeError('Native and shared GPU budgets diverged')
        save(self.root / 'progress.json', {'round': self.store.state().round_idx,
             'available_gpu_seconds': self.ledger.available(), 'promotion': promotion})
        return results

    def run(self, preflight=False):
        self.setup()
        try:
            diagnostics = {}
            while True:
                specs = json.loads(native.propose_action_group(
                    self.store.state().to_json(), json.dumps(diagnostics)))
                if not specs:
                    break
                before = self.ledger.available()
                try:
                    results = self.group(specs, diagnostics)
                except BudgetExhausted:
                    break
                if specs[0].get('is_baseline') and not all(native._is_valid_trial(r) for r in results):
                    raise RuntimeError('KuaiRec baseline failed; inspect the retained trial receipt')
                valid = [r for r in results if native._is_valid_trial(r)]
                diagnostics = valid[native._select_winner_index(valid)] if valid else results[0]
                if preflight and not specs[0].get('is_baseline'):
                    if len(results) != 4 or len(valid) != 4:
                        raise RuntimeError('KuaiRec four-slot preflight failed')
                    save(self.root / 'preflight.json', {'passed': True, 'final_test_evaluated': False,
                                                      'budget': read(self.ledger.path)})
                    return
                if self.ledger.available() >= before:
                    raise RuntimeError('Search made no compute progress; inspect candidate receipts')
            if preflight:
                raise RuntimeError('Insufficient budget for baseline and four-slot preflight')
            final = json.loads(native.evaluate_final_incumbent(
                str(self.workspace / 'best.py'), timeout_secs=self.c['final_timeout_seconds']))
            if not final.get('ok'):
                raise RuntimeError('Frozen KuaiRec evaluation failed: ' + json.dumps(final))
            result = {'status': 'complete', 'test': final, 'budget': read(self.ledger.path),
                      'llm_model': self.c['llm_model'], 'llm_effort': self.c['llm_effort']}
            save(self.root / 'summary.json', result)
            return result
        finally:
            native.execute_trial = self.native_execute
