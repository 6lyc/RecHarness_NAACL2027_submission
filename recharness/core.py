"""Shared execution, accounting, and LLM transport for matched experiments."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .paths import DATASETS, REPO, ROOT, WORKSPACE, sha

TEXT = {'type': 'string'}


def schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def strict_schema(value):
    """Close nested Pydantic definitions and require nullable optional fields."""
    if isinstance(value, list):
        return [strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: strict_schema(child) for key, child in value.items() if key != 'default'}
    if result.get('type') == 'object':
        if 'properties' not in result:
            raise ValueError('An unbounded object cannot be sent as a strict response schema')
        properties = result['properties']
        required = set(result.get('required', properties))
        for name in set(properties) - required:
            properties[name] = {'anyOf': [properties[name], {'type': 'null'}]}
        result['required'] = list(properties)
        result['additionalProperties'] = False
    return result


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


class BudgetExhausted(Exception):
    pass


class Ledger:
    """Reserve concurrent worker ceilings before launch; charge real elapsed time."""
    def __init__(self, root, total):
        self.root, self.total = Path(root), float(total)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'budget.json'
        self.mutex = threading.RLock()
        if not self.path.exists():
            save(self.path, {'total_gpu_seconds': self.total, 'spent_gpu_seconds': 0.,
                             'reservations': {}, 'settled': {}})

    @contextmanager
    def transaction(self):
        with self.mutex, (self.root / 'budget.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(self.path)
            try:
                yield state
                save(self.path, state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def available(self):
        with self.transaction() as state:
            return max(0., self.total - state['spent_gpu_seconds'] - sum(state['reservations'].values()))

    def reserve(self, key, requested, minimum=30, fair_slots=None):
        with self.transaction() as state:
            if key in state['reservations'] or key in state['settled']:
                raise RuntimeError('Duplicate budget key: ' + key)
            remaining = self.total - state['spent_gpu_seconds'] - sum(state['reservations'].values())
            if fair_slots:
                requested = min(requested, remaining / max(1, fair_slots - len(state['reservations'])))
            timeout = math.floor(min(float(requested), remaining) - 2.)
            if timeout < minimum:
                raise BudgetExhausted('No available GPU allocation for ' + key)
            # Two seconds of launch/termination overhead are also reserved.
            state['reservations'][key] = timeout + 2.
            return timeout

    def settle(self, key, elapsed):
        with self.transaction() as state:
            reservation = state['reservations'].pop(key)
            # Keep measured costs even if OS process teardown exceeds the allowance.
            state['spent_gpu_seconds'] += float(elapsed)
            state['settled'][key] = {'allocated_gpu_seconds': float(elapsed),
                                     'reservation': reservation,
                                     'overrun_seconds': max(0., elapsed - reservation)}


TASK = '''Matched RecHarness Amazon task: four datasets Movies_and_TV,
Industrial_and_Scientific, Electronics, CDs_and_Vinyl, macro-average validation
HR@10. Start from the corrected family-specific cold start supplied by the host. Preserve that model family. Preprocessing seed=0 and training
seed=42, exactly as run2.sh. The host executes your train.py and then independently
evaluates working/predict.py using the pinned RecHarness benchmark (200 validation
users/category, 99 unique negative candidates, evaluation seed=42). Test labels and
test metrics are unavailable during search. Retain all four categories and dataset
routing. DATA_DIR must remain os.environ['GAGC_DATA_DIR']; write artifacts beneath
your trial directory, as the baseline does. Training may change model architecture,
loss, optimizer, sampling and hyperparameters, as in RecHarness. Do not alter input
splits, invent metrics, read hidden test files, fetch extra datasets/pretrained weights,
or change the host evaluator. Changes to inference must also update the predictor
writer so working/predict.py loads the trained architecture. Keep execution single-GPU;
the host provides four concurrent trial slots and 64 CPU cores. No shell actions,
filesystem inspection, or training tools are needed in the LLM call: return code.
All training, diagnostics, debugging and failed trials consume the method's shared
43,200 GPU-second search budget. LLM time and final held-out inference are recorded
separately, matching run2.sh's separate final-evaluation step.
The original RecHarness tools layer sets SASREC_EPOCHS=200 (overriding the template's
standalone default of 300); this same environment default applies here.
'''


class Controller:
    def __init__(self, config, method):
        self.c, self.method = config, method
        self.root = Path(config['root']) / method
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.root, config['search_budget_gpu_seconds'])
        self.event_lock = threading.RLock()
        self.call_lock = threading.Lock()
        self.gpu_locks = {g: threading.Lock() for g in self.c['gpus']}
        self.call_index = 0
        self.best = None

    def event(self, kind, **data):
        with self.event_lock:
            row = {'time': time.time(), 'method': self.method, 'kind': kind, **data}
            with (self.root / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def llm(self, instruction, output_schema=None, search=False, key=None):
        from .codex_schema import remove_optional_nulls
        import jsonschema
        with self.call_lock:
            self.call_index += 1
            directory = self.root / 'calls' / (key or f'{self.call_index:05d}')
            directory.mkdir(parents=True, exist_ok=False)
        original = output_schema or schema({'text': TEXT})
        transport = strict_schema(original)
        save(directory / 'schema.json', transport)
        prompt = TASK + '\nMETHOD: ' + self.method + '\n' + instruction
        if output_schema is None:
            prompt += '\nReturn the requested completion verbatim inside the JSON text field.'
        save(directory / 'request.json', {'prompt': prompt, 'schema': original, 'search': search})
        command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check',
                   '--ignore-user-config', '--ignore-rules', '--model', self.c['llm_model'],
                   '--sandbox', 'read-only', '-c', 'approval_policy="never"',
                   '-c', 'model_reasoning_effort="' + self.c['llm_effort'] + '"',
                   '-c', 'web_search="live"' if search else 'web_search="disabled"',
                   '--json', '--output-schema', str(directory / 'schema.json'),
                   '-o', str(directory / 'answer.json'), '-C', str(directory), '-']
        save(directory / 'started.json', {'command': command, 'time': time.time()})
        self.event('llm_started', call=directory.name, search=search)
        start = time.monotonic()
        with (directory / 'events.jsonl').open('w') as out, (directory / 'stderr.log').open('w') as err:
            result = subprocess.run(command, input=prompt, text=True, stdout=out, stderr=err,
                                    timeout=self.c['llm_timeout'], env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        if result.returncode or not (directory / 'answer.json').exists():
            raise RuntimeError('LLM call failed; retained for inspection: ' + str(directory))
        response = remove_optional_nulls(read(directory / 'answer.json'), original)
        jsonschema.validate(response, original)
        save(directory / 'response.json', response)
        save(directory / 'receipt.json', {'wall_seconds': time.monotonic() - start,
                                         'model': self.c['llm_model'], 'effort': self.c['llm_effort']})
        self.event('llm_completed', call=directory.name)
        return response if output_schema is not None else response['text']

    def baseline(self):
        path = Path(self.c['baseline']['source'])
        if sha(path) != self.c['baseline']['sha256']:
            raise RuntimeError('Registered corrected baseline changed')
        return path.read_text()

    def trial(self, key, source, gpu, requested=None, files=None, smoke=False):
        with self.gpu_locks[gpu]:
            return self._trial(key, source, gpu, requested, files, smoke)

    def _trial(self, key, source, gpu, requested, files, smoke):
        directory = self.root / ('smoke' if smoke else 'trials') / key
        if directory.exists():
            raise RuntimeError('Existing trial is never silently retrained: ' + str(directory))
        directory.mkdir(parents=True)
        (directory / 'train.py').write_text(source)
        for name, contents in (files or {}).items():
            if name not in {'diagnosis_tools.py', 'predict.py', 'models.py'}:
                raise ValueError('Unexpected candidate file: ' + name)
            (directory / name).write_text(contents)
        save(directory / 'request.json', {'id': key, 'gpu': gpu, 'source_sha256': sha(directory / 'train.py'),
                                         'smoke': smoke, 'files': sorted((files or {}).keys())})
        try:
            ast.parse(source)
        except SyntaxError as exc:
            receipt = {'id': key, 'status': 'invalid_source', 'error': str(exc),
                       'allocated_gpu_seconds': 0., 'output': str(directory), 'score': None}
            save(directory / 'receipt.json', receipt)
            return receipt
        timeout = float(requested or self.c['trial_timeout_seconds'])
        if not smoke:
            timeout = self.ledger.reserve(key, timeout, self.c['minimum_trial_seconds'],
                                          fair_slots=len(self.c['gpus']) if requested is None else None)
        env = self.environment(gpu)
        command = [self.c['worker_python'], '-u', '-m', 'recharness.worker',
                   '--trial', str(directory), '--data', self.c['data_dir']]
        if self.c.get('single_model_search'):
            command += ['--single-model']
        if smoke:
            command += ['--smoke']
        self.event('trial_started', id=key, gpu=gpu, timeout=timeout, smoke=smoke)
        start = time.monotonic()
        try:
            with (directory / 'worker.log').open('w') as log:
                proc = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True)
                save(directory / 'pid.json', {'pid': proc.pid, 'started_at': time.time()})
                try:
                    exit_code = proc.wait(timeout=timeout)
                    status = 'success' if exit_code == 0 and (directory / 'result.json').exists() else 'crash'
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    status = 'timeout'
        finally:
            elapsed = time.monotonic() - start
            if not smoke:
                self.ledger.settle(key, elapsed)
        receipt = {'id': key, 'status': status, 'gpu': gpu, 'allocated_gpu_seconds': elapsed,
                   'output': str(directory), 'source_sha256': sha(directory / 'train.py'), 'score': None}
        if status == 'success':
            receipt.update(read(directory / 'result.json'))
        else:
            receipt['error'] = (directory / 'worker.log').read_text()[-6000:]
        save(directory / 'receipt.json', receipt)
        self.event('trial_completed', id=key, status=status, score=receipt.get('score'), seconds=elapsed)
        with self.event_lock:
            if status == 'success' and not smoke and (self.best is None or receipt['score'] > self.best['score']):
                self.best = receipt
                save(self.root / 'incumbent.json', self.brief(receipt))
        return receipt

    def environment(self, gpu):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
               'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '42',
               'GAGC_DATA_DIR': str(Path(self.c['data_dir']) / 'trainval'),
               'GAGC_DATASETS': ','.join(DATASETS), 'GAGC_FULL_RANKING': 'false',
               'PYTHONPATH': str(WORKSPACE) + os.pathsep + str(REPO)}
        for name in ['SASREC', 'GRU4REC', 'BERT4REC', 'NEXTITNET', 'HSTU']:
            env[name + '_EPOCHS'] = '200'
        slot = self.c['gpus'].index(gpu)
        cores = self.c['cpu_cores'][slot * 16:(slot + 1) * 16]
        env['RECHARNESS_CPU_CORES'] = ','.join(map(str, cores))
        env.pop('GAGC_TEST_DIR', None)
        env.pop('AMAZON_REVIEWS_INTERFACE_DIR', None)
        # LLM credentials are unnecessary in training workers.
        for key in list(env):
            if key.endswith('API_KEY') or key.endswith('ACCESS_TOKEN'):
                env.pop(key)
        return env

    def wave(self, proposals):
        remaining = self.ledger.available()
        count = min(len(proposals), len(self.c['gpus']))
        if count != len(proposals):
            raise ValueError('A wave cannot exceed four GPU slots')
        per_job = min(self.c['trial_timeout_seconds'], remaining / max(count, 1) - 3.)
        if per_job < self.c['minimum_trial_seconds']:
            raise BudgetExhausted('Insufficient aggregate allocation for another wave')
        with ThreadPoolExecutor(max_workers=count) as pool:
            return list(pool.map(lambda pair: self.trial(pair[0]['id'], pair[0]['source'], pair[1],
                                  requested=per_job, files=pair[0].get('files')),
                                 zip(proposals, self.c['gpus'])))

    @staticmethod
    def brief(receipt):
        return {key: receipt.get(key) for key in ['id', 'status', 'score', 'per_dataset', 'output',
                                                  'allocated_gpu_seconds', 'error']}


    def finish(self):
        if self.best is None:
            raise RuntimeError('No valid model is available for final evaluation')
        selected = self.best
        save(self.root / 'selection.json', {'selection_metric': 'macro validation HR@10',
                                           'selected': self.brief(selected), 'frozen_at': time.time()})
        output = self.root / 'final-test'
        output.mkdir(exist_ok=False)
        command = [self.c['worker_python'], '-u', '-m', 'recharness.worker',
                   '--trial', selected['output'], '--data', self.c['data_dir'],
                   '--mode', 'test', '--output', str(output)]
        start = time.monotonic()
        with (output / 'worker.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
                                    env=self.environment(self.c['gpus'][0]), start_new_session=True)
            try:
                code = proc.wait(timeout=self.c['final_timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = -9
        receipt = {'status': 'success' if code == 0 else 'failed', 'seconds': time.monotonic() - start,
                   'accounting': 'separate final-test overhead, matching RecHarness run2.sh'}
        save(output / 'receipt.json', receipt)
        if code:
            raise RuntimeError('Selected final test failed; do not silently retry')
        result = {'status': 'complete', 'method': self.method, 'selected': self.brief(selected),
                  'test': read(output / 'result.json'), 'budget': read(self.ledger.path),
                  'final_inference': receipt, 'llm_model': self.c['llm_model'],
                  'llm_effort': self.c['llm_effort'], 'alignment': self.c['alignment']}
        save(self.root / 'summary.json', result)
        self.event('complete', test_score=result['test']['score'])
        return result
