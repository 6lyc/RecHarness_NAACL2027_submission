"""Synthetic host-loop tests with real local CLI and training subprocesses."""
import json
import os
from pathlib import Path
import sys
import textwrap

import numpy as np
import pytest

from gagc import tools as native
from recharness import kuai_cli
from recharness.core import read
from recharness.kuai_runner import KuaiController
from .helpers import config


@pytest.fixture
def controller(tmp_path, monkeypatch):
    # Native tools use process-scoped state. Restore it after each integration test.
    for name in list(vars(native)):
        if name.startswith('_') and not name.startswith('__') or name == 'BASIN_TRANSFER_RHO':
            monkeypatch.setattr(native, name, getattr(native, name))
    monkeypatch.setattr(native, 'execute_trial', native.execute_trial)
    settings = config(tmp_path / 'run')
    for name in ['train', 'val', 'test']:
        path = tmp_path / (name + '.npy')
        path.write_bytes(name.encode())
        settings[name + '_data'] = str(path)
    settings.update(cold_start='gr', basin_transfer_rho=.25, trial_floor_reference_seconds=6400)
    c = KuaiController(settings, smoke=True)
    c.setup()
    source = '''import os
from pathlib import Path
value = 0
if os.environ['GR_RUN_MODE'] == 'train':
    assert os.environ['GR_TEST_DATA'] == os.environ['GR_VAL_DATA']
    assert 'GR_EVAL_DATA' not in os.environ
    Path(os.environ['GR_TRAIN_DATA']).read_bytes()
    Path(os.environ['GR_VAL_DATA']).read_bytes()
    Path(__file__).with_name('gr_checkpoint.pt').write_bytes(b'validation-weights')
else:
    assert 'GR_TRAIN_DATA' not in os.environ and 'GR_VAL_DATA' not in os.environ
    Path(os.environ['GR_EVAL_DATA']).read_bytes()
    Path(os.environ['GR_CHECKPOINT_PATH']).read_bytes()
print(f'XAUC={0.5 + value / 100}')
print('MAE=1.0')
print('WR_XAUC=0.6')
print('WR_MAE=0.2')
'''
    source = 'def main():\n' + textwrap.indent(source, '    ') + '\nif __name__ == "__main__":\n    main()\n'
    for name in ['train.py', 'best.py']:
        (c.workspace / name).write_text(source)
    executable = tmp_path / 'bin/codex'
    executable.parent.mkdir()
    executable.write_text(f'#!{sys.executable}\n' + '''import json
from pathlib import Path
import re
import sys
args = sys.argv[1:]
assert args[args.index('--model') + 1] == 'synthetic-test-model'
assert 'model_reasoning_effort="medium"' in args
prompt = sys.stdin.read()
assert 'RecHarness KuaiRec task' in prompt and 'RecHarness Amazon task' not in prompt
request = json.loads(prompt.rsplit('\\n', 1)[1])
if 'sources' in request:
    source = request['sources']['train.py']
    number = int(re.search(r'value = (\\d+)', source).group(1))
    answer = {'files': [{'path': 'train.py', 'edits': [
        {'find': f'value = {number}', 'replace': f'value = {number + 1}'}]}],
        'change_summary': 'synthetic edit'}
else:
    answer = {str(i): 'synthetic hypothesis' for i in range(len(request['selected']))}
Path(args[args.index('-o') + 1]).write_text(json.dumps(answer))
''')
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(executable.parent) + os.pathsep + os.environ.get('PATH', ''))
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic-test-secret')
    return c


def proposed(c, diagnostics=None):
    return json.loads(native.propose_action_group(c.store.state().to_json(), json.dumps(diagnostics or {})))


def test_baseline_and_parallel_codex_mutations_retain_weights_and_test_once(controller):
    c = controller
    baseline = c.group(proposed(c))
    assert len(baseline) == 1 and native._is_valid_trial(baseline[0])
    assert c.call_index == 0
    specs = proposed(c, baseline[0])
    assert len(specs) == 4
    results = c.group(specs, baseline[0])
    assert all(native._is_valid_trial(r) for r in results)
    assert c.call_index == 5
    assert 'value = 1' in (c.workspace / 'best.py').read_text()
    ledger = read(c.ledger.path)
    assert not ledger['reservations'] and len(ledger['settled']) == 5
    assert sum(r['wall_time_secs'] for r in baseline + results) == pytest.approx(ledger['spent_gpu_seconds'])
    assert c.store.state().global_budget == pytest.approx(ledger['total_gpu_seconds'] - ledger['spent_gpu_seconds'])
    for request in (c.root / 'calls').glob('*/request.json'):
        assert c.c['test_data'] not in request.read_text()
        assert 'synthetic-test-secret' not in request.read_text()
    for path in [c.c['train_data'], c.c['val_data']]:
        Path(path).unlink()
    final = json.loads(native.evaluate_final_incumbent(str(c.workspace / 'best.py')))
    assert final['ok'] and final['test_score'] == .51
    assert not json.loads(native.evaluate_final_incumbent(str(c.workspace / 'best.py')))['ok']
    with pytest.raises(RuntimeError, match='search is closed'):
        native.propose_action_group(c.store.state().to_json())


def test_failed_training_is_charged_and_repaired_without_changing_backend(controller, monkeypatch):
    c = controller
    c.group(proposed(c))
    calls = []
    execute = c.native_execute
    def fail_once(spec_json, script_path, *args):
        spec = json.loads(spec_json)
        assert spec['implementation_backend'] == 'host_applied'
        assert spec['max_trial_gpu_seconds'] < c.group_allowance
        calls.append(spec)
        if len(calls) == 1:
            return json.dumps({'spec': spec, 'val_score': 0., 'error_message': 'synthetic crash',
                               'wall_time_secs': 1., 'timed_out': False, 'oom': False})
        return execute(spec_json, script_path, *args)
    monkeypatch.setattr(c, 'native_execute', fail_once)
    spec = proposed(c)[0]
    native._LAST_PROPOSED_CANDIDATES = [spec]
    result = c.group([spec])[0]
    assert native._is_valid_trial(result) and len(calls) == 2
    execution = read(c.root / 'native/execution/round-0001-slot-0.json')
    assert len(execution['attempts']) == 2
    assert result['wall_time_secs'] > 0
    assert len(read(c.ledger.path)['settled']) == 3


def test_cli_checks_three_splits_without_creating_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(kuai_cli.os, 'sched_getaffinity', lambda _: set(range(64)))
    args = ['check', '--gpus', '4,5,6,7', '--output', str(tmp_path / 'output')]
    for index, name in enumerate(['train', 'val', 'test']):
        path = tmp_path / (name + '.npy')
        np.save(path, np.full((2, 53), index, dtype=float))
        args += ['--' + name + '-data', str(path)]
    kuai_cli.main(args)
    result = json.loads(capsys.readouterr().out)
    assert result['valid'] and result['config']['llm_model'] == 'gpt-6.1-sol'
    assert result['config']['llm_effort'] == 'medium'
    assert len(result['config']['data_files']) == 3
    assert not (tmp_path / 'output').exists()
    args[args.index('--val-data') + 1] = str(tmp_path / 'test.npy')
    with pytest.raises(SystemExit):
        kuai_cli.main(args)


@pytest.mark.parametrize('preflight', [False, True])
def test_host_loop_finishes_and_restores_execution_hook(controller, monkeypatch, preflight):
    c = controller
    monkeypatch.setattr(c, 'setup', lambda: None)
    propose = native.propose_action_group
    calls = []
    def two_groups(*args, **kwargs):
        calls.append(True)
        return propose(*args, **kwargs) if len(calls) <= 2 else '[]'
    monkeypatch.setattr(native, 'propose_action_group', two_groups)
    result = c.run(preflight=preflight)
    assert native.execute_trial == c.native_execute
    assert c.call_index == 5
    if preflight:
        assert result is None and read(c.root / 'preflight.json')['passed']
        assert not (c.root / 'native/logs/final_eval/attempt.json').exists()
        assert not (c.root / 'summary.json').exists()
    else:
        assert result['status'] == 'complete' and result['test']['ok']
        assert read(c.root / 'summary.json')['test']['test_score'] == .51
