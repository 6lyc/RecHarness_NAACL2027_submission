import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gagc import tools
from gagc.benchmarks.kuairec.artifacts import retain_trial


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, '_BENCHMARK_MODE', 'kuairec')
    monkeypatch.setattr(tools, '_GR_TRAIN_DATA', str(tmp_path / 'train.npy'))
    monkeypatch.setattr(tools, '_GR_VAL_DATA', str(tmp_path / 'val.npy'))
    monkeypatch.setattr(tools, '_GR_TEST_DATA', str(tmp_path / 'test.npy'))
    monkeypatch.setattr(tools, '_LOGS_ROOT', str(tmp_path / 'logs'))
    monkeypatch.setattr(tools, '_WORKSPACE_ROOT', str(tmp_path / 'workspace'))
    monkeypatch.setattr(tools, '_SERVER_CONFIG', tools.ServerConfig(num_gpus=0, num_cpus=1))
    monkeypatch.setattr(tools, '_STORE', None)
    monkeypatch.setattr(tools, '_LAST_TRIAL_RESULTS', [])
    monkeypatch.setattr(tools, '_LAST_TRIAL_GROUP_ID', '')
    monkeypatch.setattr(tools, '_LAST_PROMOTION_RESULT', {})


def test_search_environment_contains_validation_and_never_test(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    env, _ = tools._trial_runtime_env(0)
    assert env['GR_VAL_DATA'] == tools._GR_VAL_DATA
    assert env['GR_TEST_DATA'] == tools._GR_VAL_DATA
    assert tools._GR_TEST_DATA not in env.values()
    monkeypatch.setattr(tools, '_GR_VAL_DATA', '')
    with pytest.raises(ValueError, match='independent validation'):
        tools._trial_runtime_env(0)


def test_final_evaluation_uses_retained_weights_once_and_preserves_search_state(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    monkeypatch.setattr(tools, 'finalize_jump_search', lambda *_: '{"ok":true}')
    source = tmp_path / 'workspace/best.py'
    source.parent.mkdir()
    source.write_text('def main():\n    pass\n')
    (source.parent / 'gr_checkpoint.pt').write_bytes(b'frozen validation weights')
    artifact = retain_trial(source, tmp_path / 'retained')
    tools._write_json_log('kuairec_selection.json', artifact)
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b'XAUC=0.6\nMAE=1.2\nWR_XAUC=0.7\nWR_MAE=0.2\n', stderr=b'')

    monkeypatch.setattr(tools.subprocess, 'run', execute)
    before = (tools._LAST_TRIAL_RESULTS, tools._LAST_PROMOTION_RESULT)
    result = json.loads(tools.evaluate_final_incumbent(str(source)))
    assert result['ok'] and result['test_score'] == .6
    command, kwargs = calls[0]
    assert command[1] == str(tmp_path / 'retained/train.py')
    assert kwargs['env']['GR_RUN_MODE'] == 'evaluate'
    assert kwargs['env']['GR_EVAL_DATA'] == tools._GR_TEST_DATA
    assert 'GR_TRAIN_DATA' not in kwargs['env'] and 'GR_VAL_DATA' not in kwargs['env']
    assert before == (tools._LAST_TRIAL_RESULTS, tools._LAST_PROMOTION_RESULT)
    repeated = json.loads(tools.evaluate_final_incumbent(str(source)))
    assert not repeated['ok'] and len(calls) == 1
    with pytest.raises(RuntimeError, match='search is closed'):
        tools.propose_action_group('{}')
    with pytest.raises(RuntimeError, match='search is closed'):
        tools.execute_trial('{}', str(source))


def test_changed_checkpoint_blocks_test_before_execution(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    monkeypatch.setattr(tools, 'finalize_jump_search', lambda *_: '{"ok":true}')
    source = tmp_path / 'best.py'
    source.write_text('pass\n')
    (tmp_path / 'gr_checkpoint.pt').write_bytes(b'original')
    artifact = retain_trial(source, tmp_path / 'retained')
    tools._write_json_log('kuairec_selection.json', artifact)
    (tmp_path / 'retained/gr_checkpoint.pt').write_bytes(b'changed')
    monkeypatch.setattr(tools.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('test must remain unread'))
    assert not json.loads(tools.evaluate_final_incumbent(str(source)))['ok']


def test_group_retains_weights_before_slot_cleanup_and_promotes_matching_artifact(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    source = tmp_path / 'workspace/best.py'
    source.parent.mkdir()
    source.write_text('version = 0\n')

    def execute(spec_json, script_path, *_args):
        Path(script_path).write_text('version = 1\n')
        (Path(script_path).parent / 'gr_checkpoint.pt').write_bytes(b'selected weights')
        return json.dumps({'spec': json.loads(spec_json), 'val_score': .6,
                           'val_metrics': {'WT-MAE': 1.}, 'wall_time_secs': 1.,
                           'timed_out': False, 'oom': False})

    monkeypatch.setattr(tools, '_execute_trial_impl', execute)
    spec = {'dimension': 'tune_lr', 'arm': 'tune_lr', 'delta': 1.,
            'estimated_cost_secs': 10., 'code_diff': ''}
    results = json.loads(tools.execute_trial_group(json.dumps([spec]), str(source)))
    assert len(results) == 1
    artifact = results[0]['kuairec_artifact']
    assert not (source.parent / '_trial_0').exists()
    assert (Path(artifact['directory']) / 'gr_checkpoint.pt').read_bytes() == b'selected weights'
    promoted = json.loads(tools.promote_winner(script_path=str(source)))
    assert promoted['promoted']
    assert source.read_text() == 'version = 1\n'
    selected = json.loads((tmp_path / 'logs/kuairec_selection.json').read_text())
    assert selected == artifact


def test_missing_trained_artifact_cannot_overwrite_incumbent(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    source = tmp_path / 'best.py'
    source.write_text('version = 0\n')
    results = [{'spec': {'arm': 'tune_lr'}, 'val_score': .6,
                'mutated_code_content': 'version = 1\n'}]
    result = json.loads(tools.promote_winner(json.dumps(results), str(source)))
    assert not result['promoted']
    assert source.read_text() == 'version = 0\n'
