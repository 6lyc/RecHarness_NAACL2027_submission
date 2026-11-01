"""Exercise jump budgeting and rollback with synthetic scores and source fixtures."""
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from gagc import jump_policy, tools
from gagc.state import ArmState, ThompsonState


class Store:
    def __init__(self, state):
        self.value = {'content': state.to_json()}
    def get(self, namespace, key):
        return SimpleNamespace(value=self.value)
    def put(self, namespace, key, value):
        self.value = value
    def state(self):
        return ThompsonState.from_json(self.value['content'])


@pytest.fixture
def harness(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    for name in ['best.py', 'train.py']:
        (workspace / name).write_text("marker = 'original'\n")
    (workspace / 'predict.py').write_text("marker = 'original predictor'\n")
    state = ThompsonState(arms={a: ArmState(a) for a in tools._ALL_ARMS},
        global_budget=43200, initial_budget_secs=43200, score_history=[.595],
        baseline_done=True, round_idx=3, current_basin_score=.595,
        basin_score_history={'default': [.595]}, recent_trial_costs=[1000.] * 8)
    store = Store(state)
    monkeypatch.setattr(tools, '_STORE', store)
    monkeypatch.setattr(tools, '_LOGS_ROOT', str(tmp_path / 'logs'))
    monkeypatch.setattr(tools, '_WORKSPACE_ROOT', str(workspace))
    monkeypatch.setattr(tools, '_BENCHMARK_MODE', 'amazon_reviews')
    monkeypatch.setattr(tools, '_ROUTING_POLICY', 'thompson')
    monkeypatch.setattr(tools, '_LAST_PROMOTION_RESULT', {})
    monkeypatch.setattr(tools, '_LAST_TRIAL_RESULTS', [])
    monkeypatch.setattr(tools, '_SERVER_CONFIG', tools.ServerConfig(num_gpus=4))
    for name in ['GAGC_JUMP_ENABLED', 'GAGC_JUMP_RESERVE_FRACTION', 'GAGC_JUMP_MAX_ATTEMPTS',
                 'GAGC_JUMP_BUDGET_TRIGGER', 'GAGC_JUMP_STAGNANT_ROUNDS', 'GAGC_JUMP_COST_SAFETY']:
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(store=store, workspace=workspace, log=tmp_path / 'logs')


def round_result(h, score, number, jump=False, failed=False):
    plan = jump_policy.budget_plan(h.store.state(), 10800, 4, 4)
    arm = 'change_loss_function' if jump else 'tune_lr'
    spec = {'arm': arm, 'dimension': arm, 'delta': 1., 'is_jumping': jump,
            'estimated_cost_secs': 1000., 'jump_budget_plan': plan if jump else {}}
    return {'spec': spec, 'val_score': score, 'val_metrics': {}, 'wall_time_secs': 1000.,
        'timed_out': failed, 'oom': False, 'error_message': 'timeout' if failed else None,
        'trial_group_id': f'fixture-{number}', 'trial_id': 0,
        'mutated_code_content': f"marker = 'candidate-{number}'\n",
        'mutated_predict_content': f"marker = 'predictor-{number}'\n"}


def step(h, score, number, jump=False, failed=False):
    result = round_result(h, score, number, jump, failed)
    promoted = json.loads(tools.promote_winner(json.dumps([result]), script_path=str(h.workspace / 'best.py')))
    tools.update_thompson_state(json.dumps([result]), trial_group_id=result['trial_group_id'])
    return promoted, h.store.state()


def test_retune_can_climb_below_global_best_then_exceed_it(harness):
    h = harness
    promoted, state = step(h, .50, 0, jump=True)
    assert promoted['provisional_jump'] and state.provisional_incumbent
    basin = state.current_basin_id
    assert state.current_basin_score == .50 and max(state.score_history) == .595
    for i, score in enumerate([.53, .56, .60, .61], 1):
        promoted, state = step(h, score, i)
        assert promoted['promoted']
        assert state.current_basin_score == score
        assert state.current_basin_id == basin
    assert not state.pending_queue and not state.provisional_incumbent
    assert max(state.score_history) == .61
    assert state.basin_arms['default']['change_loss_function'].alpha == 2
    assert state.global_budget == 43200 - 5000
    assert 'candidate-4' in (h.workspace / 'best.py').read_text()


def test_four_unsuccessful_retune_rounds_restore_all_original_files(harness):
    h = harness
    expected = {p.name: p.read_text() for p in h.workspace.iterdir()}
    step(h, .50, 0, jump=True)
    for i, score in enumerate([.51, .52, .53, .54], 1):
        _, state = step(h, score, i)
    assert state.current_basin_id == 'default'
    assert state.current_basin_score == .595
    assert state.arms['change_loss_function'].beta == 2
    assert not state.pending_queue and not state.provisional_incumbent
    assert {p.name: p.read_text() for p in h.workspace.iterdir()} == expected


def test_local_regression_is_rejected_without_returning_to_global_threshold(harness):
    h = harness
    step(h, .50, 0, jump=True)
    assert step(h, .53, 1)[0]['promoted']
    promotion, state = step(h, .52, 2)
    assert not promotion['promoted'] and promotion['kept_incumbent']
    assert state.current_basin_score == .53
    assert 'candidate-1' in (h.workspace / 'best.py').read_text()


def test_incomplete_window_rolls_back_without_posterior_penalty(harness):
    h = harness
    before = h.store.state().arms['change_loss_function']
    step(h, .50, 0, jump=True)
    step(h, .53, 1)
    budget = h.store.state().global_budget
    report = json.loads(tools.finalize_jump_search('budget_exhausted'))
    state = h.store.state()
    assert report['ok'] and report['resolved'][0]['outcome'] == 'rolled_back_incomplete'
    assert state.global_budget == budget
    assert state.current_basin_score == .595
    assert state.arms['change_loss_function'].alpha == before.alpha
    assert state.arms['change_loss_function'].beta == before.beta
    assert 'original' in (h.workspace / 'best.py').read_text()
    assert json.loads(tools.finalize_jump_search())['resolved'] == []


def test_incomplete_improved_window_keeps_the_best_model(harness):
    h = harness
    step(h, .50, 0, jump=True)
    step(h, .61, 1)
    report = json.loads(tools.finalize_jump_search('budget_exhausted'))
    assert report['ok'] and report['resolved'][0]['outcome'] == 'kept_improved_incomplete'
    assert h.store.state().current_basin_score == .61
    assert 'candidate-1' in (h.workspace / 'best.py').read_text()


def test_rollback_failure_blocks_finalization_and_keeps_pending_state(harness, monkeypatch):
    step(harness, .50, 0, jump=True)
    monkeypatch.setattr(tools, '_restore_provisional_incumbent', lambda _: 'disk error')
    report = json.loads(tools.finalize_jump_search())
    assert not report['ok']
    assert harness.store.state().pending_queue
    assert harness.store.state().provisional_incumbent


def test_budget_plan_reserves_every_retune_trial_and_adapts_group_size(harness):
    state = harness.store.state()
    plan = jump_policy.budget_plan(state, 10800, 4, 4)
    assert 1 <= plan['retune_group_size'] < 4
    assert plan['total_reserved_secs'] == plan['trial_cap_secs'] * (1 + 4 * plan['retune_group_size'])
    assert plan['total_reserved_secs'] <= .35 * state.initial_budget_secs
    state.global_budget = plan['total_reserved_secs'] - 1
    # A smaller retune group may still fit; a full one-candidate plan may not.
    state.global_budget = 4 * plan['trial_cap_secs']
    assert jump_policy.budget_plan(state, 10800, 4, 4)['retune_group_size'] == 0


def test_reserved_exploration_trigger_is_explicit_and_uses_one_jump(harness):
    state = harness.store.state()
    state.global_budget = 16000
    state.score_history = [.50, .55, .595]
    state.basin_score_history = {'default': list(state.score_history)}
    harness.store.value = {'content': state.to_json()}
    specs = json.loads(tools.propose_action_group(state.to_json()))
    assert len(specs) == 1 and specs[0]['is_jumping']
    assert specs[0]['jump_trigger'] == 'reserved_exploration_phase'
    assert specs[0]['arm'] in tools._active_jumping_dims()
    assert specs[0]['jump_budget_plan']['total_reserved_secs'] <= state.global_budget


def test_disabled_ablation_has_no_jump_or_protected_budget(harness, monkeypatch):
    monkeypatch.setenv('GAGC_JUMP_ENABLED', 'false')
    state = harness.store.state()
    state.global_budget = 16000
    specs = json.loads(tools.propose_action_group(state.to_json()))
    assert len(specs) == 4
    assert all(not s['is_jumping'] and s['protected_gpu_seconds'] == 0 for s in specs)


def test_pending_retune_uses_its_reserved_group_and_never_jumps_again(harness):
    step(harness, .50, 0, jump=True)
    state = harness.store.state()
    specs = json.loads(tools.propose_action_group(state.to_json()))
    plan = state.pending_queue[0]['budget_plan']
    assert len(specs) == plan['retune_group_size']
    assert all(not s['is_jumping'] and s['max_trial_gpu_seconds'] == plan['trial_cap_secs'] for s in specs)
    assert sum(s['max_trial_gpu_seconds'] for s in specs) <= plan['retune_reserved_secs']


def test_underfunded_pending_window_stops_instead_of_spending_future_reserve(harness):
    step(harness, .50, 0, jump=True)
    state = harness.store.state()
    state.global_budget = 100
    harness.store.value = {'content': state.to_json()}
    assert json.loads(tools.propose_action_group(state.to_json())) == []
    assert json.loads(tools.finalize_jump_search('external_overrun'))['ok']
    assert harness.store.state().current_basin_score == .595


def test_textual_routing_keeps_jump_lifecycle_without_posterior_updates(harness, monkeypatch):
    monkeypatch.setattr(tools, '_ROUTING_POLICY', 'textual_gradient')
    step(harness, .50, 0, jump=True)
    assert harness.store.state().pending_queue
    for i in range(1, 5):
        step(harness, .50 + i * .01, i)
    state = harness.store.state()
    assert state.current_basin_score == .595 and not state.pending_queue
    assert state.arms['change_loss_function'].alpha == 1
    assert state.arms['change_loss_function'].beta == 1


def test_backend_retry_attempts_share_one_deadline(tmp_path, monkeypatch):
    from gagc import claude_code_backend as backend
    clock = [0.]
    timeouts = []
    monkeypatch.setattr(backend.time, 'monotonic', lambda: clock[0])
    train = tmp_path / 'train.py'
    predict = tmp_path / 'predict.py'
    train.write_text('x=1\n')
    predict.write_text('x=1\n')
    def invoke(command, cwd, env, prompt, timeout):
        timeouts.append(timeout)
        clock[0] += 3.
        artifact = tmp_path / '.gagc_claude'
        (artifact / 'train_returncode.txt').write_text('1\n')
        (artifact / 'train_stdout.log').write_text('')
        (artifact / 'train_stderr.log').write_text('synthetic training failure')
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
    monkeypatch.setattr(backend, '_run_claude_subprocess', invoke)
    outcome = backend.run_claude_code_trial(trial_dir=str(tmp_path), train_path=str(train),
        predict_path=str(predict), spec_data={'claude_attempts': 3}, env={}, hard_timeout=5.,
        benchmark_mode='amazon_reviews', validation_callback=lambda: None)
    assert not outcome.success and outcome.timed_out
    assert timeouts == [5., 2.]


def test_matched_executor_honors_cap_and_restores_config_after_failure(tmp_path, monkeypatch):
    from recharness import jumps as runner
    from .helpers import config as make_config
    config = make_config(tmp_path)
    controller = runner.JumpController(config)
    def group(self, specs, diagnostics=None):
        assert self.c['trial_timeout_seconds'] == 1234
        raise RuntimeError('synthetic executor failure')
    monkeypatch.setattr(runner.RecHarnessController, 'group', group)
    with pytest.raises(RuntimeError, match='synthetic executor'):
        controller.group([{'max_trial_gpu_seconds': 1234}])
    assert controller.c['trial_timeout_seconds'] == config['trial_timeout_seconds'] == 16200


def test_state_roundtrip_and_legacy_fallback(harness):
    state = harness.store.state()
    assert ThompsonState.from_json(state.to_json()).to_json() == state.to_json()
    legacy = ThompsonState.from_json(json.dumps({'score_history': [.5], 'global_budget': 43200}))
    jump_policy.initialize(legacy)
    assert legacy.current_basin_score == .5
    assert legacy.basin_score_history == {'default': [.5]}
    assert legacy.initial_budget_secs == 43200


def test_every_declared_amazon_jump_arm_is_executable(harness):
    assert tools._active_jumping_dims() <= set(tools._active_arms())


def test_group_update_is_idempotent_with_a_pending_jump(harness):
    h = harness
    result = round_result(h, .50, 0, jump=True)
    tools.promote_winner(json.dumps([result]), script_path=str(h.workspace / 'best.py'))
    tools.update_thompson_state(json.dumps([result]), trial_group_id=result['trial_group_id'])
    before = h.store.state().to_json()
    tools.update_thompson_state(json.dumps([result]), trial_group_id=result['trial_group_id'])
    assert h.store.state().to_json() == before
