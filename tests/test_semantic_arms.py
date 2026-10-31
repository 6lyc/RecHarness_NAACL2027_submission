from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gagc import tools, semantic_arms as policy, jump_policy
from gagc.state import ArmState, ThompsonState
from recharness import runner


class Store:
    def __init__(self, state): self.value = {'content': state.to_json()}
    def get(self, namespace, key): return SimpleNamespace(value=deepcopy(self.value))
    def put(self, namespace, key, value): self.value = deepcopy(value)
    def state(self): return ThompsonState.from_json(self.value['content'])


@pytest.fixture
def h(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    for name in ['best.py', 'train.py', 'predict.py']:
        (workspace / name).write_text("marker = 'original'\n")
    state = ThompsonState(arms={a: ArmState(a) for a in policy.ALL}, global_budget=43200,
        initial_budget_secs=43200, score_history=[.595], baseline_done=True, round_idx=3,
        current_basin_score=.595, basin_score_history={'default': [.595]}, recent_trial_costs=[1000.]*8)
    store = Store(state)
    for name, value in {'_STORE': store, '_LOGS_ROOT': str(tmp_path/'logs'),
        '_WORKSPACE_ROOT': str(workspace), '_BENCHMARK_MODE': 'amazon_reviews',
        '_ROUTING_POLICY': 'thompson', '_LAST_PROMOTION_RESULT': {}, '_LAST_TRIAL_RESULTS': [],
        '_SERVER_CONFIG': tools.ServerConfig(num_gpus=4)}.items(): monkeypatch.setattr(tools, name, value)
    monkeypatch.setenv('GAGC_ARM_DESIGN', 'semantic_v4')
    monkeypatch.setenv('GAGC_JUMP_ENABLED', 'true')
    return SimpleNamespace(store=store, workspace=workspace)


def step(h, scores, number, arms=None, jump=False):
    arms = arms or [policy.JUMP if jump else 'hyperparameter_tuning'] * len(scores)
    plan = jump_policy.budget_plan(h.store.state(), 10800, 4, 4)
    results = []
    for i, (arm, score) in enumerate(zip(arms, scores)):
        results.append({'spec': {'arm': arm, 'dimension': arm, 'delta': 1., 'is_jumping': jump,
            'jump_budget_plan': plan if jump else {}}, 'val_score': score, 'val_metrics': {},
            'wall_time_secs': 100., 'timed_out': False, 'oom': False, 'error_message': None,
            'trial_group_id': f'semantic-{number}', 'trial_id': i,
            'mutated_code_content': f"marker = 'candidate-{number}-{i}'\n",
            'mutated_predict_content': f"marker = 'predict-{number}-{i}'\n"})
    promoted = json.loads(tools.promote_winner(json.dumps(results), script_path=str(h.workspace/'best.py')))
    tools.update_thompson_state(json.dumps(results), trial_group_id=f'semantic-{number}')
    return promoted, h.store.state()


def test_six_arms_in_four_parent_categories(h):
    assert len(policy.REGULAR) == 6
    assert len(set(v['parent'] for v in policy.REGULAR.values())) == 4
    assert len(tools._active_exploiting_arms()) == 6
    assert tools._active_jumping_dims() == {policy.JUMP}
    assert len(tools._active_arms()) == 7  # six normal arms and one outcome counter


def test_one_ranking_six_draws_top_four_unique(h, monkeypatch):
    draws = Mock(side_effect=[.1, .8, .7, .6, .4, .5])
    monkeypatch.setattr(policy.random, 'betavariate', draws)
    selected, trace = policy.sample(h.store.state(), 4, with_trace=True)
    assert selected == ['embedding_representation', 'architecture_interaction', 'sequence_interest', 'hyperparameter_tuning']
    assert len(set(selected)) == 4 and draws.call_count == 6
    assert all(len(t['sampled_values']) == 6 for t in trace)


def test_later_round_can_repeat_the_same_four(h, monkeypatch):
    monkeypatch.setattr(policy.random, 'betavariate', Mock(side_effect=list(range(6))*2))
    assert policy.sample(h.store.state(), 4) == policy.sample(h.store.state(), 4)


def test_jump_uses_direction_posterior_and_skips_same_parent(h, monkeypatch):
    monkeypatch.setattr(policy.random, 'betavariate', Mock(side_effect=[.9, .8, .7, .6, .5, .4]))
    selected, trace = policy.sample(h.store.state(), 4, jump=True, with_trace=True)
    assert selected == [policy.JUMP]*4
    assert trace[0]['selected_directions'] == ['feature_construction', 'architecture_interaction']


def test_proposal_caps_and_scopes_are_logged(h):
    specs = json.loads(tools.propose_action_group(h.store.state().to_json()))
    assert len(specs) == 4 and len({s['arm'] for s in specs}) == 4
    assert all(s['selected_categories'] == [s['arm']] for s in specs)
    assert len({s['hypothesis_key'] for s in specs}) == 4
    assert sum(s['max_trial_gpu_seconds'] for s in specs) <= 43200 - .35*43200


def test_budget_gate_returns_one_two_direction_jump(h):
    state = h.store.state()
    state.global_budget = 16000
    state.score_history = [.50, .55, .595]
    state.basin_score_history = {'default': list(state.score_history)}
    h.store.value = {'content': state.to_json()}
    specs = json.loads(tools.propose_action_group(state.to_json()))
    assert len(specs) == specs[0]['jump_budget_plan']['jump_group_size']
    assert all(s['arm'] == policy.JUMP for s in specs)
    dirs = specs[0]['selected_categories']
    assert len(dirs) == 2 and policy.REGULAR[dirs[0]]['parent'] != policy.REGULAR[dirs[1]]['parent']
    assert specs[0]['jump_trigger'] == 'reserved_exploration_phase'


def test_single_trial_retune_gets_credit_for_real_improvement(h):
    _, state = step(h, [.60], 0)
    assert state.arms['hyperparameter_tuning'].alpha == 2
    assert set(state.arms) == set(policy.ALL)


def test_tie_gets_no_success_credit(h):
    _, state = step(h, [.595], 0)
    assert state.arms['hyperparameter_tuning'].alpha == 1
    assert state.arms['hyperparameter_tuning'].beta == 2


@pytest.mark.parametrize('success', [True, False])
def test_combination_delayed_credit_and_complete_window(h, success):
    step(h, [.50], 0, jump=True)
    source = h.store.state().basin_arms['default']
    assert source[policy.JUMP].alpha == source[policy.JUMP].beta == 1
    for i, score in enumerate([.53, .56, .60, .61] if success else [.51, .52, .53, .54], 1):
        _, state = step(h, [score], i)
    assert not state.pending_queue and not state.provisional_incumbent
    source = state.basin_arms['default']
    assert source[policy.JUMP].alpha == (2 if success else 1)
    assert source[policy.JUMP].beta == (1 if success else 2)
    # Retune observations belong to the new basin, never the source components.
    assert all(source[name].alpha == source[name].beta == 1 for name in policy.REGULAR)
    assert state.current_basin_score == (.61 if success else .595)
    assert state.global_budget == 43200 - 500


def controller(tmp_path, run=1):
    from .helpers import config
    c = runner.FamilyJumpController(config(tmp_path))
    c.store = Store(ThompsonState())
    return c


def test_fresh_budget_is_43200_and_existing_spending_is_not_reset(tmp_path):
    c = controller(tmp_path)
    assert c.c['replication']['fresh_baseline_weights']
    assert c.ledger.available() == 43200
    c.ledger.reserve('settled-test', 600)
    c.ledger.settle('settled-test', 5.)
    c2 = controller(tmp_path)
    assert c2.ledger.available() == 43195


@pytest.mark.parametrize('invalid', ['missing_direction', 'wrong_direction', 'comment_only'])
def test_structured_jump_edits_reject_incomplete_scope(tmp_path, monkeypatch, invalid):
    c = controller(tmp_path)
    selected = ['embedding_representation', 'hyperparameter_tuning']
    entries = [{'category': selected[0], 'rationale': 'change dimension', 'edits': [{'find': 'x = 1', 'replace': 'x = 2'}]},
        {'category': selected[1], 'rationale': 'change training', 'edits': [{'find': 'y = 1', 'replace': 'y = 2'}]}]
    if invalid == 'missing_direction': entries.pop()
    if invalid == 'wrong_direction': entries[1]['category'] = 'sequence_interest'
    if invalid == 'comment_only': entries[1]['edits'][0]['replace'] = 'y = 1  # only a comment'
    monkeypatch.setattr(c, 'llm', lambda *a, **k: {'category_edits': entries, 'change_summary': 'test'})
    with pytest.raises(ValueError):
        c.implement('x = 1\ny = 1\n', {'arm': policy.JUMP, 'selected_categories': selected}, '', 1, 'test')


def test_slot_hypotheses_never_overwrite_each_other(h):
    specs = [{'arm': 'hyperparameter_tuning', 'dimension': 'hyperparameter_tuning', 'hypothesis_key': f'slot-{i}'} for i in range(2)]
    prepared, error = tools._prepare_specs_for_execution(json.dumps(specs),
        json.dumps({'slot-0': 'first idea', 'slot-1': 'second idea'}), 'group', str(h.workspace/'best.py'))
    assert error is None
    assert [s['agent_hypothesis'] for s in prepared] == ['first idea', 'second idea']


def test_jump_batch_contains_distinct_two_three_four_direction_combinations(h):
    _, trace = policy.sample(h.store.state(), 4, jump=True, with_trace=True)
    combinations = [t['selected_directions'] for t in trace]
    assert [len(c) for c in combinations] == [2, 3, 4, 2]
    assert len({frozenset(c) for c in combinations}) == 4
    assert all(policy.categories(policy.JUMP, c) == c for c in combinations)


def test_multi_jump_reservation_covers_all_candidates_and_retune(h):
    state = h.store.state()
    state.recent_trial_costs = [1800.] * 8
    plan = jump_policy.budget_plan(state, 10800, 4, 4)
    # At 43,200 seconds the 35% envelope admits two jump candidates
    # and four single-candidate retunes, each capped at 2,255 seconds.
    assert plan['jump_group_size'] == 2
    assert plan['retune_group_size'] == 1
    assert plan['total_reserved_secs'] == plan['trial_cap_secs'] * (2 + 4)
    assert plan['total_reserved_secs'] <= .35 * 43200
    state.pending_queue = [{'budget_plan': plan, 'retune_rounds_completed': 1}]
    state.global_budget = plan['trial_cap_secs'] * 3
    slots, cap, protected = jump_policy.allocation(state, 4, 10800, 4)
    assert slots == 1 and cap == plan['trial_cap_secs']
    assert protected == cap * 2


@pytest.mark.parametrize('scores', [[.51, .55, .60, .54], [.51, None, .60, None], [.51, .54, .55, .52], [None]*4])
def test_parallel_jump_execution_promotes_only_best_and_charges_all(h, monkeypatch, scores):
    state = h.store.state()
    state.basin_score_history = {'default': [.595]*3}
    h.store.value = {'content': state.to_json()}
    specs = json.loads(tools.propose_action_group(state.to_json()))
    assert len(specs) == 4
    executed = []
    def worker(spec_json, script_path, timeout_multiplier, slot_id):
        spec = json.loads(spec_json)
        executed.append(slot_id)
        Path(script_path).write_text(f"marker = 'jump-{slot_id}'\n")
        score = scores[slot_id]
        return json.dumps({'spec': spec, 'val_score': score if score is not None else -1.,
            'val_metrics': {}, 'wall_time_secs': 100., 'timed_out': score is None,
            'oom': False, 'error_message': 'timeout' if score is None else None})
    monkeypatch.setattr(tools, 'execute_trial', worker)
    results = json.loads(tools.execute_trial_group(json.dumps(specs),
        script_path=str(h.workspace/'best.py'),
        hypotheses_json=json.dumps({f'slot-{i}': f'Combination {i}' for i in range(4)})))
    assert len(results) == 4 and sorted(executed) == [0, 1, 2, 3]
    promotion = json.loads(tools.promote_winner(script_path=str(h.workspace/'best.py')))
    tools.update_thompson_state('__LAST_RESULTS__', trial_group_id=results[0]['trial_group_id'])
    state = h.store.state()
    assert state.global_budget == 43200 - 400
    assert state.jump_count == 1
    source = state.basin_arms['default']
    assert all(source[a].alpha == source[a].beta == 1 for a in policy.REGULAR)
    if any(s is not None for s in scores):
        assert promotion['winner_idx'] == 2
        assert len(state.pending_queue) == 1 and state.current_basin_score == scores[2]
        assert (h.workspace/'best.py').read_text() == "marker = 'jump-2'\n"
        for i in range(4):
            _, state = step(h, [.61 if scores[2] > .595 else .54], 10+i)
        assert not state.pending_queue
        source = state.basin_arms['default'][policy.JUMP]
        if scores[2] > .595:
            assert source.alpha == 2
        else:
            assert source.beta == 2
            assert state.current_basin_id == 'default'
            assert (h.workspace/'best.py').read_text() == "marker = 'original'\n"
        assert state.global_budget == 43200 - 800
    else:
        assert not state.pending_queue
        assert state.current_basin_score == .595
        assert source[policy.JUMP].beta == 2


def test_joint_tuning_accepts_coupled_parameters(tmp_path, monkeypatch):
    c = controller(tmp_path)
    entries = [{'category': 'hyperparameter_tuning', 'rationale': 'Tune width and learning rate together',
        'edits': [{'find': 'hidden = 64', 'replace': 'hidden = 32'},
                  {'find': 'lr = 0.001', 'replace': 'lr = 0.0008'}]}]
    monkeypatch.setattr(c, 'llm', lambda *a, **k: {'category_edits': entries, 'change_summary': 'joint tuning'})
    changed, _ = c.implement('hidden = 64\nlr = 0.001\n',
        {'arm': 'hyperparameter_tuning', 'selected_categories': ['hyperparameter_tuning']}, '', 1, 'test')
    assert changed == 'hidden = 32\nlr = 0.0008\n'


def test_jump_scope_does_not_preserve_other_selected_directions():
    hint = policy.hint(policy.JUMP, ['feature_construction', 'learning_objective', 'hyperparameter_tuning'])
    assert 'Preserve feature definitions' not in hint
    assert 'loss and optimizer fixed' not in hint
    assert 'UNSELECTED directions' in hint
