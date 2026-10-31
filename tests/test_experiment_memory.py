import json
from pathlib import Path

import pytest

from gagc import experiment_memory as memory, tools
from gagc.state import ThompsonState
from recharness import runner
from .test_semantic_arms import h, step, controller


def trial(score=.49, baseline=.5):
    return {'spec': {'arm': 'hyperparameter_tuning', 'selected_categories': ['hyperparameter_tuning'],
                     'max_trial_gpu_seconds': 900, 'agent_hypothesis': 'Jointly reduce width and learning rate'},
            'val_score': score, 'val_metrics': {'Movies_and_TV': {'HR@10': score}},
            'wall_time_secs': 800, 'error_message': None, 'timed_out': False, 'oom': False,
            'trial_group_id': 'group', 'trial_id': 0,
            'memory_evidence': {'summary': 'hidden 64 to 32, LR .001 to .0008',
                                'baseline_validation': baseline, 'input_normalized_sha256': memory.normalized_hash('x=1'),
                                'candidate_normalized_sha256': memory.normalized_hash('x=2'), 'basin_id': 'default'}}


def test_local_improvement_below_global_best_is_retained_as_local_evidence():
    result = trial(.57, .55)
    record = memory.digest(result, ThompsonState(), True, .595, True, None)
    assert record['outcome'] == 'improved'
    assert record['delta_vs_input'] == .02
    assert record['delta_vs_prev_best'] == -.025
    assert record['patch_summary'].startswith('hidden 64 to 32')
    assert record['gpu_seconds'] == 800
    assert record['per_dataset_hr10'] == {'Movies_and_TV': .57}


def test_group_winner_below_incumbent_is_negative_evidence_not_promotion(h):
    promotion, state = step(h, [.58, .57], 0, arms=['hyperparameter_tuning']*2)
    assert not promotion['promoted']
    best = state.experiment_memory[0]
    assert best['is_group_winner'] and not best['promoted']
    assert best['outcome'] == 'regressed'
    assert len(state.recent_text_gradients) == 2
    assert all('regressed' in s for s in state.recent_text_gradients)
    assert not state.failure_memory  # performance evidence is not an execution-failure rule


def test_timeout_has_no_validation_delta_or_model_quality_claim(h):
    result = trial(-1.)
    result.update(timed_out=True, error_message='worker limit exceeded')
    record = memory.digest(result, h.store.state(), False, .5, False, 'timeout')
    assert record['val_score'] is None and record['delta_vs_input'] is None
    assert record['outcome'] == 'execution_failure'
    avoid = tools._extract_failure_avoid(record)
    assert 'cap=900' in avoid['avoid_rule']
    assert 'no completed validation comparison' in avoid['avoid_rule']
    assert 'Reduce model size' not in avoid['avoid_rule']


def test_retrieval_keeps_concrete_negative_records_and_excludes_test_metrics():
    state = ThompsonState()
    for i in range(60):
        r = memory.digest(trial(.49, .5), state, False, .5, True, None)
        r.update(round_idx=i, trial_id=i, test_score=123456, test_metrics={'secret': 999},
                 diff_excerpt='x'*4000)
        state.experiment_memory.append(r)
    context = memory.retrieve(state, ['hyperparameter_tuning'], 'x = 1\n')
    assert 1 <= len(context['observations']) <= 12
    assert all(r['outcome'] == 'regressed' and r['same_input_program'] for r in context['observations'])
    assert all(len(r['diff_excerpt']) <= 1600 for r in context['observations'])
    assert 'test_score' not in json.dumps(context) and 'test_metrics' not in json.dumps(context)
    assert memory.normalized_hash('x=1 # comment') == memory.normalized_hash('x = 1\n')


def test_executor_returns_actual_proposal_metadata_for_memory(tmp_path, monkeypatch):
    c = controller(tmp_path)
    path = c.workspace/'best.py'
    path.write_text('hidden = 64\n')
    spec = {'arm': 'hyperparameter_tuning', 'trial_id': 0, 'trial_group_id': 'metadata'}
    def worker(self, spec_json, script_path, timeout_multiplier, slot_id):
        Path(script_path).write_text('hidden = 32\n')
        result = {'spec': json.loads(spec_json), 'val_score': .6}
        runner.save(self.root/'native/execution/round-0000-slot-0.json', {
            'attempts': [{'proposal': {'change_summary': 'hidden 64 to 32', 'category_edits': []}}],
            'receipt': {'status': 'success'}, 'native_result': result})
        return json.dumps(result)
    monkeypatch.setattr(runner.JumpController, 'execute', worker)
    result = json.loads(c.execute(json.dumps(spec), str(path)))
    evidence = result['memory_evidence']
    assert evidence['summary'] == 'hidden 64 to 32'
    assert '-hidden = 64' in evidence['diff_excerpt'] and '+hidden = 32' in evidence['diff_excerpt']
    assert evidence['input_source_sha256'] == memory.source_hash('hidden = 64\n')
    stored = runner.read(c.root/'native/execution/round-0000-slot-0.json')
    assert stored['native_result']['memory_evidence'] == evidence


def test_hypothesis_and_implementation_receive_memory_and_current_program(tmp_path, monkeypatch):
    c = controller(tmp_path)
    (c.workspace/'best.py').write_text('hidden = 64\n')
    state = c.store.state()
    state.provisional_incumbent = {'best_code': 'SECRET_OLD_SNAPSHOT'}
    state.experiment_memory = [memory.digest(trial(), state, False, .5, True, None)]
    c.store.value = {'content': state.to_json()}
    c.data_facts = {'Movies_and_TV': {'users_with_repeated_items': 0}}
    calls = []
    def llm(instruction, *args, **kwargs):
        calls.append(instruction)
        if instruction.startswith('Write one concise'):
            return {'0': 'Reduce hidden size'}
        return {'category_edits': [{'category': 'hyperparameter_tuning', 'rationale': 'reduce width',
                                   'edits': [{'find': 'hidden = 64', 'replace': 'hidden = 32'}]}],
                'change_summary': 'hidden 64 to 32'}
    monkeypatch.setattr(c, 'llm', llm)
    spec = {'arm': 'hyperparameter_tuning', 'selected_categories': ['hyperparameter_tuning']}
    c.hypotheses([spec], {'test_score': 123456})
    c.implement('hidden = 64\n', spec, '', 1, 'test')
    assert len(calls) == 2
    assert all('hidden 64 to 32' in s and 'users_with_repeated_items' in s for s in calls)
    assert 'hidden = 64' in calls[0]
    assert all('SECRET_OLD_SNAPSHOT' not in s and 'test_score' not in s for s in calls)


def test_training_facts_use_only_training_file(tmp_path):
    (tmp_path/'trainval').mkdir()
    (tmp_path/'trainval/A_train.txt').write_text('1 1\n1 2\n2 1\n2 1\n')
    (tmp_path/'trainval/A_valid.txt').write_text('not a readable interaction')
    facts = memory.dataset_facts(tmp_path, ['A'])
    assert facts['A']['training_users'] == 2
    assert facts['A']['users_with_repeated_items'] == 1


def observation(number=0, score=.49, valid=True):
    record = memory.digest(trial(score), ThompsonState(round_idx=number), False, .5,
                           valid, None if valid else 'timeout')
    record.update(phase='regular', promoted=False)
    return record


def test_archive_survives_hot_window_eviction_and_reloads_idempotently(tmp_path):
    store = memory.MemoryStore(tmp_path, 'protocol-A')
    records = [observation(i, .99 if i == 0 else .49) for i in range(70)]
    store.remember(records)
    store.remember(records)
    restored = memory.MemoryStore(tmp_path, 'protocol-A')
    assert len(restored.all_records()) == 70
    state = ThompsonState(experiment_memory=records[-50:])
    context = memory.retrieve(state, ['hyperparameter_tuning'], 'x=1', records=restored.all_records())
    assert any(r['round_idx'] == 0 and r['val_score'] == .99 for r in context['observations'])


def test_archive_rejects_conflicts_and_cross_protocol_import(tmp_path):
    store = memory.MemoryStore(tmp_path/'run-A', 'protocol-A')
    record = observation()
    store.remember([record])
    with pytest.raises(ValueError, match='Conflicting'):
        store.remember([{**record, 'val_score': .7}])
    with pytest.raises(ValueError, match='protocol/schema mismatch'):
        memory.MemoryStore(tmp_path/'run-A', 'protocol-B')
    assert not memory.MemoryStore(tmp_path/'run-B', 'protocol-B').all_records()


def test_context_has_total_character_budget_and_balanced_evidence():
    records = [observation(i, .9 if i == 0 else .49) for i in range(40)]
    for r in records:
        r.update(patch_summary='中文修改'*250, diff_excerpt='补丁'*2000,
                 category_rationales=[{'rationale': '原理'*250}]*4)
    context = memory.retrieve(ThompsonState(experiment_memory=records), ['hyperparameter_tuning'],
                              'x=1', max_chars=7000)
    assert len(json.dumps(context, ensure_ascii=False)) <= 7000
    assert any(r['outcome'] == 'regressed' for r in context['observations'])
    assert any(r['val_score'] == .9 for r in context['observations'])


@pytest.mark.parametrize('outcome', ['accepted', 'rejected', 'rolled_back_incomplete', 'kept_improved_incomplete'])
def test_jump_lifecycle_is_reconciled_without_rewriting_observation(tmp_path, outcome):
    store = memory.MemoryStore(tmp_path, 'P')
    record = {**observation(), 'phase': 'jump', 'jump_window_id': 'basin-1',
              'candidate_source_sha256': memory.source_hash('x=2')}
    store.remember([record])
    keep = outcome in {'accepted', 'kept_improved_incomplete'}
    event = {'jump_window_id': 'basin-1', 'outcome': outcome, 'basin_retained': keep,
             'selected_source_sha256': memory.source_hash('x=2')}
    store.event('window-1', event)
    store.event('window-1', event)
    store.remember([record])
    restored = memory.MemoryStore(tmp_path, 'P').all_records()[0]
    assert restored['jump_outcome'] == outcome
    assert restored['basin_retained'] == restored['selected_at_jump_resolution'] == keep
    immutable = json.loads(next((tmp_path/'observations').glob('*.json')).read_text())['record']
    assert 'jump_outcome' not in immutable


def test_controller_syncs_incomplete_rollback_event(tmp_path):
    c = controller(tmp_path)
    c.memory_store = memory.MemoryStore(c.root/'native/memory', 'P')
    state = c.store.state()
    record = {**observation(), 'phase': 'jump', 'promoted': True}
    state.experiment_memory = [record]
    state.pending_queue = [{'trial_basin_id': 'new-basin'}]
    c.store.value = {'content': state.to_json()}
    (c.workspace/'best.py').write_text('x=2')
    c.sync_memory()
    state = c.store.state()
    state.pending_queue = []
    c.store.value = {'content': state.to_json()}
    (c.workspace/'best.py').write_text('x=1')
    runner.save(c.root/'native/logs/jump/finalization.json',
        {'resolved': [{'trial_basin_id': 'new-basin', 'outcome': 'rolled_back_incomplete',
                       'retune_rounds_completed': 2, 'posterior_update': 'none: incomplete window'}]})
    c.sync_memory()
    c.sync_memory()
    assert c.memory_store.all_records()[0]['jump_outcome'] == 'rolled_back_incomplete'
    assert not c.memory_store.all_records()[0]['basin_retained']


@pytest.mark.parametrize('prior_success', [True, False])
def test_successful_duplicate_is_rejected_but_failed_candidate_can_be_repaired(tmp_path, monkeypatch, prior_success):
    c = controller(tmp_path)
    c.memory_store = memory.MemoryStore(c.root/'native/memory', 'P')
    c.memory_store.remember([observation(valid=prior_success)])
    monkeypatch.setattr(c, 'llm', lambda *a, **k: {
        'category_edits': [{'category': 'hyperparameter_tuning', 'rationale': 'tune x',
                           'edits': [{'find': 'x=1', 'replace': 'x = 2 # same AST'}]}],
        'change_summary': 'Tune x'})
    spec = {'arm': 'hyperparameter_tuning', 'selected_categories': ['hyperparameter_tuning']}
    if prior_success:
        with pytest.raises(ValueError, match='already successfully evaluated'):
            c.implement('x=1', spec, '', 1, 'dedup-test')
    else:
        changed, _ = c.implement('x=1', spec, '', 1, 'dedup-test')
        assert 'x = 2' in changed


def test_retry_evidence_preserves_failed_attempt_before_success():
    result = trial(.6)
    execution = {'attempts': [{'attempt': 1, 'proposal': {'change_summary': 'initial'}},
                              {'attempt': 1, 'error': 'shape mismatch'},
                              {'attempt': 2, 'proposal': {'change_summary': 'repair shape'}}],
                 'training_attempts': [{'id': 'attempt-1', 'status': 'crash', 'allocated_gpu_seconds': 10},
                                       {'id': 'attempt-2', 'status': 'success', 'allocated_gpu_seconds': 20}],
                 'receipt': {'status': 'success'}}
    evidence = memory.capture(result, execution, 'x=1', 'x=2', .5, 'default')
    assert evidence['attempt_count'] == 2 and evidence['training_attempt_count'] == 2
    assert evidence['training_attempts'][0]['status'] == 'crash'
    assert evidence['repair_errors'] == ['shape mismatch']
    assert 'initial' in evidence['summary'] and 'repair shape' in evidence['summary']
    assert [p['attempt'] for p in evidence['proposal_history']] == [1, 2]


def test_memory_extracts_actual_parameter_expressions_and_changed_methods():
    before = 'HIDDEN = 64\nLR = 0.001\nclass Model:\n def encode(self): return 1\n'
    after = 'HIDDEN = 32\nLR = 0.0008\nclass Model:\n def encode(self): return 2\n'
    changes = memory.code_changes(before, after)
    assert changes['parameter_changes'][0] == {'name': 'HIDDEN', 'before_expression': '64', 'after_expression': '32'}
    assert 'Model.encode' in changes['changed_definitions']


def test_memory_source_snapshots_and_corruption_detection(tmp_path):
    store = memory.MemoryStore(tmp_path, 'P')
    source = 'x = 1\n'
    path = store.save_program(source)
    assert store.save_program(source) == path and path.read_text() == source
    path.write_text('corrupt')
    with pytest.raises(ValueError, match='source hash'):
        store.save_program(source)
    store.remember([observation()])
    path = next((tmp_path/'observations').glob('*.json'))
    payload = json.loads(path.read_text())
    payload['record']['val_score'] = .99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='checksum mismatch'):
        memory.MemoryStore(tmp_path, 'P')


def test_identical_checkpoints_are_evidence_not_prediction_equivalence(tmp_path):
    store = memory.MemoryStore(tmp_path, 'P')
    first = {**observation(), 'checkpoint_fingerprints': {'A.pt': '123', 'B.pt': '456'}}
    second = {**observation(1), 'checkpoint_fingerprints': first['checkpoint_fingerprints']}
    store.remember([first, second])
    store.remember([first, second])
    record = store.all_records()[1]
    assert record['identical_checkpoints_as'] == memory.record_key(first)


def test_full_controller_memory_flow_with_budget_ledger_and_jump_rollback(tmp_path, monkeypatch):
    """Real controller/native code, fake LLM and training workers; no GPU/API calls."""
    from .helpers import config
    for name in ['_STORE', '_BENCHMARK_MODE', '_ROUTING_POLICY', 'BASIN_TRANSFER_RHO', '_SELECTION_METRIC',
                 '_DATA_DIR', '_TEST_DIR', '_WORKSPACE_ROOT', '_LOGS_ROOT', '_SERVER_CONFIG',
                 '_LAST_PROPOSED_CANDIDATES', '_LAST_TRIAL_RESULTS', '_LAST_SELECTION_LOG',
                 '_LAST_PROMOTION_RESULT', '_LAST_TRIAL_GROUP_ID', '_trial_group_wall_time_secs', 'execute_trial']:
        monkeypatch.setattr(tools, name, getattr(tools, name))
    data = tmp_path/'data'
    (data/'trainval').mkdir(parents=True)
    (data/'prepared.json').write_text('{}')
    cfg = config(tmp_path/'run')
    cfg['data_dir'] = str(data)
    for name in cfg['alignment']['datasets']:
        (data/'trainval'/f'{name}_train.txt').write_text('1 1\n1 2\n')
    c = runner.FamilyJumpController(cfg)
    monkeypatch.setattr(c, 'baseline', lambda: 'a = 0\nb = 0\nc = 0\nd = 0\n')
    counter = [0]
    def llm(instruction, output_schema, **kwargs):
        if instruction.startswith('Write one concise'):
            return {k: 'Synthetic mutation for memory integration' for k in output_schema['properties']}
        spec = json.loads(instruction.split('SPEC:\n')[1].split('\nREPAIR FEEDBACK:')[0])
        source = instruction.split('\nCURRENT TRAIN.PY:\n')[1]
        counter[0] += 1
        entries = []
        for j, category in enumerate(spec['selected_categories']):
            variable = 'abcd'[j]
            old = next(s for s in source.splitlines() if s.startswith(variable+' ='))
            entries.append({'category': category, 'rationale': 'Synthetic independent edit',
                            'edits': [{'find': old, 'replace': f'{variable} = {counter[0]}'}]})
        return {'category_edits': entries, 'change_summary': f'Synthetic program {counter[0]}'}
    monkeypatch.setattr(c, 'llm', llm)
    def worker(key, source, gpu, requested=None, **kwargs):
        c.ledger.reserve(key, requested, c.c['minimum_trial_seconds'])
        c.ledger.settle(key, 100.)
        directory = c.root/'trials'/key
        (directory/'working').mkdir(parents=True)
        (directory/'train.py').write_text(source)
        (directory/'working/predict.py').write_text('predict = 1\n')
        score = .5 if key.startswith('round-0000-') else .45 if '-0002-' in key else .44
        receipt = {'id': key, 'output': str(directory), 'status': 'success', 'score': score,
                   'allocated_gpu_seconds': 100., 'source_sha256': memory.source_hash(source),
                   'per_dataset': {'A': {'metrics': {'HR@10': score}}}}
        runner.save(directory/'receipt.json', receipt)
        if c.best is None or score > c.best['score']:
            c.best = receipt
        return receipt
    monkeypatch.setattr(c, 'trial', worker)
    c.setup()
    for _ in range(2):
        specs = json.loads(tools.propose_action_group(c.store.state().to_json()))
        c.group(specs)
    assert len(c.memory_store.all_records()) == 5
    assert any(r['outcome'] == 'regressed' for r in c.memory_store.all_records())
    state = c.store.state()
    state.basin_score_history[state.current_basin_id] = [.5]*3
    c.store.put(('gagc', 'thompson'), '/state.json', {'content': state.to_json()})
    specs = json.loads(tools.propose_action_group(state.to_json()))
    assert len(specs) == 4 and all(s['is_jumping'] for s in specs)
    c.group(specs)
    assert len(c.store.state().pending_queue) == 1
    for _ in range(4):
        specs = json.loads(tools.propose_action_group(c.store.state().to_json()))
        c.group(specs)
    state = c.store.state()
    assert not state.pending_queue and state.current_basin_id == 'default'
    assert state.current_basin_score == .5
    records = c.memory_store.all_records()
    window = [r for r in records if r.get('jump_window_id')]
    assert window and all(r['jump_outcome'] == 'rejected' for r in window)
    assert all(not r['basin_retained'] for r in window)
    assert state.global_budget == c.ledger.available() == 43200 - len(records)*100
    context = c.memory_context(['hyperparameter_tuning'], (c.workspace/'best.py').read_text(), 'final-check')
    assert len(json.dumps(context, ensure_ascii=False)) <= memory.CONTEXT_CHAR_LIMIT
    assert len(list((c.root/'native/memory/contexts').glob('*.json'))) > 1
    assert list((c.root/'native/memory/programs').glob('*.py'))
