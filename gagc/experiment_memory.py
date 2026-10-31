"""Bounded, validation-only evidence passed between search steps."""
import ast
from collections import defaultdict
import difflib
import hashlib
import json
import math
from pathlib import Path
import threading

SCHEMA_VERSION = 2
CONTEXT_CHAR_LIMIT = 18000


def source_hash(source):
    return hashlib.sha256(source.encode()).hexdigest()


def normalized_hash(source):
    try:
        source = ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        pass
    return source_hash(source)


def code_changes(before, after):
    """Extract source expressions and changed definitions; do not invent runtime values."""
    def scan(source):
        tree = ast.parse(source)
        parameters, definitions = {}, {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = ast.unparse(target)
                    if isinstance(target, ast.Name) and target.id.isupper() or 'SASREC_' in name:
                        parameters[name] = ast.unparse(node.value)[:240]
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name] = ast.dump(node, include_attributes=False)
                if isinstance(node, ast.ClassDef):
                    definitions.update({node.name+'.'+n.name: ast.dump(n, include_attributes=False)
                                        for n in node.body if isinstance(n, ast.FunctionDef)})
        return parameters, definitions
    try:
        old, old_defs = scan(before)
        new, new_defs = scan(after)
    except SyntaxError:
        return {'parameter_changes': [], 'changed_definitions': [], 'source_parse_failed': True}
    return {'parameter_changes': [{'name': k, 'before_expression': old.get(k), 'after_expression': new.get(k)}
                                  for k in sorted(set(old)|set(new)) if old.get(k) != new.get(k)][:16],
            'changed_definitions': [k for k in sorted(set(old_defs)|set(new_defs)) if old_defs.get(k) != new_defs.get(k)][:16]}


def capture(result, execution, before, after, baseline_score, basin):
    proposals = [a['proposal'] for a in execution.get('attempts', []) if a.get('proposal')]
    last = proposals[-1] if proposals else {}
    spec = result.get('spec') or {}
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), n=2))
    summary = str(last.get('change_summary') or spec.get('agent_hypothesis') or
                  ('Unmodified cold-start baseline' if spec.get('is_baseline') else 'Proposal summary unavailable'))
    if len(proposals) > 1 and proposals[0].get('change_summary') != last.get('change_summary'):
        summary = 'Initial proposal: '+str(proposals[0].get('change_summary', ''))[:600]+'; final revision: '+summary[:500]
    return {**code_changes(before, after), 'summary': summary[:1200],
            'proposal_history': [{'attempt': a.get('attempt'), 'summary': str(a['proposal'].get('change_summary', ''))[:800]}
                                 for a in execution.get('attempts', []) if a.get('proposal')],
            'hypothesis': str(spec.get('agent_hypothesis') or '')[:800],
            'input_source_sha256': source_hash(before), 'candidate_source_sha256': source_hash(after),
            'input_normalized_sha256': normalized_hash(before), 'candidate_normalized_sha256': normalized_hash(after),
            'patch_sha256': source_hash(patch), 'diff_excerpt': patch[:3000],
            'category_rationales': [{'category': e.get('category'), 'rationale': str(e.get('rationale', ''))[:500]}
                                    for e in last.get('category_edits', [])[:4]],
            'baseline_validation': baseline_score, 'basin_id': basin,
            'execution_status': (execution.get('receipt') or {}).get('status', 'implementation_failure'),
            'attempt_count': max(len({a.get('attempt') for a in execution.get('attempts', [])}),
                                 len(execution.get('training_attempts', []))),
            'training_attempt_count': len(execution.get('training_attempts', [])),
            'training_attempts': execution.get('training_attempts', []),
            'checkpoint_fingerprints': execution.get('checkpoint_fingerprints', {}),
            'repair_errors': [str(a['error'])[:500] for a in execution.get('attempts', []) if a.get('error')][-3:]}


def digest(result, state, is_group_winner, prev_best, valid, failure_type):
    spec = result.get('spec') or {}
    evidence = result.get('memory_evidence', {})
    baseline = evidence.get('baseline_validation', prev_best)
    score = float(result.get('val_score', 0)) if valid else None
    delta = round(score - baseline, 6) if score is not None and baseline is not None else None
    outcome = ('improved' if delta is not None and delta > 1e-12 else
               'regressed' if delta is not None and delta < -1e-12 else
               'tied' if delta is not None else 'baseline' if valid else 'execution_failure')
    metrics = result.get('val_metrics') or {}
    summary = evidence.get('summary') or spec.get('agent_hypothesis') or 'Concrete patch metadata unavailable'
    return {'round_idx': state.round_idx, 'trial_group_id': result.get('trial_group_id'),
            'trial_id': result.get('trial_id'), 'arm': spec.get('arm', spec.get('dimension', 'unknown')),
            'selected_categories': spec.get('selected_categories', []),
            'patch_summary': summary, 'benchmark_score': score, 'val_score': score,
            'is_winner': is_group_winner, 'is_group_winner': is_group_winner,
            'valid': valid, 'failure_type': failure_type,
            'delta_vs_prev_best': round(score-prev_best, 6) if score is not None and prev_best is not None else None,
            'delta_vs_input': delta, 'outcome': outcome,
            'val_support': outcome, 'gpu_seconds': float(result.get('wall_time_secs') or 0),
            'trial_cap_seconds': spec.get('max_trial_gpu_seconds'),
            'per_dataset_hr10': {k: v['HR@10'] for k, v in metrics.items() if isinstance(v, dict) and 'HR@10' in v},
            'error': str(result.get('error_message') or '')[:700],
            **evidence}


def lesson(record):
    return (f"Round {record['round_idx']}, {record['arm']}: {record['patch_summary'][:400]}; "
            f"validation={record['val_score']}, delta_vs_input={record['delta_vs_input']}, "
            f"outcome={record['outcome']}, promoted={record.get('promoted', False)}, "
            f"GPU seconds={record['gpu_seconds']:.1f}. "
            + ('Execution failed; this is incomplete performance evidence.' if not record['valid'] else
               'An observation of this exact configuration, not a rule against the whole direction.'))


def retrieve(state, selected, source, limit=12, records=None, max_chars=CONTEXT_CHAR_LIMIT):
    """Include recent, relevant, successful and negative observations; never the test set."""
    records = list(state.experiment_memory if records is None else records)
    limit = min(12, max(0, int(limit)))
    norm = normalized_hash(source)
    selected = set(selected)
    recent = list(reversed(records[-4:]))
    relevant = [r for r in reversed(records) if selected.intersection(r.get('selected_categories') or [r.get('arm')])]
    related = sorted(relevant, key=lambda r: (r.get('input_normalized_sha256') == norm,
                     r.get('basin_id') == state.current_basin_id), reverse=True)
    negative = [r for r in related if r.get('outcome') in {'regressed', 'execution_failure'}]
    successes = sorted([r for r in records if r.get('valid')], key=lambda r: r.get('val_score') or 0, reverse=True)
    duplicates = [r for r in records if r.get('candidate_normalized_sha256') == norm]
    # Interleave evidence types so recency cannot consume the entire allowance.
    lanes = [duplicates[:2], related[:4], negative[:3], successes[:2], recent]
    pool = [(lane[i], label) for i in range(4) for lane, label in zip(lanes,
            ['same_program', 'relevant_arm', 'negative_evidence', 'best_validation', 'recent']) if i < len(lane)]
    seen, result, details = set(), [], []
    fields = ['round_idx', 'trial_group_id', 'trial_id', 'arm', 'selected_categories', 'patch_summary',
              'hypothesis', 'input_source_sha256', 'candidate_source_sha256', 'patch_sha256', 'diff_excerpt',
              'category_rationales', 'baseline_validation', 'basin_id', 'val_score', 'delta_vs_input',
              'delta_vs_prev_best', 'gpu_seconds', 'trial_cap_seconds', 'per_dataset_hr10',
              'outcome', 'valid', 'failure_type', 'error', 'execution_status', 'promoted', 'provisional',
              'is_group_winner', 'same_program_as_trial', 'repair_errors', 'attempt_count', 'training_attempts',
              'training_attempt_count', 'jump_window_id', 'jump_outcome', 'basin_retained', 'selected_at_jump_resolution',
              'jump_base_score', 'jump_retune_best', 'jump_retune_rounds_completed', 'jump_posterior_update',
              'identical_checkpoints_as', 'protocol_id', 'evidence_id', 'phase']
    fields += ['parameter_changes', 'changed_definitions', 'proposal_history']
    result_context = {'schema_version': SCHEMA_VERSION, 'observations': result,
            'guidance': 'Compare deltas within their input program and protocol. Group winner is not promotion. '
            'A single validation observation is configuration-specific evidence, not a general causal rule. '
            'Timeout is incomplete evidence at its recorded cap; repaired attempts do not erase earlier failures. '
            'Jump acceptance includes four retune rounds and gives no isolated component attribution. '
            'Rolled-back basin records remain historical observations, not the current incumbent. '
            'AST identity detects source duplicates, not functional equivalence. Identical checkpoints '
            'alone do not prove identical predictions. Training facts must support proposed features: '
            'zero repeated items means repeat-count features have no variation on the observed histories.'}
    if max_chars < len(json.dumps(result_context, ensure_ascii=False)):
        raise ValueError('Memory character budget is too small for its interpretation rules')
    if not limit:
        return result_context
    for r, reason in pool:
        key = (r.get('round_idx'), r.get('trial_group_id'), r.get('trial_id'))
        if key in seen:
            continue
        seen.add(key)
        item = {k: r[k] for k in fields if k in r}
        item['same_input_program'] = r.get('input_normalized_sha256') == norm
        item['same_basin'] = r.get('basin_id') == state.current_basin_id
        item['retrieval_reason'] = reason
        for field, width in [('patch_summary', 600), ('hypothesis', 400), ('error', 400)]:
            if field in item:
                item[field] = str(item[field])[:width]
        extra = {k: item.pop(k) for k in ['diff_excerpt', 'category_rationales', 'training_attempts', 'repair_errors', 'proposal_history'] if k in item}
        item['earlier_training_failures'] = sum(a.get('status') != 'success' for a in extra.get('training_attempts', []))
        # Full execution records and full diffs remain in the on-disk journal.
        if 'diff_excerpt' in extra:
            extra['diff_excerpt'] = extra['diff_excerpt'][:1600]
        candidate = {**result_context, 'observations': result + [item]}
        if len(json.dumps(candidate, ensure_ascii=False)) > max_chars:
            continue
        result.append(item)
        details.append((item, extra))
        if len(result) == limit:
            break
    # Spend on coverage before verbose code excerpts, then enrich round-robin.
    for field in ['training_attempts', 'repair_errors', 'proposal_history', 'diff_excerpt', 'category_rationales']:
        for item, extra in details:
            if field in extra:
                item[field] = extra[field]
                if len(json.dumps(result_context, ensure_ascii=False)) > max_chars:
                    item.pop(field)
    return result_context


def record_key(record):
    return source_hash(json.dumps([record.get('trial_group_id'), record.get('trial_id'), record.get('round_idx')]))


class MemoryStore:
    """Per-run immutable observations plus idempotent lifecycle events."""
    def __init__(self, root, protocol_id):
        self.root = Path(root)
        self.protocol_id = protocol_id
        self.lock = threading.RLock()
        self.records = {}
        self.events = {}
        for kind, target in [('observations', self.records), ('events', self.events)]:
            for path in sorted((self.root/kind).glob('*.json')):
                payload = json.loads(path.read_text())
                if payload['protocol_id'] != protocol_id or payload['schema_version'] != SCHEMA_VERSION:
                    raise ValueError('Memory protocol/schema mismatch: ' + str(path))
                if payload.get('record_sha256') != source_hash(json.dumps(payload['record'], sort_keys=True)):
                    raise ValueError('Memory evidence checksum mismatch: ' + str(path))
                target[path.stem] = payload['record']

    def _put(self, kind, key, record, target):
        with self.lock:
            if key in target:
                if target[key] != record:
                    raise ValueError('Conflicting immutable memory evidence: ' + key)
                return
            path = self.root/kind/(key+'.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {'schema_version': SCHEMA_VERSION, 'protocol_id': self.protocol_id, 'record': record,
                       'record_sha256': source_hash(json.dumps(record, sort_keys=True))}
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
            temporary.replace(path)
            target[key] = json.loads(json.dumps(record))

    def remember(self, records):
        for item in records:
            # Test fields are never archived as search knowledge.
            record = {k: v for k, v in item.items() if not k.startswith('test') and k not in
                      {'jump_outcome', 'basin_retained', 'selected_at_jump_resolution',
                       'jump_base_score', 'jump_retune_best', 'jump_retune_rounds_completed', 'jump_posterior_update'}}
            record.update(protocol_id=self.protocol_id, evidence_id=record_key(item))
            existing = self.records.get(record['evidence_id'])
            fingerprints = record.get('checkpoint_fingerprints')
            if existing and existing.get('identical_checkpoints_as'):
                record['identical_checkpoints_as'] = existing['identical_checkpoints_as']
            elif fingerprints and not existing:
                match = next((r for r in self.records.values() if r.get('checkpoint_fingerprints') == fingerprints), None)
                if match:
                    record['identical_checkpoints_as'] = match['evidence_id']
            self._put('observations', record['evidence_id'], record, self.records)

    def event(self, event_id, event):
        self._put('events', source_hash(event_id), event, self.events)

    def save_program(self, source):
        with self.lock:
            path = self.root/'programs'/(source_hash(source)+'.py')
            if path.exists():
                if path.read_text() != source:
                    raise ValueError('Stored program content does not match its source hash')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(source)
                temporary.replace(path)
            return path

    def all_records(self):
        with self.lock:
            records = json.loads(json.dumps(list(self.records.values())))
            for event in self.events.values():
                for record in records:
                    if record.get('jump_window_id') and record['jump_window_id'] == event.get('jump_window_id'):
                        record.update(jump_outcome=event['outcome'], basin_retained=event['basin_retained'],
                                      jump_base_score=event.get('base_score'), jump_retune_best=event.get('retune_best'),
                                      jump_retune_rounds_completed=event.get('retune_rounds_completed'),
                                      jump_posterior_update=event.get('posterior_update'),
                                      selected_at_jump_resolution=(event['basin_retained'] and
                                      record.get('candidate_source_sha256') == event.get('selected_source_sha256')))
            return sorted(records, key=lambda r: (r.get('round_idx', 0), str(r.get('trial_group_id')), str(r.get('trial_id'))))

    def duplicate(self, source):
        key = normalized_hash(source)
        return next((r for r in self.all_records() if r.get('valid') and
                     r.get('candidate_normalized_sha256') == key), None)


def state_summary(state):
    return {'round_idx': state.round_idx, 'global_budget': state.global_budget,
            'current_basin_id': state.current_basin_id, 'current_basin_score': state.current_basin_score,
            'global_best_validation': max(state.score_history) if state.score_history else None,
            'pending_retune': [{k: e.get(k) for k in ['base_score', 'retune_best', 'retune_rounds_completed']}
                               for e in state.pending_queue],
            'arm_posteriors': {k: {'alpha': v.alpha, 'beta': v.beta} for k, v in state.arms.items()}}


def dataset_facts(data_dir, datasets):
    facts = {}
    for name in datasets:
        histories = defaultdict(list)
        with (Path(data_dir) / 'trainval' / f'{name}_train.txt').open() as stream:
            for line in stream:
                user, item = map(int, line.split())
                histories[user].append(item)
        lengths = sorted(map(len, histories.values()))
        facts[name] = {'training_users': len(histories), 'training_interactions': sum(lengths),
                       'users_with_repeated_items': sum(len(h) != len(set(h)) for h in histories.values()),
                       'history_median': lengths[len(lengths)//2] if lengths else 0,
                       'history_p95': lengths[max(0, math.ceil(len(lengths)*.95)-1)] if lengths else 0,
                       'users_with_history_gt128': sum(n > 128 for n in lengths)}
    return facts
