"""Six regular arms and budgeted batches of distinct multi-direction jumps."""
import itertools
import os
import random

REGULAR = {
    'feature_construction': {'label': '特征构造', 'parent': '特征与表征',
        'scope': 'Construct input features solely from supplied training interactions, such as frequency or recency indicators. No external metadata, independent scorer or score fusion. Keep learned embedding design, registered model topology, readout, loss and optimizer fixed.'},
    'embedding_representation': {'label': '嵌入表征', 'parent': '特征与表征',
        'scope': 'Item embedding dimensions, parameterization, tying and embedding normalization. Keep feature definitions, interaction/scoring mechanism, registered model topology, history policy, loss and optimizer fixed.'},
    'architecture_interaction': {'label': '架构与交互', 'parent': '交互与架构',
        'scope': 'Coordinate backbone-appropriate blocks (GRU recurrence, NextItNet causal convolution, BERT4Rec masked bidirectional attention, or HSTU gated attention), feed-forward structure, residual paths and jointly trained interaction/scoring formulas such as bilinear compatibility. Keep the registered backbone family, input features, embedding parameterization, sequence selection/readout, loss and optimizer fixed. No independent scorer fusion.'},
    'sequence_interest': {'label': '序列与兴趣', 'parent': '序列与兴趣',
        'scope': 'Coordinate observed-history truncation, position representation and readout of encoder states, including recency or attention pooling. Keep input feature definitions, embedding parameterization, backbone topology, scoring formula, loss and optimizer fixed. Never introduce future interactions or change supplied splits.'},
    'learning_objective': {'label': '学习目标', 'parent': '目标与训练',
        'scope': 'Ranking/prediction loss, loss weighting and jointly trained auxiliary objectives. Preserve feature definitions, embeddings, registered model topology, scoring/readout, optimizer, sampling, scheduler and batch/epoch settings.'},
    'hyperparameter_tuning': {'label': '联合调参', 'parent': '目标与训练',
        'scope': 'Jointly tune multiple numeric/categorical hyperparameters of the existing program: embedding/hidden dimensions, backbone depth, attention heads where applicable, convolution kernel size where applicable, feed-forward width, max history length, learning rate, dropout, weight decay, batch size, optimizer choice and scheduler settings. A candidate may change any compatible subset together; explain their coupling. Keep the current feature definitions, embedding parameterization/tying, interaction formula, readout formula, loss definition and registered model family. Do not add new computational modules or change data splits, seed 42, epoch cap 200, batches per epoch 200 or GPU/CPU allocation.'},
}
VERSION = 'semantic_v4'
JUMP = 'combination_jump'
# One delayed-outcome counter belongs to the jump mechanism. Candidate combinations
# are not extra bandit arms with unlearned priors.
PAIRS = {JUMP: [JUMP]}
ALL = list(REGULAR) + [JUMP]


def enabled():
    value = os.getenv('GAGC_ARM_DESIGN', VERSION)
    if value not in {VERSION, 'legacy_v2'}:
        raise ValueError('Unknown GAGC_ARM_DESIGN: ' + value)
    return value == VERSION


def categories(arm, selected=None):
    if arm in REGULAR:
        return [arm]
    if arm != JUMP or selected is None:
        return []
    if not 2 <= len(selected) <= 4 or len(set(selected)) != len(selected) or any(v not in REGULAR for v in selected):
        raise ValueError('Jump requires two to four distinct registered directions')
    if len({REGULAR[v]['parent'] for v in selected}) < 2:
        raise ValueError('Jump must span at least two parent categories')
    return list(selected)


def hint(arm, selected=None):
    directions = categories(arm, selected)
    if arm == JUMP and selected is None:
        return 'Combine two to four registered directions spanning at least two parent categories in ONE jointly trained registered model; no independent scorer/model fusion.'
    prefix = ('COMBINATION JUMP: make a substantive, coordinated change in EACH selected direction. '
              if arm == JUMP else 'REGULAR MUTATION: change only this one direction. ')
    scopes = []
    for name in directions:
        scope = REGULAR[name]['scope']
        if arm == JUMP:
            # A regular arm's preservation clauses must not forbid another
            # explicitly selected direction (e.g. feature + loss + tuning).
            scope = scope.split(' Keep ', 1)[0].split(' Preserve ', 1)[0]
        scopes.append(REGULAR[name]['label'] + ': ' + scope)
    return prefix + '\n'.join(scopes) + (
        '\nFor a jump, preservation rules apply to UNSELECTED directions. Shared changes must '
        'have one owner; do not claim the same edit for multiple directions. Keep observed-history data, '
        'registered model family, seed 42, epoch cap 200 and batches per epoch 200 fixed.'
        if arm == JUMP else '') + (
        '\nMaintain shape, checkpoint and predictor consistency for changes in the selected directions; '
        'preserve the registered direction boundaries.')


HINTS = {arm: hint(arm) for arm in ALL}


def sample(state, slots, jump=False, with_trace=False):
    # One independent Beta draw per regular direction per decision. Four GPUs
    # select FOUR of SIX without replacement, leaving budget allocation adaptive.
    pool = list(REGULAR)
    draws = {arm: random.betavariate(state.arms[arm].alpha, state.arms[arm].beta)
             if arm in state.arms else random.betavariate(1., 1.) for arm in pool}
    ranked = sorted(pool, key=lambda arm: draws[arm], reverse=True)
    if jump:
        selected, trace, used = [], [], set()
        for slot in range(max(0, min(slots, 4))):
            size = 2 + slot % 3
            choices = [tuple(c) for c in itertools.combinations(ranked, size)
                       if frozenset(c) not in used and len({REGULAR[a]['parent'] for a in c}) >= 2]
            combination = max(choices, key=lambda c: sum(draws[a] for a in c))
            used.add(frozenset(combination))
            selected.append(JUMP)
            trace.append({'slot': slot, 'arm': JUMP, 'selected_directions': list(combination),
                          'sampled_values': dict(draws), 'direction_count': size,
                          'selection': 'best unused TS-ranked combination at sizes 2,3,4,2'})
    else:
        selected = ranked[:min(slots, len(pool))]
        trace = [{'slot': i, 'rank': i+1, 'arm': arm, 'sampled_values': dict(draws)}
                 for i, arm in enumerate(selected)]
    return (selected, trace) if with_trace else selected


def policy():
    return {'version': VERSION, 'regular_arms': REGULAR,
        'regular_routing': 'one TS draw for each of six arms; select top four without replacement',
        'gpu_budget_allocation': 'selection frequency allocates slots over rounds; equal per-candidate caps within a group; charge actual worker duration',
        'regular_reward': 'Bernoulli success iff valid host validation strictly exceeds the group-start local incumbent',
        'jump_routing': 'budget-safe batch of up to four unique combinations, each with 2-4 directions spanning at least two parents; only best valid candidate opens one retune window',
        'jump_credit': 'one delayed combination_jump outcome counter; no unidentifiable component-direction credit and no separate prior for each pair',
        'jump_direction_counts': [2, 3, 4], 'max_jump_candidates': 4, 'within_run_jump_limit': 1,
        'edit_contract': 'exact selected direction set, per-direction nontrivial AST edits, logged rationale; semantic scope prompt-enforced'}
