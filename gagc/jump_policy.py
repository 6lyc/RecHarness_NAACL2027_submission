"""Compute-bounded jump planning; no model training or held-out data access."""
from __future__ import annotations

import math
import os


def enabled() -> bool:
    return os.getenv('GAGC_JUMP_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'on'}


def option(name: str, default: float, low: float, high: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite and in [{low}, {high}]')
    return value


def initialize(state):
    if state.initial_budget_secs <= 0:
        state.initial_budget_secs = state.global_budget
    if state.current_basin_score is None and state.score_history:
        state.current_basin_score = max(state.score_history)
    if state.current_basin_id not in state.basin_score_history and state.score_history:
        state.basin_score_history[state.current_basin_id] = list(state.score_history)


def cost_estimate(state, fallback: float) -> float:
    costs = sorted(float(v) for v in state.recent_trial_costs[-32:] if float(v) > 0 and math.isfinite(float(v)))
    p75 = costs[min(len(costs) - 1, math.ceil(.75 * len(costs)) - 1)] if costs else fallback
    safety = option('GAGC_JUMP_COST_SAFETY', 1.25, 1., 4.)
    floor = option('GAGC_MIN_TRIAL_SECS', 300., 1., 16200.)
    return float(math.ceil(max(floor + 5, p75 * safety + 5)))


def budget_plan(state, fallback: float, slots: int, rounds: int) -> dict:
    initialize(state)
    cap = cost_estimate(state, fallback)
    fraction = option('GAGC_JUMP_RESERVE_FRACTION', .35, 0., .8)
    allowance = min(state.global_budget, state.initial_budget_secs * fraction) if fraction else state.global_budget
    maximum = min(max(1, slots), int(option('GAGC_JUMP_CANDIDATES', 4, 1, 4))) if os.getenv('GAGC_ARM_DESIGN', 'semantic_v4') == 'semantic_v4' else 1
    # Reserve the WHOLE candidate batch and four rounds for its selected winner.
    # Prefer multiple combinations; reduce width rather than shortening training.
    for width in range(maximum, 0, -1):
        for group in range(max(1, slots), 0, -1):
            total = cap * (width + rounds * group)
            if total <= allowance:
                return {'trial_cap_secs': cap, 'jump_group_size': width,
                        'jump_reserved_secs': cap * width,
                        'retune_group_size': group, 'retune_rounds': rounds,
                        'retune_reserved_secs': cap * rounds * group, 'total_reserved_secs': total,
                        'protected_envelope_secs': allowance, 'cost_source': 'p75 measured trial cost * safety + teardown margin'}
    return {'trial_cap_secs': cap, 'jump_group_size': 0, 'retune_group_size': 0, 'retune_rounds': rounds,
            'retune_reserved_secs': 0., 'total_reserved_secs': cap * (1 + rounds),
            'protected_envelope_secs': 0., 'cost_source': 'insufficient budget for even one retune candidate per round'}


def gate(state, fallback, slots, rounds, ceiling, gap, stagnant, jumping_arms):
    initialize(state)
    plan = budget_plan(state, fallback, slots, rounds)
    reasons = []
    maximum = int(option('GAGC_JUMP_MAX_ATTEMPTS', 1, 1, 100))
    if not enabled():
        reasons.append('jump_disabled')
    if state.jump_count >= maximum:
        reasons.append('jump_attempt_limit')
    if state.pending_queue or state.provisional_incumbent:
        reasons.append('pending_retune_active')
    if not jumping_arms:
        reasons.append('no_executable_jump_arm')
    if not plan['retune_group_size']:
        reasons.append('insufficient_measured_retune_budget')
    minimum_stalls = int(option('GAGC_JUMP_STAGNANT_ROUNDS', 2, 1, 20))
    threshold = option('GAGC_JUMP_THETA', .03, 0., 1.)
    plateau = stagnant >= minimum_stalls and gap < threshold
    history = state.basin_score_history.get(state.current_basin_id, state.score_history)
    budget_phase = (os.getenv('GAGC_JUMP_BUDGET_TRIGGER', 'true').lower() in {'1', 'true', 'yes', 'on'}
        and len(history) >= 2 and plan['retune_group_size'] > 0
        and state.global_budget - plan['protected_envelope_secs'] < slots * plan['trial_cap_secs'])
    if not plateau and not budget_phase:
        reasons.append('no_plateau_or_reserved_exploration_phase')
    return {'ceiling': ceiling, 'basin_gap': gap, 'theta': threshold, 'stagnant_rounds': stagnant,
            'min_stagnant_rounds_before_jump': minimum_stalls, 'min_jump_budget': plan['total_reserved_secs'],
            'active_pending_jumps': list(state.pending_queue), 'jump_eligible_strategy_arms': sorted(jumping_arms),
            'jump_gate_open': not reasons, 'jump_suppressed_reason': ';'.join(reasons),
            'jump_trigger': 'reserved_exploration_phase' if budget_phase else 'plateau' if plateau else None,
            'jump_budget_plan': plan}


def allocation(state, requested_slots, fallback, rounds):
    """Return a hard per-trial cap; pending retune envelopes cannot be borrowed."""
    initialize(state)
    if state.pending_queue:
        entry = state.pending_queue[0]
        plan = entry.get('budget_plan', {})
        slots = int(plan.get('retune_group_size', requested_slots))
        cap = float(plan.get('trial_cap_secs', cost_estimate(state, fallback)))
        remaining_rounds = max(0, int(plan.get('retune_rounds', rounds)) - int(entry.get('retune_rounds_completed', 0)))
        required = cap * slots * remaining_rounds
        # External overruns must not silently borrow the rest of the window.
        # Returning no admissible cap makes the controller finalize safely.
        if state.global_budget + 1e-6 < required:
            return max(1, slots), 0., required
        return max(1, slots), cap, cap * slots * max(0, remaining_rounds - 1)
    plan = budget_plan(state, fallback, requested_slots, rounds)
    protect = 0.
    if enabled() and state.jump_count < int(option('GAGC_JUMP_MAX_ATTEMPTS', 1, 1, 100)) and plan['retune_group_size']:
        protect = plan['protected_envelope_secs']
    spendable = max(0., state.global_budget - protect)
    minimum = option('GAGC_MIN_TRIAL_SECS', 300., 1., 16200.) + 5
    slots = min(requested_slots, max(1, int(spendable // minimum)))
    return slots, spendable / max(1, slots), protect
