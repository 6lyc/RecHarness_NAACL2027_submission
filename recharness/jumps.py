"""Budgeted search with safe final jump resolution."""
import json
from .core import BudgetExhausted, read, save
from .controller import RecHarnessController, native, digest
class JumpController(RecHarnessController):
    def group(self, specs, diagnostics=None):
        original_limit = self.c['trial_timeout_seconds']
        caps = [float(s['max_trial_gpu_seconds']) for s in specs if s.get('max_trial_gpu_seconds') is not None]
        if caps:
            self.c['trial_timeout_seconds'] = min(original_limit, min(caps))
        try:
            return super().group(specs, diagnostics)
        finally:
            self.c['trial_timeout_seconds'] = original_limit

    def run(self):
        self.setup()
        diagnostics = {}
        for _ in range(1000):
            previous = self.store.state()
            specs = json.loads(native.propose_action_group(previous.to_json(), json.dumps(diagnostics)))
            if not specs:
                self.event('search_budget_stop', reason='no budget-safe proposal')
                break
            try:
                results = self.group(specs, diagnostics)
            except BudgetExhausted as exc:
                self.event('search_budget_stop', reason=str(exc))
                break
            valid = [r for r in results if native._is_valid_trial(r)]
            diagnostics = max(valid, key=lambda r: r['val_score']) if valid else results[0]
            if not previous.baseline_done and not valid:
                raise RuntimeError('Common baseline failed')
            if self.ledger.available() < self.c['minimum_trial_seconds'] + 3:
                break
        finalized = json.loads(native.finalize_jump_search('matched_budget_stop'))
        if not finalized['ok']:
            raise RuntimeError(finalized['error'])
        selected = digest((self.workspace / 'best.py').read_text())
        if selected not in self.promotion_receipts:
            raise RuntimeError('Final incumbent has no matching trained artifact')
        global_best = self.best
        self.best = self.promotion_receipts[selected]
        history = self.store.state().score_history
        if history and self.best['score'] < max(history) - 1e-12:
            raise RuntimeError('Final model is worse than the retained global validation incumbent')
        save(self.root / 'native/final-selection.json', {'policy': 'safe incumbent after final jump resolution',
            'selected': self.brief(self.best), 'global_best': self.brief(global_best),
            'jump_finalization': finalized, 'state': json.loads(self.store.state().to_json())})
        return self.finish()
