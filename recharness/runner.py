"""Six-arm search, combination jumps and within-run evidence memory."""
import ast
from contextlib import contextmanager
import contextvars
import json
import os
from pathlib import Path
import time
from .paths import REPO as FORK, WORKSPACE, sha
from .core import read, save, schema, TEXT
from .jumps import JumpController, native
from . import transport
from .protocol import SINGLE_MODEL_CONTRACT
from gagc import semantic_arms, experiment_memory
class FamilyJumpController(JumpController):
    llm_slot = transport.LLMTransport.llm_slot

    def run_attempt(self, directory, prompt, transport_schema, search):
        try:
            return transport.LLMTransport.run_attempt(self, directory, prompt, transport_schema, search)
        except OSError as exc:
            raise RuntimeError('LLM CLI launch failed: ' + str(exc)) from exc

    def __init__(self, config, smoke=False):
        super().__init__(config, smoke=smoke)
        self.scope = contextvars.ContextVar('family_request_scope', default='baseline')

    @contextmanager
    def request_scope(self, value):
        token = self.scope.set(value)
        try:
            yield
        finally:
            self.scope.reset(token)

    def llm(self, instruction, output_schema=None, search=False, key=None):
        family = self.c['model_family']
        instruction = (f'COLD START AND SEARCH FAMILY: {family}. Preserve this registered backbone family. '
            'GRU4Rec retains a recurrent GRU backbone; NextItNet retains causal convolutions; '
            'BERT4Rec retains bidirectional masked-item modeling with an exclusive mask token; '
            'HSTU retains gated sequential attention. LEFT-padded readout uses the final real position, '
            'never count-1. BERT4Rec must never expose shifted targets through bidirectional input; '
            'retain masked-target training and appended-mask inference. Predictor/checkpoint shapes '
            'must agree. Use seed 42, maximum epochs 200, batches per epoch 200 and the fixed splits. '
            'History contains only observed interactions. One model per category, no score fusion.\n' +
            SINGLE_MODEL_CONTRACT + '\n' + instruction)
        with self.request_scope(key or 'unspecified'):
            while True:
                try:
                    return transport.LLMTransport.llm(
                        self, instruction, output_schema, search, key)
                except transport.TransportUnavailable as exc:
                    # Retry the same pending request, in this still-running native
                    # step. No budget reset, new selection or retraining occurs.
                    self.event('llm_cooldown', key=key, error=str(exc), seconds=60)
                    if self.smoke:
                        raise
                    time.sleep(60)

    def baseline(self):
        path = Path(self.c['base_model']['source'])
        if sha(path) != self.c['base_model']['sha256']:
            raise RuntimeError('Registered family baseline changed')
        return path.read_text()

    def setup(self):
        os.environ['GAGC_ARM_DESIGN'] = 'semantic_v4'
        super().setup()
        os.environ['GAGC_COLD_START'] = 'perdataset'
        state = self.store.state()
        state.global_budget = self.ledger.available()
        state.initial_budget_secs = self.c['search_budget_gpu_seconds']
        state.experiment_skill = ('Use the six registered arms, including joint hyperparameter tuning. '
            'Regular candidates select one arm; the tuning arm may change multiple compatible hyperparameters. A jump candidate jointly changes two to four directions spanning at least two parent categories '
            f'inside one {self.c["model_family"]} model. A batch tests distinct combinations; only its best valid candidate receives four local retune groups and safe rollback. '
            'No additional regular arm, independent model fusion, external data, or held-out feedback.')
        self.store.put(('gagc', 'thompson'), '/state.json', {'content': state.to_json()})
        (self.workspace / 'predict.py').write_text(
            (FORK / 'gagc/templates/perdataset/predict.py').read_text())
        self.data_facts = experiment_memory.dataset_facts(self.c['data_dir'], self.c['alignment']['datasets'])
        save(self.root / 'training-data-facts.json', self.data_facts)
        protocol = {k: self.c['alignment'].get(k) for k in ['datasets', 'preprocessing_seed',
            'training_seed', 'evaluation_seed', 'validation_users_per_dataset', 'negative_candidates',
            'cold_start', 'training_environment_defaults']}
        protocol.update(run_root=str(self.root.resolve()), training_data_facts=self.data_facts,
                        prepared_sha256=self.c['data_manifest_sha256'],
                        grader_sha256=sha(WORKSPACE/'gagc/benchmarks/amazon_reviews/grader.py'),
                        harness_sha256=sha(WORKSPACE/'gagc/benchmarks/amazon_reviews/harness.py'),
                        worker_sha256=sha(WORKSPACE/'recharness/worker.py'))
        protocol_id = experiment_memory.source_hash(json.dumps(protocol, sort_keys=True))
        self.memory_store = experiment_memory.MemoryStore(self.root/'native/memory', protocol_id)
        save(self.root/'native/memory/protocol.json', {'protocol_id': protocol_id, 'protocol': protocol})

    def memory_context(self, categories, source, key):
        store = getattr(self, 'memory_store', None)
        context = experiment_memory.retrieve(self.store.state(), categories, source,
            records=store.all_records() if store else None)
        save(self.root/'native/memory/contexts'/f'{experiment_memory.source_hash(key)}.json',
             {'key': key, 'input_source_sha256': experiment_memory.source_hash(source), 'context': context})
        return context

    def sync_memory(self):
        if not hasattr(self, 'memory_store'):
            return
        state = self.store.state()
        for record in state.experiment_memory:
            if record.get('phase') == 'jump' and record.get('promoted') and not record.get('jump_window_id') and state.pending_queue:
                record['jump_window_id'] = state.pending_queue[0]['trial_basin_id']
        self.memory_store.remember(state.experiment_memory)
        self.store.put(('gagc', 'thompson'), '/state.json', {'content': state.to_json()})
        paths = list((self.root/'native/logs/jump').glob('outcome_*.json'))
        finalization = self.root/'native/logs/jump/finalization.json'
        if finalization.exists():
            paths.append(finalization)
        for path in paths:
            report = read(path)
            outcomes = report.get('resolved', []) if path == finalization else [report]
            for i, outcome in enumerate(outcomes):
                event_id = str(path.relative_to(self.root)) + ':' + str(i)
                if experiment_memory.source_hash(event_id) in self.memory_store.events:
                    continue
                self.memory_store.event(event_id, {
                    'jump_window_id': outcome.get('trial_basin_id'), 'outcome': outcome['outcome'],
                    'basin_retained': outcome['outcome'] in {'accepted', 'kept_improved_incomplete'},
                    'selected_source_sha256': sha(self.workspace/'best.py'),
                    'base_score': outcome.get('base_score'), 'retune_best': outcome.get('retune_best'),
                    'retune_rounds_completed': outcome.get('retune_rounds_completed'),
                    'posterior_update': outcome.get('posterior_update', 'complete window')})

    def group(self, specs, diagnostics=None):
        results = super().group(specs, diagnostics)
        self.sync_memory()
        return results

    def execute(self, spec_json, script_path='/workspace/best.py', timeout_multiplier=1.5, slot_id=0):
        before = Path(script_path).read_text()
        state = self.store.state()
        spec = json.loads(spec_json)
        result = json.loads(super().execute(spec_json, script_path, timeout_multiplier, slot_id))
        path = self.root / 'native/execution' / f"round-{state.round_idx:04d}-slot-{spec['trial_id']}.json"
        execution = read(path)
        key = f"round-{state.round_idx:04d}-slot-{spec['trial_id']}"
        training_attempts = []
        for receipt_path in sorted((self.root/('smoke' if self.smoke else 'trials')).glob(key+'-attempt-*/receipt.json')):
            receipt = read(receipt_path)
            entry = {k: receipt.get(k) for k in ['id', 'status', 'source_sha256', 'allocated_gpu_seconds', 'error', 'score']}
            entry['error'] = str(entry.get('error') or '')[:700]
            training_attempts.append(entry)
        execution['training_attempts'] = training_attempts
        audit_path = Path((execution.get('receipt') or {}).get('output', self.root/'missing'))/'single-model-audit.json'
        if audit_path.exists():
            execution['checkpoint_fingerprints'] = {Path(k).name: v for k, v in read(audit_path).get('checkpoint_sha256', {}).items()}
        result['memory_evidence'] = experiment_memory.capture(result, execution, before,
            Path(script_path).read_text(), state.current_basin_score, state.current_basin_id)
        result['memory_evidence']['phase'] = ('baseline' if spec.get('is_baseline') else
            'jump' if spec.get('is_jumping') else 'retune' if state.pending_queue else 'regular')
        if state.pending_queue:
            result['memory_evidence']['jump_window_id'] = state.pending_queue[0]['trial_basin_id']
        if hasattr(self, 'memory_store'):
            self.memory_store.save_program(before)
            self.memory_store.save_program(Path(script_path).read_text())
        execution['native_result'] = result
        save(path, execution)
        return json.dumps(result)

    def hypotheses(self, specs, diagnostics):
        state = self.store.state()
        source = (self.workspace / 'best.py').read_text()
        selected = [a for s in specs for a in s.get('selected_categories', [s['arm']])]
        reply = self.llm('Write one concise hypothesis for EACH supplied slot, under 800 characters. '
            'The selected category scopes are mandatory. Slots assigned the same category need '
            'different concrete edits. A jump must explain why all selected changes work together. Joint tuning may adjust multiple coupled hyperparameters. '
            'Return hypotheses only; routing and code execution belong to the host.\n' +
            json.dumps({'selected': [{k: v for k, v in s.items() if k != 'gagc_memory_context'} for s in specs],
                        'diagnostics': {k: diagnostics.get(k) for k in ['val_score', 'wall_time_secs', 'timed_out', 'oom', 'error_message']},
                        'state': experiment_memory.state_summary(state),
                        'memory': self.memory_context(selected, source, f'hypotheses-{state.round_idx:04d}'),
                        'training_data_facts': getattr(self, 'data_facts', {}), 'current_train_source': source}, ensure_ascii=False),
            schema({str(i): TEXT for i in range(len(specs))}),
            key=f'hypotheses-{self.store.state().round_idx:04d}')
        if any(len(value) > 800 for value in reply.values()):
            raise ValueError('Native hypothesis limit exceeded')
        return {spec.get('hypothesis_key', f'slot-{i}'): reply[str(i)] for i, spec in enumerate(specs)}

    def implement(self, source, spec, feedback, attempt, key):
        categories = semantic_arms.categories(spec['arm'], spec.get('selected_categories'))
        if not categories or categories != spec.get('selected_categories'):
            raise ValueError('Unregistered semantic mutation scope')
        category_enum = {'type': 'string', 'enum': list(semantic_arms.REGULAR)}
        grouped = schema({'category': category_enum, 'rationale': TEXT,
            'edits': {'type': 'array', 'items': schema({'find': TEXT, 'replace': TEXT})}})
        instruction = ('Implement the supplied semantic-arm mutation using exact find/replace edits '
            'to train.py, including its embedded predictor writer when necessary. Return one '
            'category_edits entry per selected category, and no others. Each entry must make a '
            'substantive change, not a comment or an empty edit. For a jump, ALL selected categories '
            f'must change coherently in this one jointly trained {self.c["model_family"]} model. Shared shape/checkpoint '
            'plumbing may be included under its originating category. Preserve the supplied data '
            'splits, observed histories, seed 42, evaluator, per-dataset routing and the registered model backbone. '
            'Keep epoch cap 200 and batches per epoch 200. Do not inspect files or use shell tools. Do not introduce independent '
            'scorers, ensembles, imported weights, or extra datasets.\nSCOPE:\n' +
            semantic_arms.hint(spec['arm'], categories) + '\nSPEC:\n' + json.dumps({k: v for k, v in spec.items() if k != 'gagc_memory_context'}) +
            '\nREPAIR FEEDBACK:\n' + feedback + '\nATTEMPT: ' + str(attempt) +
            '\nEVIDENCE MEMORY:\n' + json.dumps(self.memory_context(categories, source, key), ensure_ascii=False) +
            '\nTRAINING DATA FACTS:\n' + json.dumps(getattr(self, 'data_facts', {})) +
            '\nCURRENT TRAIN.PY:\n' + source)
        reply = self.llm(instruction, schema({'category_edits': {'type': 'array', 'items': grouped},
                                            'change_summary': TEXT}), key=key)
        entries = reply['category_edits']
        actual = [entry['category'] for entry in entries]
        if len(actual) != len(categories) or set(actual) != set(categories):
            raise ValueError('Edits must cover exactly the selected category set')
        changed = source
        flattened = []
        for entry in entries:
            if not entry['edits'] or not entry['rationale'].strip():
                raise ValueError('Each selected category needs real edits and a rationale')
            before = changed
            changed, error = native._apply_edits(changed, entry['edits'])
            if error:
                raise ValueError(error)
            if changed == before:
                raise ValueError('No-op category edit: ' + entry['category'])
            # Require executable syntax changes for each declared category.
            # This does not claim to prove semantic scope: the source and
            # category rationale are retained for inspection.
            if ast.dump(ast.parse(changed)) == ast.dump(ast.parse(before)):
                raise ValueError('Comment-only category edit: ' + entry['category'])
            flattened.extend(entry['edits'])
        store = getattr(self, 'memory_store', None)
        duplicate = store.duplicate(changed) if store else None
        if duplicate:
            raise ValueError('This program was already successfully evaluated in this run: ' +
                json.dumps({k: duplicate.get(k) for k in ['evidence_id', 'patch_summary', 'val_score', 'gpu_seconds']}) +
                '. Propose a different configuration; no GPU training was started for this duplicate.')
        return changed, {**reply, 'edits': flattened, 'scope_validation': 'exact categories and nontrivial AST changes; semantic labels prompt-enforced'}

    def environment(self, gpu):
        result = super().environment(gpu)
        result['GAGC_MODEL_FAMILY'] = self.c['model_family']
        if self.smoke:
            result.update(SASREC_EPOCHS='5', SASREC_BATCHES='2', SASREC_BATCH='32')
        return result

    def finish(self):
        self.sync_memory()
        # Preserve native jump promotion/rollback selection. Every admitted
        # candidate was already audited inside its charged GPU trial.
        trial = Path(self.best['output'])
        audit = read(trial / 'single-model-audit.json')
        if audit.get('eligible') is not True:
            raise RuntimeError('Native incumbent did not pass the single-model audit')
        if sha(trial / 'working/predict.py') != audit['predictor_sha256']:
            raise RuntimeError('Selected predictor changed after its audit')
        if any(sha(Path(p)) != expected for p, expected in audit['checkpoint_sha256'].items()):
            raise RuntimeError('Selected checkpoint changed after its audit')
        save(self.root / 'single-model-selection.json', {
            'policy': 'native incumbent after jump resolution; verified single model',
            'selected': self.brief(self.best), 'audit': audit})
        return super().finish()
