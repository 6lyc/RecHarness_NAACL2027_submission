"""Audit one independently trained model per dataset and its registered family."""
import argparse
import ast
from collections import defaultdict
import importlib.util
import os
from pathlib import Path
import pickle
import time
from .core import read, save
from .paths import DATASETS, sha
def model_audit(trial, data):
    """Inspect the actual predictor, metadata and CPU checkpoint-loading path."""
    predictor = trial / 'working/predict.py'
    source = predictor.read_text()
    tree = ast.parse(source)
    names = {n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    banned = {'rank_fusion', 'load_ensemble', 'fused_scores', 'score_members',
              'rankensemble', 'ensemblepredictionmodel'}
    multi = sorted(names & banned)
    if multi:
        return {'eligible': False, 'reason': 'multiple independently trained model scores are combined',
                'evidence_names': multi}
    if not (trial / 'train.py').exists():
        return {'eligible': False, 'reason': 'score ensemble of other trial predictors'}
    # Fixed transition/count scorers combined at the score level are excluded.
    # Jointly learned features/branches of one trained network are allowed.
    for path in sorted((trial / 'working').glob('*meta.pkl')):
        meta = pickle.loads(path.read_bytes())
        for key in ['graph_weight', 'transition_weight', 'transition_alpha']:
            value = meta.get(key, 0.)
            if isinstance(value, (int, float)) and value != 0:
                return {'eligible': False, 'reason': 'neural score fused with an independently fitted transition scorer',
                        'evidence': {'metadata': path.name, key: value}}
        if meta.get('residual_mode') in {'transition', 'popularity'} and 'transition_table' in meta:
            return {'eligible': False, 'reason': 'neural score fused with a separately fitted transition/popularity scorer',
                    'evidence': {'metadata': path.name, 'residual_mode': meta['residual_mode']}}
    import torch
    torch.set_num_threads(4)
    loaded = defaultdict(set)
    models = defaultdict(list)
    last_file = [None]
    original_load = torch.load
    original_state = torch.nn.Module.load_state_dict
    def dataset_of(path):
        matches = [name for name in DATASETS if name in str(path)]
        return matches[0] if len(matches) == 1 else 'unrouted'
    def load(path, *args, **kwargs):
        value = original_load(path, *args, **kwargs)
        last_file[0] = str(path)
        loaded[dataset_of(path)].add(str(Path(path).resolve()))
        return value
    def state(model, *args, **kwargs):
        models[dataset_of(last_file[0])].append({'class': type(model).__name__, 'identity': id(model)})
        return original_state(model, *args, **kwargs)
    torch.load = load
    torch.nn.Module.load_state_dict = state
    try:
        spec = importlib.util.spec_from_file_location('_single_model_predictor', predictor)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for dataset in DATASETS:
            with (data / 'trainval' / (dataset + '_train.txt')).open() as stream:
                first = stream.readline().split()
                user = int(first[0])
                history = [int(first[1])]
                for line in stream:
                    uid, item = map(int, line.split())
                    if uid != user:
                        break
                    history.append(item)
            values = module.predict(user, history, list(range(1, 11)), dataset_name=dataset)
            if len(values) != 10 or not torch.isfinite(torch.as_tensor(values)).all():
                raise ValueError('Predictor audit returned invalid scores')
    finally:
        torch.load = original_load
        torch.nn.Module.load_state_dict = original_state
    cached = defaultdict(list)
    visited = set()
    independent_fusions = []
    def inspect_cache(value, dataset='unrouted'):
        key = (id(value), dataset)
        if key in visited:
            return
        visited.add(key)
        if isinstance(value, torch.nn.Module):
            cached[dataset].append({'class': type(value).__name__, 'identity': id(value)})
            if getattr(value, 'transition_graph', None) is not None and getattr(value, 'graph_weight', 0.) != 0:
                independent_fusions.append({'dataset': dataset, 'kind': 'independent transition graph score'})
            if getattr(value, 'residual_mode', None) in {'transition', 'popularity'}:
                independent_fusions.append({'dataset': dataset, 'kind': 'independent count-based residual score'})
        elif isinstance(value, dict):
            for name, child in value.items():
                inspect_cache(child, name if name in DATASETS else dataset)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect_cache(child, dataset)
        elif hasattr(value, 'models') and isinstance(value.models, dict):
            inspect_cache(value.models, dataset)
    for name in ['_models', '_MODELS', 'models', '_cache', '_predictor']:
        if name in vars(module):
            inspect_cache(vars(module)[name])
    counts = {name: len(loaded[name]) for name in DATASETS}
    model_counts = {name: max(len({m['identity'] for m in models[name]}),
                              len({m['identity'] for m in cached[name]})) for name in DATASETS}
    result = {'checkpoint_files_per_dataset': counts, 'loaded_models_per_dataset': model_counts,
              'checkpoint_files': {name: sorted(loaded[name]) for name in DATASETS},
              'model_classes': {name: sorted({m['class'] for m in models[name] + cached[name]}) for name in DATASETS}}
    if independent_fusions:
        return {**result, 'eligible': False, 'reason': 'loaded neural model adds an independent fitted scorer',
                'evidence': independent_fusions}
    if loaded['unrouted'] or any(counts[name] == 0 for name in DATASETS):
        return {**result, 'eligible': None, 'reason': 'unrecognized checkpoint routing requires manual review'}
    if any(counts[name] > 1 or model_counts[name] > 1 for name in DATASETS):
        return {**result, 'eligible': False, 'reason': 'inference loads multiple trained checkpoints/models for a category'}
    if any('ensemble' in c.lower() for classes in result['model_classes'].values() for c in classes):
        return {**result, 'eligible': False, 'reason': 'loaded model class explicitly combines independent experts'}
    family = os.environ.get('GAGC_MODEL_FAMILY')
    if family:
        registry = getattr(module, '_models', {})
        for dataset in DATASETS:
            model = registry.get(dataset)
            if not isinstance(model, torch.nn.Module):
                return {**result, 'eligible': False, 'reason': 'family audit requires the per-dataset model registry'}
            children = list(model.modules())
            recurrent = any(isinstance(m, torch.nn.GRU) for m in children)
            attention = any(isinstance(m, torch.nn.MultiheadAttention) for m in children)
            convolution = any(isinstance(m, torch.nn.Conv1d) for m in children)
            encoder = any(isinstance(m, torch.nn.TransformerEncoder) for m in children)
            gated = any(hasattr(m, 'gate') and isinstance(m.gate, torch.nn.Linear) for m in children)
            allowed = {'gru4rec': recurrent, 'nextitnet': convolution and not recurrent and not attention,
                       'bert4rec': encoder and getattr(model, 'mask_token', 0) > getattr(model, 'itemnum', 0),
                       'hstu': attention and gated,
                       'sasrec': attention and not recurrent and not convolution and not encoder}
            if not allowed.get(family, False):
                return {**result, 'eligible': False, 'reason': f'candidate escaped registered model family {family}',
                        'family_evidence': {'dataset': dataset, 'GRU': recurrent, 'attention': attention,
                                            'convolution': convolution, 'encoder': encoder, 'gate': gated}}
        result['registered_family'] = family
    return {**result, 'eligible': True, 'reason': 'one trained model/checkpoint per dataset; no independent score fusion',
            'predictor_sha256': sha(predictor), 'training_source_sha256': sha(trial / 'train.py')}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = model_audit(args.audit, args.data)
    result['predictor_sha256'] = sha(args.audit / 'working/predict.py')
    result['source_files'] = {str(p.relative_to(args.audit)): sha(p) for p in args.audit.rglob('*.py') if '__pycache__' not in p.parts}
    result['checkpoint_sha256'] = {name: sha(Path(name)) for paths in result.get('checkpoint_files', {}).values() for name in paths}
    result['auditor_sha256'] = sha(Path(__file__))
    result['audited_at'] = time.time()
    save(args.output, result)

if __name__ == '__main__':
    main()
