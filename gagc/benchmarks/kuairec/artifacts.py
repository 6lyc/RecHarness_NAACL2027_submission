"""Validation-selected checkpoints and immutable KuaiRec trial artifacts."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_splits(train, val, test):
    paths = [Path(p).resolve() for p in [train, val, test] if p]
    if len(paths) != 3 or any(not p.is_file() for p in paths):
        raise ValueError('KuaiRec requires existing train, validation and test files')
    if len(set(paths)) != 3 or len({digest(p) for p in paths}) != 3:
        raise ValueError('KuaiRec train, validation and test must be distinct files and contents')
    import numpy as np
    pairs = []
    for path in paths:
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        if data.ndim != 2 or data.shape[1] != 53 or len(data) == 0:
            raise ValueError('KuaiRec splits must be nonempty arrays with the current 53-column schema')
        keys = np.empty(len(data), dtype=[('user', 'i8'), ('item', 'i8')])
        keys['user'], keys['item'] = data[:, 0], data[:, 19]
        pairs.append(np.unique(keys))
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        if len(np.intersect1d(pairs[left], pairs[right], assume_unique=True)):
            raise ValueError('KuaiRec splits contain overlapping user/item interactions')
    return tuple(map(str, paths))


def checkpoint_path(script=None):
    return Path(os.environ.get('GR_CHECKPOINT_PATH') or
                Path(script or __import__('sys').argv[0]).resolve().parent / 'gr_checkpoint.pt')


def save_checkpoint(model, metrics, epoch, extra, script):
    import torch
    if not all(math.isfinite(float(v)) for v in metrics):
        raise ValueError('Cannot select a checkpoint using non-finite validation metrics')
    destination = checkpoint_path(script)
    temporary = destination.with_suffix('.tmp')
    # Checkpoints are trusted local artifacts. Full modules preserve candidate
    # architectures; the host snapshots and hashes their source before test use.
    torch.save({'schema_version': 1, 'model': model, 'extra': extra,
                'validation_metrics': tuple(metrics), 'epoch': epoch,
                'source_sha256': digest(script)}, temporary)
    temporary.replace(destination)


def print_metrics(metrics):
    for name, value in zip(['XAUC', 'MAE', 'WR_XAUC', 'WR_MAE'], metrics):
        print(f'{name}={float(value):.4f}')


def run_frozen(evaluator, script):
    mode = os.environ.get('GR_RUN_MODE', 'train')
    if mode == 'train':
        return False
    if mode != 'evaluate':
        raise ValueError('Unknown GR_RUN_MODE: ' + mode)
    import numpy as np
    import torch
    path = checkpoint_path(script)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('source_sha256') != digest(script):
        raise ValueError('Checkpoint source does not match the frozen training program')
    model = saved['model'].to(device).eval()
    if hasattr(model, 'device'):
        model.device = device
    data = np.load(os.environ['GR_EVAL_DATA'], allow_pickle=False)
    with torch.no_grad():
        metrics = evaluator(model, saved['extra'], data, device)
    print_metrics(metrics)
    return True


def retain_trial(trial_script, archive_dir):
    """Retain trained weights and exact source before the isolated slot is removed."""
    script, destination = Path(trial_script), Path(archive_dir)
    trial = script.parent
    checkpoint = trial / 'gr_checkpoint.pt'
    if not checkpoint.is_file():
        raise ValueError('Successful KuaiRec trial did not save its validation-selected checkpoint')
    destination.mkdir(parents=True, exist_ok=False)
    paths = [(script, Path('train.py')), (checkpoint, Path('gr_checkpoint.pt'))]
    paths.extend((p, p.relative_to(trial)) for p in (trial / 'model').rglob('*.py')
                 if '__pycache__' not in p.parts)
    hashes = {}
    for path, relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = digest(target)
    record = {'directory': str(destination.resolve()), 'files': hashes}
    (destination / 'artifact.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def verify_artifact(record, selected_script):
    directory = Path(record['directory'])
    for name, expected in record['files'].items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Frozen KuaiRec artifact changed: ' + name)
    if digest(selected_script) != record['files']['train.py']:
        raise ValueError('Final incumbent has no matching frozen training artifact')
    return directory
