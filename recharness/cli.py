"""Portable entrypoint for independent, fresh RecHarness experiments."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .core import read, save
from .paths import DATASETS, REPO, sha

FAMILIES = {
    'sasrec': 'perdataset',
    'gru4rec': 'gru4rec_perdataset',
    'bert4rec': 'bert4rec_perdataset',
    'nextitnet': 'nextitnet_perdataset',
    'hstu': 'hstu_perdataset',
}


def integers(value):
    try:
        result = [int(v) for v in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated integer IDs') from exc
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError('IDs must be distinct nonnegative integers')
    return result


def fingerprint(data):
    files = {}
    for dataset in DATASETS:
        for folder, suffix in [('trainval', '_train.txt'), ('trainval', '_valid.txt'),
                               ('test', '_test.txt')]:
            relative = Path(folder) / (dataset + suffix)
            path = data / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError('Missing or empty prepared split: ' + str(path))
            files[str(relative)] = sha(path)
    return files


def configure(args):
    data = args.data_dir.resolve()
    output = args.output.resolve()
    if len(args.gpus) != 4:
        raise ValueError('The registered protocol requires exactly four GPUs')
    if len(args.cpu_cores) != 64:
        raise ValueError('Provide 64 CPU cores, partitioned into four 16-core slots')
    if not set(args.cpu_cores) <= os.sched_getaffinity(0):
        raise ValueError('Requested CPU cores are outside this process allocation')
    if args.budget < 4 * 300 or args.trial_timeout < 300 or args.llm_timeout <= 0:
        raise ValueError('Budget must be >=1200s, trial timeout >=300s, LLM timeout >0s')
    files = fingerprint(data)
    import hashlib
    manifest_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    baseline = REPO / 'gagc/templates' / FAMILIES[args.family] / 'train.py'
    info = {'source': str(baseline), 'sha256': sha(baseline), 'weights': 'fresh initialization'}
    return {
        'root': str(output), 'data_dir': str(data), 'data_files': files,
        'data_manifest_sha256': manifest_hash,
        'worker_python': str(Path(args.worker_python).resolve()),
        'gpus': args.gpus, 'cpu_cores': args.cpu_cores, 'model_family': args.family,
        'baseline': info, 'base_model': info,
        'search_budget_gpu_seconds': args.budget, 'trial_floor_reference_seconds': 10800,
        'trial_timeout_seconds': args.trial_timeout, 'minimum_trial_seconds': 300,
        'final_timeout_seconds': 1800, 'single_model_search': True,
        'llm_model': args.llm_model, 'llm_effort': args.llm_effort,
        'llm_timeout': args.llm_timeout, 'llm_attempts': 3,
        'retry_backoff_seconds': [5, 15], 'global_llm_concurrency': 4,
        'llm_slot_directory': str(output / 'llm-slots'),
        'alignment': {
            'datasets': DATASETS, 'preprocessing_seed': 0, 'training_seed': 42,
            'evaluation_seed': 42, 'validation_users_per_dataset': 200,
            'test_users_per_dataset': 10000, 'negative_candidates': 99,
            'cold_start': args.family + '_perdataset',
            'training_environment_defaults': {'epochs': 200, 'batches_per_epoch': 200},
            'accounting': 'sum of all worker durations; final test and LLM time reported separately',
            'execution_isolation': 'trusted local subprocesses; protocol isolation, no OS filesystem jail',
        },
        'replication': {'independent_search': True, 'fresh_baseline_weights': True,
                        'cross_run_response_or_memory_reuse': False,
                        'training_seed': 42, 'routing_seed': 42},
    }


def code_manifest():
    paths = [p for p in [REPO / 'pyproject.toml'] if p.is_file()]
    for directory in ['gagc', 'recharness', 'configs']:
        paths.extend(p for p in (REPO / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix in {'.py', '.json', '.toml', '.md'})
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def preflight(config):
    from .runner import FamilyJumpController, native
    controller = FamilyJumpController(config, smoke=True)
    controller.setup()
    baseline = controller.group(json.loads(native.propose_action_group(controller.store.state().to_json())))
    if len(baseline) != 1 or not native._is_valid_trial(baseline[0]):
        raise RuntimeError('Real GPU baseline preflight failed')
    specs = json.loads(native.propose_action_group(controller.store.state().to_json(), json.dumps(baseline[0])))
    results = controller.group(specs, baseline[0])
    if len(results) != 4 or not all(native._is_valid_trial(r) for r in results):
        raise RuntimeError('Real LLM / four-GPU preflight failed; inspect the retained trial logs')
    save(Path(config['root']) / 'preflight.json', {
        'passed': True, 'baseline': baseline, 'mutations': results,
        'gpu_seconds_outside_formal_budget': sum(r['wall_time_secs'] for r in baseline + results),
        'final_test_evaluated': False,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'preflight', 'run'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=list(FAMILIES), default='sasrec')
    parser.add_argument('--gpus', type=integers, required=True)
    parser.add_argument('--cpu-cores', type=integers,
                        default=sorted(os.sched_getaffinity(0))[:64])
    parser.add_argument('--budget', type=float, default=43200)
    parser.add_argument('--trial-timeout', type=float, default=16200)
    parser.add_argument('--worker-python', default=sys.executable)
    parser.add_argument('--llm-model', default='gpt-6.1-sol')
    parser.add_argument('--llm-effort', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--llm-timeout', type=float, default=1200)
    args = parser.parse_args(argv)
    try:
        config = configure(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == 'check':
        print(json.dumps({'valid': True, 'config': config}, indent=2))
        return
    output = Path(config['root'])
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['GAGC_ARM_DESIGN'] = 'semantic_v4'
        os.environ['GAGC_JUMP_ENABLED'] = 'true'
        save(output / 'config.json', config)
        frozen = code_manifest()
        save(output / 'frozen.json', {'code': frozen, 'data': config['data_files']})
        save(output / 'status.json', {'status': 'running', 'mode': args.mode, 'started_at': time.time()})
        try:
            if args.mode == 'preflight':
                preflight(config)
            else:
                from .runner import FamilyJumpController
                FamilyJumpController(config).run()
            if code_manifest() != frozen or fingerprint(Path(config['data_dir'])) != config['data_files']:
                raise RuntimeError('Frozen experiment inputs changed during execution')
        except BaseException as exc:
            save(output / 'status.json', {'status': 'failed', 'error': str(exc), 'time': time.time()})
            raise
        save(output / 'status.json', {'status': 'complete', 'mode': args.mode, 'time': time.time()})


if __name__ == '__main__':
    main()
