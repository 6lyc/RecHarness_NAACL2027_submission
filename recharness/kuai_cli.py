"""Check, preflight or run KuaiRec with the common RecHarness LLM configuration."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import time

from . import llm_config
from .cli import code_manifest, integers
from .core import save
from .paths import sha


def fingerprint(config):
    return {name: sha(Path(config[name + '_data'])) for name in ['train', 'val', 'test']}


def configure(args):
    from gagc.benchmarks.kuairec.artifacts import validate_splits
    if len(args.gpus) != 4 or len(args.cpu_cores) != 64:
        raise ValueError('Provide four distinct GPU IDs and 64 CPU cores')
    if not set(args.cpu_cores) <= os.sched_getaffinity(0):
        raise ValueError('Requested CPU cores are outside this process allocation')
    if any(not math.isfinite(value) for value in
           [args.budget, args.trial_timeout, args.trial_secs, args.rho]):
        raise ValueError('Budget, trial limits and rho must be finite')
    if args.budget < 1200 or args.trial_timeout < 300 or args.trial_secs < 300:
        raise ValueError('Budget must be >=1200s and trial limits >=300s')
    if not 0 <= args.rho <= 1:
        raise ValueError('rho must be between zero and one')
    train, val, test = validate_splits(args.train_data, args.val_data, args.test_data)
    config = {'root': str(args.output.resolve()), 'train_data': train,
              'val_data': val, 'test_data': test, 'cold_start': args.cold_start,
              'gpus': args.gpus, 'cpu_cores': args.cpu_cores,
              'search_budget_gpu_seconds': args.budget,
              'trial_floor_reference_seconds': args.trial_secs,
              'trial_timeout_seconds': args.trial_timeout,
              'minimum_trial_seconds': 300, 'final_timeout_seconds': 12800,
              'basin_transfer_rho': args.rho, **llm_config.configuration(args),
              'llm_slot_directory': str(args.output.resolve() / 'llm-slots')}
    config['data_files'] = fingerprint(config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['check', 'preflight', 'run'], default='run')
    for name in ['train-data', 'val-data', 'test-data', 'output']:
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--gpus', type=integers)
    parser.add_argument('--cpu-cores', type=integers, default=sorted(os.sched_getaffinity(0))[:64])
    parser.add_argument('--budget', type=float, default=43200)
    parser.add_argument('--trial-secs', type=float, default=6400)
    parser.add_argument('--trial-timeout', type=float, default=9600)
    parser.add_argument('--cold-start', choices=['gr', 'd2q', 'ks_d2q', 'tpm'], default='gr')
    parser.add_argument('--rho', type=float, default=.25)
    llm_config.add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.print_llm_config:
            llm_config.print_configuration(args)
            return
        if any(getattr(args, name) is None for name in
               ['train_data', 'val_data', 'test_data', 'output', 'gpus']):
            parser.error('--train-data, --val-data, --test-data, --output and --gpus are required')
        config = configure(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == 'check':
        import json
        print(json.dumps({'valid': True, 'config': config}, indent=2))
        return
    output = Path(config['root'])
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    with (output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(output / 'config.json', config)
        frozen = code_manifest()
        save(output / 'frozen.json', {'code': frozen, 'data': config['data_files']})
        save(output / 'status.json', {'status': 'running', 'mode': args.mode, 'started_at': time.time()})
        try:
            from .kuai_runner import KuaiController
            result = KuaiController(config).run(preflight=args.mode == 'preflight')
            if code_manifest() != frozen or fingerprint(config) != config['data_files']:
                raise RuntimeError('Frozen experiment inputs changed during execution')
        except BaseException as exc:
            save(output / 'status.json', {'status': 'failed', 'error': str(exc), 'time': time.time()})
            raise
        save(output / 'status.json', {'status': 'complete', 'mode': args.mode, 'time': time.time()})
        return result


if __name__ == '__main__':
    main()
