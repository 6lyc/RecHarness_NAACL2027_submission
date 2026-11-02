"""Shared Codex settings for every experiment entrypoint."""
import json
import math

DEFAULT_MODEL = 'gpt-6.1-sol'
DEFAULT_EFFORT = 'medium'
DEFAULT_TIMEOUT = 1200


def add_arguments(parser):
    parser.add_argument('--llm-model', default=DEFAULT_MODEL)
    parser.add_argument('--llm-effort', choices=['low', 'medium', 'high'], default=DEFAULT_EFFORT)
    parser.add_argument('--llm-timeout', type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument('--print-llm-config', action='store_true',
                        help='Print effective model settings without starting a run')


def configuration(args):
    if not args.llm_model.strip():
        raise ValueError('LLM model must not be empty')
    if not math.isfinite(args.llm_timeout) or args.llm_timeout <= 0:
        raise ValueError('LLM timeout must be positive and finite')
    return {'llm_transport': 'codex_cli', 'llm_model': args.llm_model,
            'llm_effort': args.llm_effort, 'llm_timeout': args.llm_timeout,
            'llm_attempts': 3, 'retry_backoff_seconds': [5, 15],
            'global_llm_concurrency': 4}


def print_configuration(args):
    settings = configuration(args)
    print(json.dumps({**settings, 'workloads': ['hypotheses', 'implementation']}, indent=2))
