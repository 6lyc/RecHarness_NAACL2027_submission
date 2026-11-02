"""Exercise the runner's real request, slot-lock and cache path."""
import fcntl
import os
from pathlib import Path
import sys

import pytest

from recharness import transport
from recharness.core import TEXT, read, schema
from recharness.runner import FamilyJumpController
from .helpers import config


def controller(tmp_path):
    settings = config(tmp_path)
    settings.update(global_llm_concurrency=1, llm_attempts=2)
    return FamilyJumpController(settings, smoke=True)


def assert_slot_available(c):
    with (Path(c.c['llm_slot_directory']) / 'slot-0.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)


def test_first_runner_request_acquires_slot_validates_response_and_caches(tmp_path, monkeypatch):
    c = controller(tmp_path)
    attempts = []

    def execute(directory, prompt, response_schema, search):
        with (Path(c.c['llm_slot_directory']) / 'slot-0.lock').open('a') as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts.append((directory, prompt, response_schema, search))
        return {'text': 'ok'}

    monkeypatch.setattr(c, 'run_attempt', execute)
    output_schema = schema({'text': TEXT})
    assert c.llm('a pending request', output_schema, key='round-1') == {'text': 'ok'}
    assert_slot_available(c)
    assert c.llm('a pending request', output_schema, key='round-1') == {'text': 'ok'}
    assert len(attempts) == 1 and c.call_index == 1
    assert c.scope.get() == 'baseline'
    assert read(c.root / 'calls/00001/response.json') == {'text': 'ok'}


def test_failed_request_releases_slot_before_retry(tmp_path, monkeypatch):
    c = controller(tmp_path)
    calls = []
    waits = []

    def execute(*_args):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('simulated external process failure')
        return {'text': 'ok'}

    def backoff(seconds):
        assert_slot_available(c)
        waits.append(seconds)

    monkeypatch.setattr(c, 'run_attempt', execute)
    monkeypatch.setattr(transport.time, 'sleep', backoff)
    assert c.llm('retry request', schema({'text': TEXT}), key='retry') == {'text': 'ok'}
    assert len(calls) == 2 and len(waits) == 1
    assert_slot_available(c)


def test_first_runner_request_through_an_external_process(tmp_path, monkeypatch):
    c = controller(tmp_path / 'run')
    executable = tmp_path / 'bin/codex'
    executable.parent.mkdir()
    executable.write_text(f'#!{sys.executable}\n' + '''import json
from pathlib import Path
import sys
args = sys.argv[1:]
assert args[args.index('--model') + 1] == 'synthetic-test-model'
assert 'model_reasoning_effort="medium"' in args
assert 'FAMILY: gru4rec' in sys.stdin.read()
Path(args[args.index('-o') + 1]).write_text(json.dumps({'text': 'process-ok'}))
''')
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(executable.parent) + os.pathsep + os.environ.get('PATH', ''))
    assert c.llm('process request', schema({'text': TEXT}), key='process') == {'text': 'process-ok'}
    assert read(c.root / 'calls/00001/receipt.json')['attempts'] == 1
    assert_slot_available(c)
