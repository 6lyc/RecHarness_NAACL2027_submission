from pathlib import Path

import pytest

from recharness import runner
from recharness.cli import FAMILIES
from .helpers import config


@pytest.mark.parametrize('family', FAMILIES)
def test_registered_cold_start_is_verified_and_uses_requested_resources(tmp_path, family):
    c = runner.FamilyJumpController(config(tmp_path, family))
    assert c.baseline() == Path(c.c['base_model']['source']).read_text()
    assert len(c.c['gpus']) == 4 and len(c.c['cpu_cores']) == 64
    env = c.environment(c.c['gpus'][2])
    assert env['GAGC_MODEL_FAMILY'] == family
    assert env['RECHARNESS_CPU_CORES'] == ','.join(map(str, range(32, 48)))
    assert c.c['replication']['fresh_baseline_weights']
    assert not c.c['replication']['cross_run_response_or_memory_reuse']


@pytest.mark.parametrize('family', FAMILIES)
def test_transport_retry_preserves_pending_request(tmp_path, monkeypatch, family):
    c = runner.FamilyJumpController(config(tmp_path, family))
    calls = []

    def request(self, instruction, output_schema, search, key):
        calls.append((instruction, output_schema, search, key, self.scope.get()))
        if len(calls) == 1:
            raise runner.transport.TransportUnavailable('fault injection')
        return {'edits': []}

    monkeypatch.setattr(runner.transport.LLMTransport, 'llm', request)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    assert c.llm('a mutation', {}, False, 'round-1') == {'edits': []}
    assert calls[0] == calls[1]
    assert 'FAMILY: ' + family in calls[0][0]
    assert 'appended-mask inference' in calls[0][0]


def test_finish_blocks_changed_checkpoint_before_test(tmp_path):
    c = runner.FamilyJumpController(config(tmp_path))
    trial = tmp_path / 'selected'
    (trial / 'working').mkdir(parents=True)
    (trial / 'working/predict.py').write_text('pass\n')
    checkpoint = trial / 'working/a.pt'
    checkpoint.write_text('changed')
    runner.save(trial / 'single-model-audit.json', {
        'eligible': True, 'predictor_sha256': runner.sha(trial / 'working/predict.py'),
        'checkpoint_sha256': {str(checkpoint): 'oldhash'},
    })
    c.best = {'output': str(trial)}
    with pytest.raises(RuntimeError, match='checkpoint changed'):
        c.finish()
