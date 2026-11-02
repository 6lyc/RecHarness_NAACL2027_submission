"""All public entrypoints resolve the same transport and workload settings."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from recharness.paths import REPO


@pytest.mark.parametrize('entry', [
    [sys.executable, '-m', 'recharness'],
    [sys.executable, '-m', 'recharness.kuai_cli'],
    ['bash', str(REPO / 'run2.sh')],
    ['bash', str(REPO / 'gr.sh')],
])
def test_defaults_and_overrides_agree_without_allocating_a_run(tmp_path, entry):
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'RECHARNESS_PYTHON': sys.executable,
           'PYTHONDONTWRITEBYTECODE': '1', 'RECHARNESS_LLM_MODEL': 'unused-provider-setting',
           'GAGC_CLAUDE_MODEL': 'unused-code-setting'}
    def inspect(*args):
        result = subprocess.run([*entry, *args, '--print-llm-config'], cwd=tmp_path,
                                env=env, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    defaults = inspect()
    assert defaults['llm_transport'] == 'codex_cli'
    assert defaults['llm_model'] == 'gpt-6.1-sol'
    assert defaults['llm_effort'] == 'medium'
    assert defaults['llm_timeout'] == 1200
    assert defaults['workloads'] == ['hypotheses', 'implementation']
    explicit = inspect('--llm-model', 'synthetic-model', '--llm-effort', 'high', '--llm-timeout', '30')
    assert explicit['llm_model'] == 'synthetic-model' and explicit['llm_effort'] == 'high'
    assert explicit['llm_timeout'] == 30
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('module', ['recharness', 'recharness.kuai_cli'])
def test_invalid_configuration_fails_before_allocating_a_run(tmp_path, module):
    result = subprocess.run([sys.executable, '-m', module, '--llm-timeout', 'nan',
                             '--print-llm-config'], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'},
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'positive and finite' in result.stderr
    assert not list(tmp_path.iterdir())
