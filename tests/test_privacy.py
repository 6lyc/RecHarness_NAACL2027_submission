import subprocess

from gagc.privacy import TRACING_FLAGS, configure_tracing, llm_environment, worker_environment


def test_tracing_requires_explicit_opt_in_even_with_inherited_credentials():
    inherited = {'LANGSMITH_API_KEY': 'sample', 'LANGSMITH_TRACING': 'true',
                 'LANGCHAIN_TRACING_V2': 'true', 'OPENAI_API_KEY': 'sample'}
    env = llm_environment(inherited)
    assert all(env[key] == 'false' for key in TRACING_FLAGS)
    assert env['OPENAI_API_KEY'] == inherited['OPENAI_API_KEY']
    assert all(llm_environment({**inherited, 'RECHARNESS_ENABLE_TRACING': 'true'})[key] == 'true'
               for key in TRACING_FLAGS)
    assert inherited['LANGSMITH_TRACING'] == 'true'


def test_training_workers_keep_resources_and_drop_authentication():
    auth = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN',
            'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'DATABASE_PASSWORD',
            'GOOGLE_APPLICATION_CREDENTIALS', 'SSH_AUTH_SOCK', 'CODEX_HOME']
    inherited = {name: 'sample' for name in auth}
    inherited.update(CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='4',
                     GAGC_DATA_DIR='/input/trainval', RECHARNESS_ENABLE_TRACING='true')
    safe = worker_environment(inherited)
    assert all(name not in safe for name in auth)
    assert safe['CUDA_VISIBLE_DEVICES'] == '2' and safe['GAGC_DATA_DIR'] == '/input/trainval'
    assert all(safe[key] == 'false' for key in TRACING_FLAGS)


def test_direct_provider_setup_disables_inherited_tracing(monkeypatch):
    monkeypatch.delenv('RECHARNESS_ENABLE_TRACING', raising=False)
    for key in TRACING_FLAGS:
        monkeypatch.setenv(key, 'true')
    configure_tracing()
    import os
    assert all(os.environ[key] == 'false' for key in TRACING_FLAGS)


def test_coding_transport_keeps_auth_but_its_training_command_drops_it():
    from gagc.claude_code_backend import _default_train_command
    env = {'ANTHROPIC_API_KEY': 'sample', 'GR_VAL_DATA': '/input/val.npy',
           'GR_TRAIN_DATA': '/input/train.npy'}
    command = _default_train_command(env)
    assert '-u ANTHROPIC_API_KEY' in command and 'sample' not in command
    result = subprocess.run(['bash', '-c', command.replace('train.py', '-c \'import os; assert "ANTHROPIC_API_KEY" not in os.environ; assert os.environ["GR_VAL_DATA"] == "/input/val.npy"\'')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_training_environment_is_sanitized_at_execution(monkeypatch, tmp_path):
    from recharness import runner
    from .helpers import config
    monkeypatch.setenv('GITHUB_TOKEN', 'sample')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'sample')
    controller = runner.FamilyJumpController(config(tmp_path))
    env = controller.environment(controller.c['gpus'][0])
    assert 'GITHUB_TOKEN' not in env and 'AWS_SECRET_ACCESS_KEY' not in env
