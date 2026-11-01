"""Explicit tracing consent and credential-free training worker environments."""
import os
import re

TRACING_FLAGS = ('LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2', 'LANGSMITH_TRACING')
PRIVATE_NAME = re.compile(r'(?:^|_)(?:APIKEY|KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?|AUTHORIZATION)(?:_|$)')
AUTH_LOCATIONS = {'SSH_AUTH_SOCK', 'SSH_AGENT_PID', 'GIT_ASKPASS', 'SSH_ASKPASS',
                  'NETRC', 'CODEX_HOME'}


def llm_environment(environment=None):
    env = dict(os.environ if environment is None else environment)
    enabled = env.get('RECHARNESS_ENABLE_TRACING', '').lower() in {'1', 'true', 'yes', 'on'}
    for name in TRACING_FLAGS:
        env[name] = 'true' if enabled else 'false'
    return env


def configure_tracing():
    env = llm_environment()
    for name in TRACING_FLAGS:
        os.environ[name] = env[name]


def worker_environment(environment=None):
    env = dict(os.environ if environment is None else environment)
    for name in list(env):
        if PRIVATE_NAME.search(name.upper()) or name.upper() in AUTH_LOCATIONS:
            env.pop(name)
    env['RECHARNESS_ENABLE_TRACING'] = 'false'
    for name in TRACING_FLAGS:
        env[name] = 'false'
    return env
