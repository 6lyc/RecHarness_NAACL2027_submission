"""Select the arm policy required by each regression module."""
import pytest


@pytest.fixture(autouse=True)
def explicit_arm_design(request, monkeypatch):
    legacy = request.module.__name__.split('.')[-1] in {
        'test_jump_v2', 'test_grpo', 'test_gr_arms', 'test_claude_code_backend'}
    monkeypatch.setenv('GAGC_ARM_DESIGN', 'legacy_v2' if legacy else 'semantic_v4')
