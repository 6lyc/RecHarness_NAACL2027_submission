from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts import review_artifact as review


def test_program_archive_excludes_nested_logs_data_auth_and_caches(tmp_path):
    root = tmp_path / 'source'
    (root / 'gagc').mkdir(parents=True)
    (root / 'gagc/core.py').write_text('value = 1\n')
    for name in ['gagc/debug.log', 'gagc/state.json', 'gagc/calls/response.json',
                 'gagc/logs/record.py', 'gagc/__pycache__/cache.py', '.env', 'results/table.csv']:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('private runtime output')
    paths = review.files(root)
    assert paths == [root / 'gagc/core.py']
    destination = tmp_path / 'program.zip'
    review.archive(paths, destination, root=root)
    with zipfile.ZipFile(destination) as handle:
        assert handle.namelist() == ['code/gagc/core.py']
        assert handle.comment == b'' and handle.testzip() is None


def test_tracked_log_is_rejected_even_when_export_whitelist_excludes_it(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'debug.log').write_text('runtime output')
    subprocess.run(['git', 'add', 'debug.log'], cwd=tmp_path, check=True)
    assert review.tracked_findings(tmp_path) == ['debug.log: tracked file is outside the program-only whitelist']


def test_secret_scan_reports_location_without_echoing_the_secret(tmp_path):
    (tmp_path / 'gagc').mkdir()
    path = tmp_path / 'gagc/core.py'
    token = 'ghp_' + 'A' * 40
    path.write_text('value = ' + repr(token) + '\n')
    findings = review.audit([path], root=tmp_path)
    assert any('credential' in finding and 'core.py:1' in finding for finding in findings)
    assert all(token not in finding for finding in findings)


def test_extra_identifier_check_does_not_require_shipping_private_names(tmp_path):
    path = tmp_path / 'README.md'
    path.write_text('internal-research-group')
    assert review.audit([path], root=tmp_path, deny_terms=['internal-research-group'])


def test_source_symlink_cannot_include_an_external_file(tmp_path):
    root = tmp_path / 'source'
    (root / 'gagc').mkdir(parents=True)
    private = tmp_path / 'private.py'
    private.write_text('value = 1')
    (root / 'gagc/core.py').symlink_to(private)
    with pytest.raises(ValueError, match='Symlinks'):
        review.files(root)
