import json

import pytest

from recharness import cli


def arguments(tmp_path):
    data = tmp_path / 'input'
    for category in cli.DATASETS:
        for folder, suffix in [('trainval', '_train.txt'), ('trainval', '_valid.txt'),
                               ('test', '_test.txt')]:
            path = data / folder / (category + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('1 1\n1 2\n')
    return ['--data-dir', str(data), '--output', str(tmp_path / 'run'),
            '--gpus', '4,5,6,7', '--cpu-cores', ','.join(map(str, range(64)))]


def test_check_fingerprints_inputs_without_creating_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.os, 'sched_getaffinity', lambda _: set(range(64)))
    args = arguments(tmp_path)
    cli.main(['check', *args])
    result = json.loads(capsys.readouterr().out)
    assert result['valid']
    assert len(result['config']['data_files']) == 12
    assert not (tmp_path / 'run').exists()
    old_hash = result['config']['data_manifest_sha256']
    (tmp_path / 'input/trainval/Movies_and_TV_train.txt').write_text('1 3\n')
    cli.main(['check', *args])
    assert json.loads(capsys.readouterr().out)['config']['data_manifest_sha256'] != old_hash


def test_existing_run_is_rejected_before_importing_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.os, 'sched_getaffinity', lambda _: set(range(64)))
    args = arguments(tmp_path)
    (tmp_path / 'run').mkdir()
    with pytest.raises(FileExistsError):
        cli.main(['run', *args])


def test_check_rejects_missing_test_split(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.os, 'sched_getaffinity', lambda _: set(range(64)))
    args = arguments(tmp_path)
    (tmp_path / 'input/test/Movies_and_TV_test.txt').unlink()
    with pytest.raises(SystemExit) as error:
        cli.main(['check', *args])
    assert error.value.code == 2
