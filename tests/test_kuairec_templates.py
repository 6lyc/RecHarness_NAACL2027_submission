"""Optional real CPU training and frozen-inference checks for KuaiRec templates."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from gagc.benchmarks.kuairec.artifacts import retain_trial

pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('family', ['gr', 'd2q', 'tpm'])
def test_test_mode_loads_weights_without_training_or_validation_files(tmp_path, family):
    template = ROOT / 'gagc/templates' / family
    shutil.copytree(template, tmp_path / 'trial', ignore=shutil.ignore_patterns('__pycache__'))
    script = tmp_path / 'trial/train.py'
    for name, offset in [('train', 0), ('val', 1), ('test', 2)]:
        data = np.ones((12, 53), dtype=np.float64)
        data[:, 0] = np.arange(12) % 3
        data[:, 19] = np.arange(12) % 4
        data[:, 48] = .01 + np.arange(12) * .001 + offset * .0001
        data[:, 49] = 1. + np.arange(12) % 2
        data[:, 50] = data[:, 48] / data[:, 49]
        data[:, 51], data[:, 52] = 19, 5
        np.save(tmp_path / (name + '.npy'), data)
    env = {**os.environ, 'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
           'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'GR_TRAIN_DATA': str(tmp_path / 'train.npy'), 'GR_VAL_DATA': str(tmp_path / 'val.npy'),
           'GR_RUN_MODE': 'train', 'GR_NUM_EPOCHS': '2', 'GR_BATCH_SIZE': '12',
           'GR_FEAT_DIM': '8', 'GR_HIDDEN_DIM': '8', 'GR_N_HEAD': '1', 'GR_DEC_LAYERS': '1',
           'GR_FFN_DIM': '8', 'D2Q_EMB_DIM': '4', 'D2Q_HIDDEN_DIM': '8',
           'D2Q_NUM_GROUPS': '2', 'D2Q_SPARSE_BUCKETS': '64', 'D2Q_DURATION_BUCKETS': '16',
           'TPM_EMB_DIM': '4', 'TPM_HIDDEN_DIM': '8', 'TPM_NUM_LEAVES': '4',
           'TPM_DURATION_GROUPS': '2', 'TPM_SPARSE_BUCKETS': '64'}
    trained = subprocess.run([sys.executable, str(script)], env=env, cwd=script.parent,
                             capture_output=True, text=True, timeout=90)
    assert trained.returncode == 0, trained.stderr
    retained = retain_trial(script, tmp_path / 'retained')
    shutil.rmtree(script.parent)
    script = Path(retained['directory']) / 'train.py'
    checkpoint = script.parent / 'gr_checkpoint.pt'
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path / 'train.npy').unlink()
    (tmp_path / 'val.npy').unlink()
    env.update(GR_RUN_MODE='evaluate', GR_EVAL_DATA=str(tmp_path / 'test.npy'), GR_NUM_EPOCHS='0')
    tested = subprocess.run([sys.executable, str(script)], env=env, cwd=script.parent,
                            capture_output=True, text=True, timeout=90)
    assert tested.returncode == 0, tested.stderr
    assert 'XAUC=' in tested.stdout and 'Epoch ' not in tested.stdout
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
