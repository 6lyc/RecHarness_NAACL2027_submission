import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import prepare_gr_data as preparation
from gagc.benchmarks.kuairec.artifacts import validate_splits


def interactions():
    big = pd.DataFrame({
        'user_id': np.repeat(np.arange(4), 5), 'item_id': np.tile(np.arange(5), 4),
        'watch_ratio_raw': np.linspace(.1, 2., 20),
        'video_duration_sec': np.linspace(1., 3., 20),
        'play_duration_sec': np.linspace(.1, 6., 20),
    })
    small = big.iloc[[0, 6, 12]].copy()
    return pd.concat([big, big.iloc[[2]]], ignore_index=True), small


def keys(frame):
    return set(map(tuple, frame[['user_id', 'item_id']].to_numpy()))


def test_three_way_split_is_disjoint_grouped_and_row_order_independent():
    big, small = interactions()
    train, val, test, audit = preparation.split_interactions(big, small, .2, 2024)
    assert not keys(train) & keys(val)
    assert not keys(train) & keys(test)
    assert not keys(val) & keys(test)
    assert len(train) + len(val) + len(test) == len(big)
    assert audit['removed_big_rows_overlapping_test'] == 3
    repeated = preparation.split_interactions(big.sample(frac=1, random_state=3), small, .2, 2024)
    assert keys(repeated[0]) == keys(train)
    assert keys(repeated[1]) == keys(val)


def test_normalization_ignores_evaluation_targets_and_durations():
    big, small = interactions()
    train, val, test, _ = preparation.split_interactions(big, small)
    original = preparation.normalize_splits(train, val, test)[0]
    val['watch_ratio_raw'] = 1e9
    test['video_duration_sec'] = 1e9
    changed = preparation.normalize_splits(train, val, test)[0]
    pd.testing.assert_frame_equal(original, changed)


def test_preparation_writes_verified_three_way_arrays_and_invalidates_old_cache(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'input'
    raw.mkdir()
    big, small = interactions()
    big.to_csv(raw / 'big_matrix_processed.csv', index=False)
    small.to_csv(raw / 'small_matrix_processed.csv', index=False)
    pd.DataFrame({'item_id': range(5), 'feat': ['[0, 1, 2, 3]'] * 5}).to_csv(raw / 'item_categories.csv', index=False)
    pd.DataFrame({'user_id': range(4)}).to_csv(raw / 'user_features.csv', index=False)
    args = SimpleNamespace(raw_dir=raw, output_dir=output, seed=2024,
                           validation_fraction=.2, no_cache=False)
    preparation.run(args)
    manifest = json.loads((output / 'splits.json').read_text())
    assert manifest['settings']['normalization_fit'] == 'train_only'
    train, val, test = [np.load(output / (name + '_data.npy')) for name in ['train', 'val', 'test']]
    assert train.shape[1] == val.shape[1] == test.shape[1] == 53
    assert set(map(tuple, train[:, [0, 19]])).isdisjoint(map(tuple, val[:, [0, 19]]))
    np.testing.assert_allclose(test[:, 48], small.play_duration_sec)
    np.testing.assert_allclose(test[:, 49], small.video_duration_sec)
    np.testing.assert_allclose(test[:, 50], small.watch_ratio_raw)
    before = (output / 'val_data.npy').stat().st_mtime_ns
    preparation.run(args)
    assert (output / 'val_data.npy').stat().st_mtime_ns == before
    (output / 'splits.json').unlink()
    with pytest.raises(ValueError, match='Old arrays'):
        preparation.run(args)


def test_runtime_rejects_missing_or_aliased_validation(tmp_path):
    paths = [tmp_path / (name + '.npy') for name in ['train', 'val', 'test']]
    for index, path in enumerate(paths):
        np.save(path, np.full((2, 53), index))
    assert validate_splits(*paths) == tuple(str(p) for p in paths)
    with pytest.raises(ValueError, match='requires existing'):
        validate_splits(paths[0], None, paths[2])
    with pytest.raises(ValueError, match='distinct'):
        validate_splits(paths[0], paths[2], paths[2])
    paths[1].write_bytes(paths[2].read_bytes())
    with pytest.raises(ValueError, match='distinct'):
        validate_splits(*paths)


def test_runtime_rejects_interaction_overlap_even_when_file_bytes_differ(tmp_path):
    paths = [tmp_path / (name + '.npy') for name in ['train', 'val', 'test']]
    for index, path in enumerate(paths):
        data = np.full((2, 53), index, dtype=float)
        data[:, [0, 19]] = 7  # same interaction under different target values
        np.save(path, data)
    with pytest.raises(ValueError, match='overlapping user/item'):
        validate_splits(*paths)
