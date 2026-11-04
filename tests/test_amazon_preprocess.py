"""Synthetic split-output checks for Amazon preprocessing."""
import gzip
import json

import pandas as pd
import pytest

from gagc import data_preprocess as preparation


def raw_2023(root, category, include_metadata=True):
    directory = root / 'benchmark/5core/last_out'
    directory.mkdir(parents=True, exist_ok=True)
    for split, items in [('train', range(5)), ('valid', [5]), ('test', [6])]:
        records = [{'user_id': f'user-{user}', 'parent_asin': f'item-{item}',
                    'timestamp': 100 * user + item}
                   for user in range(6) for item in items]
        pd.DataFrame(records).to_csv(directory / f'{category}.{split}.csv', index=False)
    if include_metadata:
        metadata = root / 'raw/meta_categories' / f'meta_{category}.jsonl'
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(''.join(json.dumps({'parent_asin': f'item-{item}',
                                              'title': 'synthetic title', 'description': []}) + '\n'
                                    for item in range(7)))


def assert_split_outputs(trainval, test, categories):
    assert {p.name for p in trainval.iterdir()} == {
        f'{category}_{split}.txt' for category in categories for split in ['train', 'valid']}
    assert {p.name for p in test.iterdir()} == {f'{category}_test.txt' for category in categories}
    for category in categories:
        rows = {}
        for split, directory in [('train', trainval), ('valid', trainval), ('test', test)]:
            rows[split] = [tuple(map(int, line.split())) for line in
                           (directory / f'{category}_{split}.txt').read_text().splitlines()]
        assert {split: len(values) for split, values in rows.items()} == {
            'train': 30, 'valid': 6, 'test': 6}
        assert not set(rows['train']) & set(rows['valid'])
        assert not set(rows['train']) & set(rows['test'])
        assert not set(rows['valid']) & set(rows['test'])


def test_2023_writes_only_split_interactions_and_removes_auxiliary_metadata(tmp_path):
    raw, trainval, test = [tmp_path / name for name in ['raw', 'trainval', 'test']]
    trainval.mkdir()
    for category in ['Category_A', 'Category_B']:
        raw_2023(raw, category)
        for name in ['text_name_dict.json.gz', f'{category}_text_name_dict.json.gz']:
            (trainval / name).write_bytes(b'synthetic auxiliary metadata')
        preparation.preprocess_raw_5core(category, str(raw), str(trainval), str(test), sample_ratio=1.)
    assert_split_outputs(trainval, test, ['Category_A', 'Category_B'])


def test_2023_does_not_require_item_metadata_or_timestamps(tmp_path):
    raw, trainval, test = [tmp_path / name for name in ['raw', 'trainval', 'test']]
    raw_2023(raw, 'Category', include_metadata=False)
    for path in raw.rglob('*.csv'):
        frame = pd.read_csv(path).drop(columns='timestamp')
        frame.to_csv(path, index=False)
    preparation.preprocess_raw_5core('Category', str(raw), str(trainval), str(test), sample_ratio=1.)
    assert_split_outputs(trainval, test, ['Category'])


@pytest.mark.parametrize('preprocess', [
    preparation.preprocess_amazon_2014_5core, preparation.preprocess_amazon_2014_sasrec])
def test_2014_writes_only_split_interactions_and_removes_auxiliary_metadata(tmp_path, preprocess):
    raw, trainval, test = [tmp_path / name for name in ['raw', 'trainval', 'test']]
    raw.mkdir()
    trainval.mkdir()
    with gzip.open(raw / 'reviews_Category.json.gz', 'wt') as handle:
        for user in range(6):
            for item in range(7):
                handle.write(json.dumps({'reviewerID': f'user-{user}', 'asin': f'item-{item}',
                                         'unixReviewTime': item}) + '\n')
    for name in ['text_name_dict.json.gz', 'Category_text_name_dict.json.gz']:
        (trainval / name).write_bytes(b'synthetic auxiliary metadata')
    preprocess('Category', str(raw), str(trainval), str(test), sample_ratio=1.)
    assert_split_outputs(trainval, test, ['Category'])
