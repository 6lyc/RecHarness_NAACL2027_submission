#!/usr/bin/env python3
"""prepare_gr_data.py — Preprocess local KuaiRec data for GR training.

Self-contained pipeline that takes raw KuaiRec files and produces the
train_data.npy / val_data.npy / test_data.npy expected by gr.sh.

Usage:
    python prepare_gr_data.py --output-dir ./input --raw-dir /data/kuairec

Options:
    --output-dir DIR     Where to write train_data.npy / val_data.npy / test_data.npy  [./input]
    --raw-dir    DIR     Local directory containing the raw CSV files    [required]
    --no-cache           Reprocess even if output arrays already exist
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Normalize raw interaction matrices for the GR input pipeline.
# Converts big_matrix.csv + small_matrix.csv → *_processed.csv
# ---------------------------------------------------------------------------

def load_interactions(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw targets without fitting any statistic on evaluation rows."""
    frames = []
    for name in ['big_matrix', 'small_matrix']:
        raw = raw_dir / (name + '.csv')
        processed = raw_dir / (name + '_processed.csv')
        df = pd.read_csv(raw if raw.exists() else processed)
        df = df.rename(columns={'video_id': 'item_id'}) if 'item_id' not in df else df
        if 'video_duration_sec' not in df:
            duration = 'video_duration' if 'video_duration' in df else 'duration_ms'
            df['video_duration_sec'] = df[duration].astype(float) / 1000.0
        if 'watch_ratio_raw' not in df:
            if 'watch_ratio' in df:
                df['watch_ratio_raw'] = df['watch_ratio'].astype(float)
            elif 'play_duration' in df:
                df['watch_ratio_raw'] = (df['play_duration'].astype(float) / 1000.0
                                          / df['video_duration_sec'].clip(lower=1e-9))
            else:
                raise ValueError('Input requires raw watch_ratio or play_duration')
        if 'play_duration_sec' not in df:
            df['play_duration_sec'] = (df['play_duration'].astype(float) / 1000.0
                if 'play_duration' in df else df['watch_ratio_raw'] * df['video_duration_sec'])
        columns = ['user_id', 'item_id', 'watch_ratio_raw', 'video_duration_sec', 'play_duration_sec']
        if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
            raise ValueError('Interaction IDs and raw targets must be finite')
        for key in ['user_id', 'item_id']:
            if (df[key] < 0).any() or (df[key] != df[key].astype('int64')).any():
                raise ValueError('Interaction IDs must be nonnegative integers')
            df[key] = df[key].astype('int64')
        frames.append(df[columns].copy())
    return tuple(frames)


def split_interactions(big, small, validation_fraction=0.1, seed=2024):
    """Keep the small matrix for test; split remaining big-matrix interaction groups."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be strictly between zero and one')
    keys = ['user_id', 'item_id']
    test_pairs = pd.MultiIndex.from_frame(small[keys]).unique()
    overlap = pd.MultiIndex.from_frame(big[keys]).isin(test_pairs)
    remaining = big.loc[~overlap].copy()
    groups = remaining[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    if len(groups) < 2 or small.empty:
        raise ValueError('Need at least two non-test interaction groups and a nonempty test set')
    # The RNG sees sorted unique keys: row order and duplicate observations cannot
    # change group membership. Every observation of a user/item pair stays together.
    order = np.random.default_rng(seed).permutation(len(groups))
    count = min(len(groups) - 1, max(1, int(round(len(groups) * validation_fraction))))
    val_pairs = pd.MultiIndex.from_frame(groups.iloc[order[:count]])
    mask = pd.MultiIndex.from_frame(remaining[keys]).isin(val_pairs)
    train, val = remaining.loc[~mask].copy(), remaining.loc[mask].copy()
    return train, val, small.copy(), {
        'removed_big_rows_overlapping_test': int(overlap.sum()),
        'validation_groups': count, 'non_test_groups': len(groups),
        'split_unit': 'user_id,item_id', 'pairwise_disjoint': True,
    }


def normalize_splits(train, val, test):
    """Fit normalization on training rows, then apply the frozen statistics."""
    duration_min = float(train['video_duration_sec'].min())
    duration_max = float(train['video_duration_sec'].max())
    duration_scale = max(duration_max - duration_min, 1e-9)
    stats = train.groupby('item_id')['watch_ratio_raw'].agg(['min', 'max'])
    global_min = float(train['watch_ratio_raw'].min())
    global_max = float(train['watch_ratio_raw'].max())
    result = []
    for frame in [train, val, test]:
        frame = frame.copy()
        frame['duration_normed'] = (frame['video_duration_sec'] - duration_min) / duration_scale
        minimum = frame['item_id'].map(stats['min']).fillna(global_min)
        maximum = frame['item_id'].map(stats['max']).fillna(global_max)
        frame['watch_ratio_normed'] = ((frame['watch_ratio_raw'] - minimum)
                                     / (maximum - minimum).clip(lower=1e-9))
        result.append(frame)
    return tuple(result)


# ---------------------------------------------------------------------------
# Build item_categories.csv from a local multi-category CSV.
# ---------------------------------------------------------------------------

def build_item_categories_from_local(raw_dir: Path) -> Path:
    """Convert video_raw_categories_multi.csv → item_categories.csv format.

    data_process.py expects item_categories.csv with columns:
        item_id (index), feat  (Python-literal list of up to 4 int tags)
    """
    out_path = raw_dir / "item_categories.csv"
    if out_path.exists():
        return out_path

    src = raw_dir / "video_raw_categories_multi.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Provide the local category feature file."
        )

    df = pd.read_csv(str(src))

    # Identify tag columns — typically 'feat0'..'feat3' or 'category_0'..'category_3'
    tag_cols = [c for c in df.columns if c.startswith("feat") or c.startswith("category")]
    if not tag_cols:
        # Treat all non-id columns as tag columns
        id_col   = "item_id" if "item_id" in df.columns else "video_id"
        tag_cols = [c for c in df.columns if c != id_col]
    tag_cols = tag_cols[:4]  # cap at 4

    if "item_id" not in df.columns and "video_id" in df.columns:
        df.rename(columns={"video_id": "item_id"}, inplace=True)

    # Encode each tag column: string/numeric → integer via LabelEncoder; NaN → -1
    for col in tag_cols:
        nan_mask = df[col].isna()
        lbe = LabelEncoder()
        df[col] = lbe.fit_transform(df[col].fillna("__NAN__").astype(str))
        df[col] = df[col].where(~nan_mask, -1)   # restore NaN positions → -1
        df.loc[df[col] != -1, col] += 1           # shift: reserve 0 for unknown

    def _row_to_list(row):
        vals = [int(row[c]) for c in tag_cols]
        while len(vals) < 4:
            vals.append(-1)
        return vals

    df["feat"] = df.apply(_row_to_list, axis=1)
    df[["item_id", "feat"]].to_csv(str(out_path), index=False)
    print(f"  Built {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# data_process pipeline (embedded from snailma0229/GR data_process.py)
# ---------------------------------------------------------------------------

ORDINAL_COLS = [
    "user_active_degree", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
]
ONEHOT_COLS = [f"onehot_feat{x}" for x in range(18)]


def _label_encode_col(series: pd.Series) -> pd.Series:
    series = series.map(lambda x: chr(0) if x == "UNKNOWN" else x)
    lbe = LabelEncoder()
    encoded = lbe.fit_transform(series)
    unknown_in_classes = (chr(0) in lbe.classes_.tolist()
                          or -124 in lbe.classes_.tolist())
    if not unknown_in_classes:
        encoded = encoded + 1
    return pd.Series(encoded, index=series.index)


def load_item_category_feat(data_raw: Path) -> pd.DataFrame:
    filepath = data_raw / "item_categories.csv"
    df_raw = pd.read_csv(str(filepath), header=0)
    df_raw["feat"] = df_raw["feat"].map(eval)
    df_feat = pd.DataFrame(df_raw["feat"].tolist(),
                           columns=["feat0", "feat1", "feat2", "feat3"])
    df_feat.index = df_raw['item_id'].to_numpy()
    df_feat.index.name = "item_id"
    df_feat[df_feat.isna()] = -1
    df_feat = (df_feat + 1).astype(int)
    return df_feat


def load_item_duration(data_raw: Path, train: pd.DataFrame) -> pd.Series:
    """Compute item-duration statistics from training rows."""
    return train.groupby('item_id')['duration_normed'].mean()


def load_user_feat(data_raw: Path, user_filter=None) -> pd.DataFrame:
    # Try user_features.csv first, then user_features_raw.csv
    for fname in ("user_features.csv", "user_features_raw.csv"):
        fp = data_raw / fname
        if fp.exists():
            break
    else:
        raise FileNotFoundError(
            "Neither user_features.csv nor user_features_raw.csv found in "
            f"{data_raw}. Please download the KuaiRec user features file."
        )

    needed = ["user_id"] + ORDINAL_COLS + ONEHOT_COLS
    df_user = pd.read_csv(str(fp), usecols=lambda c: c in needed)

    for col in ORDINAL_COLS:
        if col in df_user.columns:
            df_user[col] = _label_encode_col(df_user[col])

    for col in ONEHOT_COLS:
        if col in df_user.columns:
            df_user[col] = df_user[col].fillna(-124)
            df_user[col] = _label_encode_col(df_user[col])
        else:
            df_user[col] = 0

    df_user = df_user.set_index("user_id")

    if user_filter is not None:
        df_user = df_user.reindex(user_filter).fillna(0)
    return df_user


def build_dataset(df_interact: pd.DataFrame,
                  df_feat: pd.DataFrame,
                  df_user: pd.DataFrame,
                  df_item: pd.DataFrame) -> pd.DataFrame:
    df = df_interact.join(df_feat[["feat0", "feat1", "feat2", "feat3"]],
                          on="item_id", how="left")
    df = df.join(
        df_item[["duration_normed"]].rename(
            columns={"duration_normed": "item_duration_normed"}),
        on="item_id", how="left")
    df = df.join(df_user, on="user_id", how="left")
    return df


def _build_npy(df: pd.DataFrame) -> np.ndarray:
    data_sets = []
    for index, data in df.iterrows():
        if index % 50000 == 0:
            print(f"    Processing row {index:,} ...")

        usr_list = data[["user_id"] + ONEHOT_COLS].values.tolist()
        usr_list = [0 if i == 12345 else int(i) + 1 for i in usr_list]
        usr_list[0] = usr_list[0] - 1
        usr_len = len(usr_list)
        assert usr_len == 19

        item_list = [int(data["item_id"]),
                     int(data["feat0"]), int(data["feat1"]),
                     int(data["feat2"]), int(data["feat3"])]
        item_list = item_list[:5]
        item_len  = 1 + sum(1 for v in item_list[1:] if v != 0)
        item_list = item_list + [0] * (5 - len(item_list))

        usr_mask  = [0.0 if v == 0 and idx != 0 else 1.0
                     for idx, v in enumerate(usr_list)]
        item_mask = [1.0] * item_len + [0.0] * (5 - item_len)

        # ── 53-col layout (raw watch-time space, matches paper MAE) ─────
        #   [48] play_duration_sec  — WT MAE GT
        #   [49] video_duration_sec — duration feature and WR recovery
        #   [50] watch_ratio_raw    — WR metric ground truth
        #   [51] usr_len, [52] item_len
        play_dur_sec  = float(data.get("play_duration_sec", 0.0))
        watch_ratio   = float(data.get("watch_ratio_raw", 0.0))
        video_dur_sec = float(data.get("video_duration_sec", 0.0))

        row = (usr_list + item_list + usr_mask + item_mask
               + [play_dur_sec, video_dur_sec, watch_ratio,
                  float(usr_len), float(item_len)])
        data_sets.append(row)

    return np.array(data_sets, dtype=float)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args) -> None:
    import hashlib
    import json
    output_dir, raw_dir = Path(args.output_dir), Path(args.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / (name + '_data.npy') for name in ['train', 'val', 'test']}
    manifest = output_dir / 'splits.json'
    settings = {'schema_version': 2, 'seed': args.seed,
                'validation_fraction': args.validation_fraction,
                'test_source': 'small_matrix', 'normalization_fit': 'train_only'}
    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                value.update(chunk)
        return value.hexdigest()
    inputs = {}
    for name in ['big_matrix', 'small_matrix']:
        path = raw_dir / (name + '.csv')
        if not path.exists():
            path = raw_dir / (name + '_processed.csv')
        inputs[path.name] = digest(path)
    for name in ['item_categories.csv', 'video_raw_categories_multi.csv',
                 'user_features.csv', 'user_features_raw.csv']:
        path = raw_dir / name
        if path.exists():
            inputs[name] = digest(path)
    if any(p.exists() for p in paths.values()) and not args.no_cache:
        if not manifest.exists():
            raise ValueError('Prepared arrays require a verified split manifest; use --no-cache')
        previous = json.loads(manifest.read_text())
        if previous.get('settings') != settings or previous.get('inputs') != inputs:
            raise ValueError('Cached split settings or input hashes do not match; use --no-cache')
        if any(not path.exists() or digest(path) != previous['files'][name]['sha256']
               for name, path in paths.items()):
            raise ValueError('Prepared arrays fail completeness or checksum checks; use --no-cache')
        print('Verified cached train/validation/test arrays')
        return
    big, small = load_interactions(raw_dir)
    train, val, test, partition = split_interactions(big, small, args.validation_fraction, args.seed)
    train, val, test = normalize_splits(train, val, test)
    if not (raw_dir / 'item_categories.csv').exists():
        build_item_categories_from_local(raw_dir)
    # An automatically derived category file is a deterministic input on later runs.
    inputs['item_categories.csv'] = digest(raw_dir / 'item_categories.csv')
    features = load_item_category_feat(raw_dir)
    items = features.join(load_item_duration(raw_dir, train), on='item_id', how='left')
    users = load_user_feat(raw_dir)
    record = {'settings': settings, 'inputs': inputs, 'partition': partition,
              'columns': {'48': 'play_duration_sec', '49': 'video_duration_sec', '50': 'watch_ratio_raw'},
              'files': {}}
    for name, frame in [('train', train), ('val', val), ('test', test)]:
        data = _build_npy(build_dataset(frame.reset_index(drop=True), features, users, items))
        if data.shape[1] != 53 or not np.isfinite(data).all():
            raise ValueError('Prepared arrays must contain 53 finite columns')
        temporary = paths[name].with_suffix('.tmp.npy')
        np.save(temporary, data)
        temporary.replace(paths[name])
        record['files'][name] = {'name': paths[name].name, 'rows': len(data), 'sha256': digest(paths[name])}
        print(f'{name}: {len(data):,} rows')
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(manifest)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess local KuaiRec data for GR training."
    )
    parser.add_argument("--output-dir",    default="./input",
                        help="Output directory for train_data.npy / val_data.npy / test_data.npy  [./input]")
    parser.add_argument("--raw-dir", required=True,
                        help="Local directory containing raw KuaiRec CSV files")
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                        help="Fraction of non-test user/item groups used for validation [0.1]")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--no-cache",      action="store_true",
                        help="Reprocess even if outputs already exist")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
