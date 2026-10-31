from recharness.cli import FAMILIES
from recharness.paths import DATASETS, REPO, sha


def config(tmp_path, family='gru4rec'):
    baseline = REPO / 'gagc/templates' / FAMILIES[family] / 'train.py'
    info = {'source': str(baseline), 'sha256': sha(baseline)}
    return {
        'root': str(tmp_path), 'data_dir': str(tmp_path / 'data'),
        'data_manifest_sha256': 'synthetic-test-protocol',
        'gpus': [0, 1, 2, 3], 'cpu_cores': list(range(64)), 'model_family': family,
        'baseline': info, 'base_model': info, 'search_budget_gpu_seconds': 43200,
        'trial_floor_reference_seconds': 10800, 'trial_timeout_seconds': 16200,
        'minimum_trial_seconds': 300, 'final_timeout_seconds': 1800,
        'llm_model': 'synthetic-test-model', 'llm_effort': 'medium',
        'llm_timeout': 1200, 'llm_attempts': 3, 'single_model_search': True,
        'llm_slot_directory': str(tmp_path / 'slots'), 'global_llm_concurrency': 4,
        'worker_python': 'python',
        'alignment': {'datasets': DATASETS, 'preprocessing_seed': 0, 'training_seed': 42,
                      'evaluation_seed': 42, 'validation_users_per_dataset': 200,
                      'negative_candidates': 99, 'cold_start': family + '_perdataset'},
        'replication': {'fresh_baseline_weights': True, 'cross_run_response_or_memory_reuse': False},
    }
