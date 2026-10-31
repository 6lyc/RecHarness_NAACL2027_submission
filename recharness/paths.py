"""Portable paths shared by the controller and worker."""
import hashlib
from pathlib import Path
WORKSPACE = Path(__file__).resolve().parents[1]
REPO = WORKSPACE
ROOT = WORKSPACE / 'runs'
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
