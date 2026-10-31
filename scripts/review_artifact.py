"""Audit distributable sources and create a neutral ZIP without Git metadata."""
import argparse
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {'README.md', 'LICENSE', 'MANIFEST.in', 'pyproject.toml', 'requirements.txt', '.gitignore',
             '.env.example', 'run2.sh', 'gr.sh', 'prepare_gr_data.py', 'gagc',
             'recharness', 'tests', 'configs', 'docs', 'scripts'}
EXCLUDED = {'__pycache__', '.pytest_cache', '.git', '.venv'}
PATTERNS = {
    'host-specific home path': re.compile(r'/(?:home|Users|root|mnt)/[A-Za-z0-9_.-]+/'),
    'personal source or preprint link': re.compile(r'https?://(?:www\.)?(?:github\.com|arxiv\.org)/[^\s)]+'),
    'email address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'credential': re.compile(r'\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,})\b'),
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
}


def files():
    result = []
    for name in sorted(TOP_LEVEL):
        path = ROOT / name
        candidates = path.rglob('*') if path.is_dir() else [path]
        for p in candidates:
            if p.is_file() and not EXCLUDED.intersection(p.relative_to(ROOT).parts) and p.suffix not in {'.pyc', '.pyo'}:
                if p.is_symlink():
                    raise ValueError('Symlinks are not allowed: ' + str(p.relative_to(ROOT)))
                result.append(p)
    return sorted(result)


def audit(paths):
    findings = []
    for path in paths:
        relative = path.relative_to(ROOT)
        try:
            source = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            findings.append(f'{relative}: binary file')
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            findings.append(f'{relative}: unexpectedly large file')
        for label, pattern in PATTERNS.items():
            for number, line in enumerate(source.splitlines(), 1):
                if pattern.search(line):
                    findings.append(f'{relative}:{number}: {label}')
    return findings


def archive(paths, destination):
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError('Write the review ZIP outside the source checkout')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            info = zipfile.ZipInfo('code/' + path.relative_to(ROOT).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            mode = 0o755 if path.suffix == '.sh' else 0o644
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, path.read_bytes())
    print(f'Archive: {destination.name}; {len(paths)} files; {destination.stat().st_size} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    paths = files()
    findings = audit(paths)
    if findings:
        print('\n'.join(findings), file=sys.stderr)
        return 1
    print(f'PASS: {len(paths)} distributable text files; no direct identifiers or credential patterns found')
    if args.archive:
        archive(paths, args.archive)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
