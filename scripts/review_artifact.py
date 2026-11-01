"""Audit and package program sources, excluding runtime records and Git metadata."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIXED_FILES = {'README.md', 'LICENSE', 'MANIFEST.in', 'pyproject.toml', 'requirements.txt',
               '.gitignore', '.env.example', 'run2.sh', 'gr.sh', 'prepare_gr_data.py'}
SOURCE_DIRS = {'gagc', 'recharness', 'tests', 'scripts'}
DOCUMENTS = {'docs/method.md', 'docs/reproduction.md', 'docs/kuairec.md'}
CONFIGS = {'configs/JUMP_POLICY.json', 'configs/MEMORY_DESIGN.json', 'configs/cold_starts.json'}
EXCLUDED = {'.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
            'build', 'dist', 'logs', 'log', 'results', 'runs', 'workspace', 'input', 'data',
            'artifacts', 'calls', 'request-cache', 'observations', 'checkpoints'}
PATTERNS = {
    'host-specific home path': re.compile(r'/(?:home|Users|root|mnt)/[A-Za-z0-9_.-]+/'),
    'personal source or preprint link': re.compile(r'https?://(?:www\.)?(?:github\.com|arxiv\.org)/[^\s)]+'),
    'email address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'credential': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16})\b'),
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----'),
    'credential assignment': re.compile(r'''(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*["'][A-Za-z0-9_./+=-]{16,}["']'''),
    'authenticated URL': re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
}


def allowed(relative):
    path = Path(relative)
    name = path.as_posix()
    if name in FIXED_FILES or name in DOCUMENTS or name in CONFIGS:
        return True
    if not path.parts or any(part.lower() in EXCLUDED or part.endswith('.egg-info') for part in path.parts):
        return False
    return path.parts[0] in SOURCE_DIRS and path.suffix == '.py'


def files(root=ROOT):
    root = Path(root)
    paths = [root / name for name in FIXED_FILES | DOCUMENTS | CONFIGS if (root / name).is_file()]
    for name in SOURCE_DIRS:
        directory = root / name
        if directory.is_dir():
            paths.extend(p for p in directory.rglob('*') if p.is_file() and allowed(p.relative_to(root)))
    for path in paths:
        if path.is_symlink():
            raise ValueError('Symlinks are not allowed: ' + str(path.relative_to(root)))
    return sorted(set(paths))


def audit(paths, root=ROOT, deny_terms=()):
    root = Path(root)
    patterns = dict(PATTERNS)
    if deny_terms:
        patterns['specified private identifier'] = re.compile('|'.join(re.escape(term) for term in deny_terms), re.I)
    findings = []
    for path in paths:
        path = Path(path)
        relative = path.relative_to(root)
        if not allowed(relative):
            findings.append(f'{relative}: file is not permitted in a program-only artifact')
        if path.is_symlink():
            findings.append(f'{relative}: symlink')
            continue
        try:
            source = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            findings.append(f'{relative}: binary file')
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            findings.append(f'{relative}: unexpectedly large file')
        for label, pattern in patterns.items():
            for number, line in enumerate(source.splitlines(), 1):
                if pattern.search(line):
                    # Do not echo matching text: the finding itself may be a secret.
                    findings.append(f'{relative}:{number}: {label}')
    return findings


def tracked_findings(root=ROOT):
    root = Path(root)
    if not (root / '.git').exists():
        return []
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root, text=True).split('\0')
    return [f'{name}: tracked file is outside the program-only whitelist'
            for name in tracked if name and (root / name).exists() and not allowed(name)]


def history_findings(root=ROOT):
    """Inspect contributor metadata without printing names or email addresses."""
    root = Path(root)
    if not (root / '.git').exists():
        return []
    rows = subprocess.check_output(['git', 'log', '--format=%h%x00%an%x00%ae%x00%cn%x00%ce'], cwd=root, text=True)
    findings = []
    for line in rows.splitlines():
        commit, author, email, committer, committer_email = line.split('\0')
        anonymous = lambda name, address: name == 'Anonymous Authors' and address.endswith('@example.invalid')
        if not anonymous(author, email) or not anonymous(committer, committer_email):
            findings.append(f'commit {commit}: existing Git history contains identifying contributor metadata')
    return findings


def archive(paths, destination, root=ROOT):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents:
        raise ValueError('Write the program ZIP outside the source checkout')
    findings = audit(paths, root=root)
    if findings:
        raise ValueError('Refusing to package files that failed the source audit')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(paths):
            info = zipfile.ZipInfo('code/' + path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            mode = 0o755 if path.suffix == '.sh' else 0o644
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, path.read_bytes())
    temporary.replace(destination)
    print(f'Archive: {destination.name}; {len(paths)} program/configuration/instruction files; {destination.stat().st_size} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--deny-term', action='append', default=[], help='Extra private identifiers to scan without embedding them in source')
    parser.add_argument('--history', action='store_true', help='Also check existing Git contributor metadata')
    args = parser.parse_args()
    paths = files()
    findings = tracked_findings() + audit(paths, deny_terms=args.deny_term)
    if args.history:
        findings += history_findings()
    if findings:
        print('\n'.join(findings), file=sys.stderr)
        return 1
    print(f'PASS: {len(paths)} program files; no runtime records, direct identifiers or credential patterns found')
    if args.archive:
        archive(paths, args.archive)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
