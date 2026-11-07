"""Deploy Git projects to Kaggle using named accounts and project-owned settings."""
import argparse
import ast
import base64
import fcntl
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
import yaml

import database
from kaggle_client import KaggleClient
from remote import inside, sha

CONFIG = Path(os.getenv('XDG_CONFIG_HOME', Path.home() / '.config')) / 'kdeploy'
STATE = Path(os.getenv('XDG_STATE_HOME', Path.home() / '.local/state')) / 'kdeploy'
SECRETS_FILE = '.kdeploy-secrets.yaml'


def write(path, value):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def name(value):
    if not re.fullmatch(r'[a-z][a-z0-9_-]*', value):
        raise argparse.ArgumentTypeError('use a lowercase name containing letters, digits, - or _')
    return value


def git(root, *args, data=None):
    return subprocess.check_output(['git', '-C', str(root), *args], input=data)


def account(profile):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', profile):
        raise ValueError('invalid account name')
    with database.connect(STATE) as db:
        row = db.execute('SELECT credentials_path FROM accounts WHERE name = ?', (profile,)).fetchone()
    path = Path(row[0]) if row else CONFIG / 'accounts' / profile / 'kaggle.json'
    if not path.is_file():
        raise ValueError(f'unknown account: {profile}; use kdeploy hydrate accs')
    return path, json.loads(path.read_text())


def hydrate():
    root = Path.home() / 'Downloads' / 'kdeploy'
    if not root.is_dir():
        raise ValueError(f'account directory does not exist: {root}')
    found = {}
    for path in sorted(root.glob('*/kaggle.json')):
        credentials = json.loads(path.read_text())
        if not all(isinstance(credentials.get(key), str) and credentials[key] for key in ('username', 'key')):
            raise ValueError(f'invalid credentials in {path}')
        profile = credentials['username']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', profile):
            raise ValueError(f'invalid Kaggle username in {path}')
        if profile in found:
            raise ValueError(f'duplicate credentials for Kaggle username {profile}')
        found[profile] = path.resolve()
    with database.connect(STATE) as db:
        for profile, path in found.items():
            db.execute('''INSERT INTO accounts(name, credentials_path) VALUES (?, ?)
                          ON CONFLICT(name) DO UPDATE SET credentials_path = excluded.credentials_path''',
                       (profile, str(path)))
    print(f'Hydrated {len(found)} account(s) from {root}')


def refresh_account(db, profile):
    rows = db.execute("SELECT s.run_id FROM status s JOIN accounts a ON a.id = s.acc WHERE a.name = ? AND s.status IN ('running', 'not_started')", (profile,)).fetchall()
    for (run_id,) in rows:
        path = STATE / 'runs' / run_id / 'run.json'
        if not path.exists():
            raise ValueError(f'account {profile} has an unfinished run {run_id} without a local record')
        record = json.loads(path.read_text())
        if record.get('state') not in ('submitted', 'submitting'):
            raise ValueError(f'account {profile} has an unfinished run {run_id}; inspect its local record')
        result = str(api(profile).status(record['kernel']).status).lower()
        state = kaggle_status(result)
        if state == 'running':
            raise ValueError(f'account {profile} has a running or unresolved run: {run_id} ({result})')
        database.set_status(db, run_id, state)


def kaggle_status(result):
    result = str(result).lower()
    if any(word in result for word in ('complete', 'success')):
        return 'successful'
    if any(word in result for word in ('error', 'fail', 'cancel')) and 'cancel_requested' not in result:
        return 'failed'
    return 'running'


def select_account(db, requested=None, dry_run=False, accelerator=None):
    rows = db.execute('''SELECT a.name, MAX(s.updated_at) AS last_used FROM accounts a
                         LEFT JOIN status s ON s.acc = a.id GROUP BY a.id
                         ORDER BY last_used IS NOT NULL, last_used, a.name''').fetchall()
    candidates = [row[0] for row in rows if not requested or row[0] == requested]
    if not candidates:
        raise ValueError('no matching accounts; use kdeploy hydrate accs')
    if not dry_run:
        refresh_account(db, candidates[0])
        if accelerator:
            api(candidates[0]).check_quota(candidates[0], accelerator)
    return candidates[0]


def api(profile):
    path, credentials = account(profile)
    return KaggleClient(path, credentials)


def read_secrets(root):
    path = root / SECRETS_FILE
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError('secrets file must be a regular file')
    if git(root, 'ls-files', '--', SECRETS_FILE).strip():
        raise ValueError('secrets file must not be tracked by Git')
    try:
        values = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        raise ValueError('invalid secrets YAML') from None
    if not isinstance(values, dict) or not values:
        raise ValueError('secrets YAML must be a nonempty string mapping')
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or not isinstance(value, str):
            raise ValueError('secrets YAML must be a string mapping with environment variable keys')
        if key.startswith(('KAGGLE_', 'KDEPLOY_')) or key in ('HOME', 'PATH', 'PYTHONHOME', 'PYTHONPATH'):
            raise ValueError('secrets YAML contains a reserved environment variable')
    return values


def secrets_dataset_ref(owner, project):
    slug = 'kdeploy-' + re.sub('[^a-z0-9-]+', '-', project['name'].lower()).strip('-') + '-secrets'
    if len(slug) > 50:
        slug = slug[:40].rstrip('-') + '-' + hashlib.sha256(project['root'].encode()).hexdigest()[:8]
    return owner + '/' + slug


def project_for(settings):
    matches = [(path, yaml.safe_load(path.read_text())) for path in (CONFIG / 'projects').glob('*.yaml')]
    selected = yaml.safe_load(settings.read_text()).get('project')
    matches = [(path, project) for path, project in matches
               if (path.stem == selected if selected is not None else settings.is_relative_to(Path(project['root'])))]
    if not matches:
        raise ValueError('no matching registered project; external settings require project: NAME')
    return max(matches, key=lambda pair: len(pair[1]['root']))


def snapshot(root, settings, exclude=()):
    buffer = io.BytesIO()
    paths = {Path(p.decode()) for p in git(root, 'ls-files', '-z').split(b'\0') if p}
    if Path(SECRETS_FILE) in paths:
        raise ValueError('secrets file must not be tracked by Git')
    skipped = {p for p in paths if any(p.match(pattern) for pattern in exclude)}
    paths -= skipped
    external = not settings.is_relative_to(root)
    if external:
        if Path('.kdeploy-settings.yaml') in paths or (root / '.kdeploy-settings.yaml').exists():
            raise ValueError('reserved injected settings path already exists: .kdeploy-settings.yaml')
    else:
        paths.add(settings.relative_to(root))
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(paths):
            if relative.parts[0] in ('.git', '.kdeploy'):
                raise ValueError(f'reserved source path: {relative}')
            path = inside(root, str(relative))
            if path.is_symlink():
                raise ValueError(f'tracked symlinks must be handled by project setup: {relative}')
            if path.is_file():
                archive.write(path, str(relative))
        if external:
            archive.writestr('.kdeploy-settings.yaml', settings.read_bytes())
        archive.writestr('.kdeploy/head.commit', git(root, 'cat-file', 'commit', 'HEAD'))
        objects = {git(root, 'rev-parse', 'HEAD^{tree}').strip()}
        for entry in git(root, 'ls-tree', '-r', '-t', '-z', 'HEAD').split(b'\0'):
            if entry:
                header, filename = entry.split(b'\t', 1)
                _, kind, identity = header.split()
                if kind == b'tree' or Path(filename.decode()) not in skipped:
                    objects.add(identity)
        archive.writestr('.kdeploy/tree.pack', git(root, 'pack-objects', '--stdout', data=b'\n'.join(sorted(objects)) + b'\n'))
        archive.writestr('.kdeploy/skipped', b'\0'.join(str(p).encode() for p in sorted(skipped)) + (b'\0' if skipped else b''))
    return buffer.getvalue()


def prepare(settings, project, profile, previous=None):
    root = Path(project['root'])
    _, credentials = account(profile)
    secrets = read_secrets(root)
    secrets_dataset = secrets_dataset_ref(credentials['username'], project) if secrets is not None else None
    if previous:
        if previous.get('secrets_dataset') and secrets is None:
            raise ValueError('continuation requires the project secrets YAML file')
        notebook = json.loads((STATE / 'runs' / previous['id'] / 'kernel.ipynb').read_text())
        invocation = ast.parse(notebook['cells'][0]['source']).body[-1].value
        payload = base64.b64decode(ast.literal_eval(invocation.args[0]), validate=True)
        if hashlib.sha256(payload).hexdigest() != previous['source_sha256']:
            raise ValueError('saved source snapshot hash mismatch')
        project = previous['configuration']
    else:
        payload = snapshot(root, settings, project.get('exclude', []))
    identifier = datetime.now(UTC).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    relative = previous['settings_relative'] if previous else str(settings.relative_to(root)) if settings.is_relative_to(root) else '.kdeploy-settings.yaml'
    slug = re.sub('[^a-z0-9]+', '-', project['name'].lower()).strip('-')[:30]
    record = {'id': identifier, 'project': project['name'], 'account': profile,
              'kernel': credentials['username'] + '/' + slug + '-' + identifier,
              'settings': str(settings), 'settings_relative': relative, 'settings_sha256': sha(settings),
              'git_commit': previous['git_commit'] if previous else git(root, 'rev-parse', 'HEAD').decode().strip(),
              'git_status': previous['git_status'] if previous else git(root, 'status', '--porcelain').decode(),
              'source_sha256': hashlib.sha256(payload).hexdigest(), 'configuration': project,
              'secrets_dataset': secrets_dataset,
              'previous': {key: previous[key] for key in ('id', 'kernel')} if previous else None}
    directory = STATE / 'runs' / identifier
    code = Path(__file__).with_name('remote.py').read_text()
    code += f'\nrun({base64.b64encode(payload).decode()!r}, {record!r}, {project!r})\n'
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
        {'cell_type': 'code', 'id': 'launch', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code}]}
    write(directory / 'kernel.ipynb', notebook)
    sources = list(project['kernels'])
    if previous:
        sources.append(previous['kernel'])
    write(directory / 'kernel-metadata.json', {'id': record['kernel'], 'title': record['kernel'].split('/')[1],
        'code_file': 'kernel.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': bool(project['gpu']), 'enable_internet': True, 'machine_shape': project['gpu'],
        'dataset_sources': list(dict.fromkeys([*project['datasets'], *([secrets_dataset] if secrets_dataset else [])])),
        'kernel_sources': list(dict.fromkeys(sources))})
    record['state'] = 'prepared'
    write(directory / 'run.json', record)
    return directory, record


def latest(settings, profile=None, identifier=None):
    paths = [STATE / 'runs' / identifier / 'run.json'] if identifier else sorted((STATE / 'runs').glob('*/run.json'), reverse=True)
    for path in paths:
        record = json.loads(path.read_text())
        if record['settings'] == str(settings) and record.get('state') in ('submitted', 'submitting') and (not profile or record['account'] == profile):
            return path.parent, record
    raise ValueError('no submitted run found for these settings/account')


def fetch(url, target, expected=None):
    if expected and target.is_file() and sha(target) == expected:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    if expected and sha(temporary) != expected:
        raise ValueError(f'download hash mismatch: {target.name}')
    temporary.replace(target)


def download(client, record, target):
    urls = client.outputs(record['kernel'])
    if not {'kdeploy-output/run.json', 'kdeploy-output/manifest.json'} <= urls.keys():
        raise ValueError('saved output is not ready; check the run status')
    for filename in ('run.json', 'manifest.json'):
        relative = 'kdeploy-output/' + filename
        fetch(urls[relative], target / filename)
    remote = json.loads((target / 'run.json').read_text())
    if remote['id'] != record['id'] or remote['source_sha256'] != record['source_sha256']:
        raise ValueError('Kaggle output does not match the submitted source/run')
    manifest = json.loads((target / 'manifest.json').read_text())
    for filename, expected in manifest.items():
        destination = inside(target / 'files', filename)
        fetch(urls['kdeploy-output/files/' + filename], destination, expected)
    if 'kdeploy-output/run.log' in urls:
        fetch(urls['kdeploy-output/run.log'], target / 'run.log')
    print(target)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='kdeploy', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    hydrate_parser = commands.add_parser('hydrate')
    hydrate_parser.add_argument('kind', choices=['accs'])
    for kind in ('account', 'project'):
        group = commands.add_parser(kind).add_subparsers(dest='action', required=True)
        group.add_parser('list')
        add = group.add_parser('add')
        add.add_argument('name', type=name)
        if kind == 'account':
            add.add_argument('--credentials', type=Path, required=True, help='existing legacy Kaggle JSON key')
        else:
            add.add_argument('path', type=Path)
            add.add_argument('--account', type=name, help=argparse.SUPPRESS)
            add.add_argument('--command', default='python train.py --settings {settings}', dest='entry')
            add.add_argument('--setup', default='')
    for kind in ('deploy', 'continue', 'status', 'logs', 'download'):
        command = commands.add_parser(kind)
        command.add_argument('settings', type=Path, help='settings YAML; external files select a registered project with project: NAME')
        command.add_argument('--account')
        if kind in ('deploy', 'continue'):
            command.add_argument('--dry-run', action='store_true', help='prepare and inspect without contacting Kaggle')
        else:
            command.add_argument('--run', help='submitted run ID; default: latest matching run')
        if kind == 'download':
            command.add_argument('--to', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'hydrate':
            hydrate()
            return 0
        if args.command in ('account', 'project'):
            folder = CONFIG / (args.command + 's')
            if args.action == 'list':
                if args.command == 'account':
                    with database.connect(STATE) as db:
                        names = {row[0] for row in db.execute('SELECT name FROM accounts')}
                    names.update(path.parent.name for path in folder.glob('*/kaggle.json'))
                    for profile in sorted(names):
                        print(profile)
                    return 0
                for path in sorted(folder.glob('*/kaggle.json' if args.command == 'account' else '*.yaml')):
                    print(path.parent.name if args.command == 'account' else path.stem)
                return 0
            path = folder / args.name / 'kaggle.json' if args.command == 'account' else folder / (args.name + '.yaml')
            if path.exists():
                raise ValueError(f'already exists: {path}')
            if args.command == 'account':
                credentials = json.loads(args.credentials.expanduser().read_text())
                if not all(isinstance(credentials.get(key), str) and credentials[key] for key in ('username', 'key')):
                    raise ValueError('credentials need nonempty username and key fields')
                path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                path.parent.chmod(0o700)
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, 'w') as stream:
                    json.dump({key: credentials[key] for key in ('username', 'key')}, stream)
                with database.connect(STATE) as db:
                    database.upsert_account(db, args.name, path)
            else:
                root = args.path.expanduser().resolve()
                if git(root, 'rev-parse', '--show-toplevel').decode().strip() != str(root):
                    raise ValueError('register the Git repository root')
                project = {'name': args.name, 'root': str(root), 'command': args.entry,
                           'setup': args.setup, 'outputs': ['results'], 'env': {}, 'datasets': [], 'kernels': [], 'gpu': 'NvidiaTeslaT4'}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(project, sort_keys=False))
            print(path)
            return 0
        if getattr(args, 'run', None) and not re.fullmatch(r'[0-9]{8}-[0-9]{6}-[a-f0-9]{8}', args.run):
            raise ValueError('invalid run ID')
        settings = args.settings.expanduser().resolve(strict=True)
        if not isinstance(yaml.safe_load(settings.read_text()), dict):
            raise ValueError('settings must contain a YAML mapping')
        _, project = project_for(settings)
        if not isinstance(project['outputs'], list) or not project['outputs'] or not all(isinstance(item, str) and item for item in project['outputs']) or len(set(project['outputs'])) != len(project['outputs']):
            raise ValueError('project outputs must be a nonempty list of relative directories')
        for output in project['outputs']:
            inside(Path(project['root']), output)
            if output == '.' or Path(output).parts[0] in ('.git', '.kdeploy') or any(other != output and Path(output).is_relative_to(other) for other in project['outputs']):
                raise ValueError('output directories must be separate, non-root paths')
        if not all(isinstance(value, str) for value in project['env'].values()):
            raise ValueError('project environment values must be strings')
        if args.command in ('deploy', 'continue'):
            lock_path = STATE / 'deploy.lock'
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                previous = latest(settings, args.account)[1] if args.command == 'continue' else None
                if previous and previous['settings_sha256'] != sha(settings):
                    raise ValueError('settings changed since the previous run; use deploy for a new configuration')
                with database.connect(STATE) as db:
                    # Import existing local profiles without requiring users to register them again.
                    for path in (CONFIG / 'accounts').glob('*/kaggle.json'):
                        if not db.execute('SELECT 1 FROM accounts WHERE name = ?', (path.parent.name,)).fetchone():
                            database.upsert_account(db, path.parent.name, path)
                    for path in (STATE / 'runs').glob('*/run.json'):
                        old = json.loads(path.read_text())
                        if old.get('state') in ('submitted', 'submitting'):
                            stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
                            db.execute('''INSERT OR IGNORE INTO status(run_id, status, acc, project_path, created_at, updated_at)
                                          SELECT ?, 'running', id, ?, ?, ? FROM accounts WHERE name = ?''',
                                       (old['id'], old['configuration']['root'], stamp, stamp, old['account']))
                    db.commit()
                    profile = select_account(db, args.account or (previous['account'] if previous else None), args.dry_run, project['gpu'])
                directory, record = prepare(settings, project, profile, previous)
                if not args.dry_run:
                    client = api(profile)
                    if previous:
                        status = str(client.status(previous['kernel']).status).lower()
                        if any(word in status for word in ('running', 'queued', 'cancel_requested', 'new_script')):
                            raise ValueError('previous run is still active; use status or logs')
                        # Verify that the saved output actually exists before starting a GPU session.
                        available = client.outputs(previous['kernel'])
                        if 'kdeploy-output/manifest.json' not in available:
                            raise ValueError('previous run has no finalized saved artifacts')
                    secrets = read_secrets(Path(project['root']))
                    if secrets is not None:
                        client.publish_secrets(record['secrets_dataset'], secrets, STATE / 'secrets' / profile / project['name'] / 'state.json')
                    record['state'] = 'submitting'
                    write(directory / 'run.json', record)
                    with database.connect(STATE) as db:
                        database.create_run(db, record['id'], profile, project['root'])
                    print('submitting', record['id'], record['kernel'], flush=True)
                    response = client.submit(directory, project['gpu'])
                    if response.error or any(getattr(response, 'invalid_' + key, []) for key in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources')):
                        with database.connect(STATE) as db:
                            database.set_status(db, record['id'], 'failed')
                        raise RuntimeError(response.error or 'Kaggle rejected one or more input sources')
                    canonical = urlsplit(response.url).path.removeprefix('/code/').strip('/')
                    if not re.fullmatch(r'[^/]+/[^/]+', canonical):
                        canonical = getattr(response, 'ref', '')
                    if not re.fullmatch(r'[^/]+/[^/]+', canonical):
                        raise RuntimeError('Kaggle returned an invalid notebook identity')
                    record.update(state='submitted', kernel=canonical, version=response.version_number, url=response.url)
                    write(directory / 'run.json', record)
                    with database.connect(STATE) as db:
                        database.set_status(db, record['id'], 'running')
                print(json.dumps({'id': record['id'], 'kernel': record['kernel'], 'state': record['state'], 'prepared': str(directory)}, indent=2))
            return 0
        directory, record = latest(settings, args.account, args.run)
        client = api(record['account'])
        if args.command == 'status':
            response = client.status(record['kernel'])
            with database.connect(STATE) as db:
                database.set_status(db, record['id'], kaggle_status(response.status))
            print(record['id'], record['kernel'], str(response.status), response.failure_message)
        elif args.command == 'logs':
            print(client.logs(record['kernel']), end='')
        else:
            download(client, record, args.to.expanduser().resolve() if args.to else directory / 'output')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error, subprocess.CalledProcessError, requests.RequestException, yaml.YAMLError) as error:
        print(f'kdeploy: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
