"""Deploy Git projects to Kaggle using named accounts and project-owned settings."""
import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import uuid
import zipfile
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import yaml

from remote import inside, sha

CONFIG = Path(os.getenv('XDG_CONFIG_HOME', Path.home() / '.config')) / 'kdeploy'
STATE = Path(os.getenv('XDG_STATE_HOME', Path.home() / '.local/state')) / 'kdeploy'


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
    path = CONFIG / 'accounts' / name(profile) / 'kaggle.json'
    if not path.is_file():
        raise ValueError(f'unknown account: {profile}; use kdeploy account add')
    return path, json.loads(path.read_text())


def api(profile):
    path, credentials = account(profile)
    # A global token must never override the explicitly selected account.
    for key in list(os.environ):
        if key.startswith('KAGGLE_'):
            del os.environ[key]
    os.environ.update(KAGGLE_CONFIG_DIR=str(path.parent), KAGGLE_USERNAME=credentials['username'], KAGGLE_KEY=credentials['key'])
    # Both CLI authentication and the SDK HTTP layer prefer ~/.kaggle/access_token.
    # This process only uses explicitly selected legacy profiles; disable discovery.
    for source in ('kagglesdk', 'kagglesdk.kaggle_http_client', 'kaggle.api.kaggle_api_extended'):
        module: Any = import_module(source)
        module.get_access_token_from_env = lambda: (None, None)
    from kaggle.api.kaggle_api_extended import KaggleApi
    client = KaggleApi()
    client.authenticate()
    return client


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
    payload = snapshot(root, settings, project.get('exclude', []))
    identifier = datetime.now(UTC).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    relative = str(settings.relative_to(root)) if settings.is_relative_to(root) else '.kdeploy-settings.yaml'
    slug = re.sub('[^a-z0-9]+', '-', project['name'].lower()).strip('-')[:30]
    record = {'id': identifier, 'project': project['name'], 'account': profile,
              'kernel': credentials['username'] + '/' + slug + '-' + identifier,
              'settings': str(settings), 'settings_relative': relative, 'settings_sha256': sha(settings),
              'git_commit': git(root, 'rev-parse', 'HEAD').decode().strip(),
              'git_status': git(root, 'status', '--porcelain').decode(),
              'source_sha256': hashlib.sha256(payload).hexdigest(), 'configuration': project,
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
        'dataset_sources': project['datasets'], 'kernel_sources': list(dict.fromkeys(sources))})
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


def outputs(client, kernel):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner, slug = kernel.split('/')
    request = ApiListKernelSessionOutputRequest()
    request.user_name, request.kernel_slug, request.page_size = owner, slug, 200
    files = {}
    seen = set()
    with client.build_kaggle_client() as service:
        while True:
            response = service.kernels.kernels_api_client.list_kernel_session_output(request)
            files.update({item.file_name: item.url for item in response.files or []})
            token = response.next_page_token
            if not token:
                break
            if token in seen:
                raise RuntimeError('Kaggle repeated an output page token')
            seen.add(token)
            request.page_token = token
    return files


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
    urls = outputs(client, record['kernel'])
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
    for kind in ('account', 'project'):
        group = commands.add_parser(kind).add_subparsers(dest='action', required=True)
        group.add_parser('list')
        add = group.add_parser('add')
        add.add_argument('name', type=name)
        if kind == 'account':
            add.add_argument('--credentials', type=Path, required=True, help='existing legacy Kaggle JSON key')
        else:
            add.add_argument('path', type=Path)
            add.add_argument('--account', type=name, required=True)
            add.add_argument('--command', default='python train.py --settings {settings}', dest='entry')
            add.add_argument('--setup', default='')
    for kind in ('deploy', 'continue', 'status', 'logs', 'download'):
        command = commands.add_parser(kind)
        command.add_argument('settings', type=Path, help='settings YAML; external files select a registered project with project: NAME')
        command.add_argument('--account', type=name)
        if kind in ('deploy', 'continue'):
            command.add_argument('--dry-run', action='store_true', help='prepare and inspect without contacting Kaggle')
        else:
            command.add_argument('--run', help='submitted run ID; default: latest matching run')
        if kind == 'download':
            command.add_argument('--to', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ('account', 'project'):
            folder = CONFIG / (args.command + 's')
            if args.action == 'list':
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
            else:
                account(args.account)
                root = args.path.expanduser().resolve()
                if git(root, 'rev-parse', '--show-toplevel').decode().strip() != str(root):
                    raise ValueError('register the Git repository root')
                project = {'name': args.name, 'root': str(root), 'account': args.account, 'command': args.entry,
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
        profile = args.account or project['account']
        account(profile)
        if args.command in ('deploy', 'continue'):
            lock_path = STATE / 'deploy.lock'
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                previous = latest(settings, profile)[1] if args.command == 'continue' else None
                if previous and previous['settings_sha256'] != sha(settings):
                    raise ValueError('settings changed since the previous run; use deploy for a new configuration')
                directory, record = prepare(settings, project, profile, previous)
                if not args.dry_run:
                    client = api(profile)
                    if previous:
                        status = str(client.kernels_status(previous['kernel']).status).lower()
                        if any(word in status for word in ('running', 'queued', 'cancel_requested', 'new_script')):
                            raise ValueError('previous run is still active; use status or logs')
                        # Verify that the saved output actually exists before starting a GPU session.
                        available = outputs(client, previous['kernel'])
                        if 'kdeploy-output/manifest.json' not in available:
                            raise ValueError('previous run has no finalized saved artifacts')
                    record['state'] = 'submitting'
                    write(directory / 'run.json', record)
                    print('submitting', record['id'], record['kernel'], flush=True)
                    response = client.kernels_push(str(directory), acc=project['gpu'] or None)
                    if response.error or any(getattr(response, 'invalid_' + key, []) for key in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources')):
                        raise RuntimeError(response.error or 'Kaggle rejected one or more input sources')
                    canonical = urlsplit(response.url).path.removeprefix('/code/').strip('/')
                    if not re.fullmatch(r'[^/]+/[^/]+', canonical):
                        canonical = getattr(response, 'ref', '')
                    if not re.fullmatch(r'[^/]+/[^/]+', canonical):
                        raise RuntimeError('Kaggle returned an invalid notebook identity')
                    record.update(state='submitted', kernel=canonical, version=response.version_number, url=response.url)
                    write(directory / 'run.json', record)
                print(json.dumps({'id': record['id'], 'kernel': record['kernel'], 'state': record['state'], 'prepared': str(directory)}, indent=2))
            return 0
        directory, record = latest(settings, args.account, args.run)
        client = api(record['account'])
        if args.command == 'status':
            response = client.kernels_status(record['kernel'])
            print(record['id'], record['kernel'], str(response.status), response.failure_message)
        elif args.command == 'logs':
            print(client.kernels_logs(record['kernel']), end='')
        else:
            download(client, record, args.to.expanduser().resolve() if args.to else directory / 'output')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, requests.RequestException, yaml.YAMLError) as error:
        print(f'kdeploy: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
