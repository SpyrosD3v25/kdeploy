"""Standard-library Kaggle launcher, embedded into each submitted notebook."""
import base64
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import time
import zipfile
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'path escapes its root: {name}')
    return path


def restore(previous, target):
    manifest = json.loads((previous / 'manifest.json').read_text())
    for name, expected in manifest.items():
        source = inside(previous / 'files', name)
        destination = inside(target, name)
        if sha(source) != expected:
            raise ValueError(f'saved artifact hash mismatch: {name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def run(bundle, record, project, work=Path('/kaggle/working'), inputs=Path('/kaggle/input')):
    started = time.time()
    payload = base64.b64decode(bundle)
    if hashlib.sha256(payload).hexdigest() != record['source_sha256']:
        raise ValueError('source snapshot hash mismatch')
    source, output = work / 'project', work / 'kdeploy-output'
    source.mkdir(parents=True, exist_ok=False)
    output.mkdir(exist_ok=False)
    files = output / 'files'
    files.mkdir()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            path = inside(source, info.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(info))
            path.chmod((info.external_attr >> 16) & 0o777 or 0o644)
    # Restore the original HEAD and index without uploading credentials, remotes or history.
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    with (source / '.kdeploy/tree.pack').open('rb') as stream:
        subprocess.run(['git', '-C', str(source), 'unpack-objects', '-q'], stdin=stream, check=True)
    subprocess.run(['git', '-C', str(source), 'hash-object', '-w', '-t', 'commit', '--stdin'],
                   input=(source / '.kdeploy/head.commit').read_bytes(), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', '-C', str(source), 'update-ref', 'HEAD', record['git_commit']], check=True)
    (source / '.git/shallow').write_text(record['git_commit'] + '\n')
    subprocess.run(['git', '-C', str(source), 'read-tree', 'HEAD'], check=True)
    skipped = (source / '.kdeploy/skipped').read_bytes()
    if skipped:
        subprocess.run(['git', '-C', str(source), 'update-index', '--skip-worktree', '-z', '--stdin'], input=skipped, check=True)
    shutil.rmtree(source / '.kdeploy')
    settings = inside(source, record['settings_relative'])
    if sha(settings) != record['settings_sha256']:
        raise ValueError('settings hash mismatch')
    previous = record.get('previous')
    if previous:
        candidates = [path.parent for depth in range(1, 4)
                      for path in inputs.glob('*/' * depth + 'kdeploy-output/run.json')
                      if json.loads(path.read_text()).get('id') == previous['id']]
        if len(candidates) != 1:
            raise ValueError('the previous saved notebook output must be attached exactly once')
        restore(candidates[0], files)
    for name in project['outputs']:
        path = inside(source, name)
        target = inside(files, name)
        target.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target, target_is_directory=True)
    env = {**os.environ, **{key: value.replace('{settings}', str(settings)) for key, value in project['env'].items()},
           'KDEPLOY_SETTINGS': str(settings), 'KDEPLOY_STARTED_AT': str(started),
           'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    secrets_dataset = record.get('secrets_dataset')
    if secrets_dataset:
        slug = secrets_dataset.split('/')[1]
        matches = [path for path in inputs.rglob('secrets.yaml') if slug in path.parts and path.is_file()]
        if len(matches) != 1:
            raise ValueError(f'expected one attached kdeploy secrets dataset file; found {len(matches)}')
        secrets_path = matches[0]
        secrets = json.loads(secrets_path.read_text())
        if not isinstance(secrets, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                                     for key, value in secrets.items()):
            raise ValueError('attached kdeploy secrets dataset is invalid')
        env.update(secrets)
    code = 1
    try:
        with (output / 'run.log').open('w') as log:
            for command in (project['setup'], project['command']):
                if not command:
                    continue
                code = 1
                command = command.replace('{settings}', shlex.quote(str(settings)))
                with subprocess.Popen(['bash', '-euc', command], cwd=source, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
                    assert child.stdout is not None
                    for line in child.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = child.wait()
                if code:
                    break
    finally:
        manifest = {str(path.relative_to(files)): sha(path) for path in sorted(files.rglob('*')) if path.is_file()}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (output / 'run.json').write_text(json.dumps({**record, 'returncode': code}, indent=2))
        for name in project['outputs']:
            (source / name).unlink()  # keep one physical/exported copy of each output
    if code:
        raise RuntimeError(f'project command exited with status {code}; partial artifacts were preserved')
