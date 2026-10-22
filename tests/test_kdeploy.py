"""No Kaggle jobs, network calls, models or optimizer steps in these tests."""
import hashlib
import json
import os
import subprocess
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import kdeploy as k
import remote


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(k, 'CONFIG', tmp_path / 'config')
    monkeypatch.setattr(k, 'STATE', tmp_path / 'state')
    root = tmp_path / 'source with spaces'
    root.mkdir()
    (root / 'train.py').write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--settings');args=p.parse_args()
s=json.loads(Path(args.settings).read_text())
f=Path('results/counter.txt');f.parent.mkdir(exist_ok=True)
f.write_text(str((int(f.read_text()) if f.exists() else 0)+s['shard']))
print('fixture completed')
''')
    (root / 'settings.yaml').write_text('{"shard": 2}\n')
    (root / '.gitignore').write_text('results/\n')
    for command in (['init','-q',str(root)], ['-C',str(root),'add','.'],
                    ['-C',str(root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                     '-c','commit.gpgsign=false','commit','-qm','fixture source']):
        subprocess.run(['git',*command],check=True)
    credentials=tmp_path/'key.json'
    credentials.write_text('{"username":"fixtureuser","key":"never-print-this"}')
    assert k.main(['account','add','research','--credentials',str(credentials)]) == 0
    assert k.main(['project','add','example',str(root),'--account','research']) == 0
    config=yaml.safe_load((k.CONFIG/'projects/example.yaml').read_text())
    return root,root/'settings.yaml',config


def test_account_storage_and_no_overwrite(project, capsys):
    path,credentials=k.account('research')
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert credentials['username']=='fixtureuser'
    assert 'never-print-this' not in capsys.readouterr().out
    assert k.main(['account','add','research','--credentials',str(path)]) == 1


def test_accounts_are_isolated(project, monkeypatch):
    captured=[]
    class Client:
        def authenticate(self):
            captured.append(dict(os.environ))
    module: Any = types.ModuleType('kaggle.api.kaggle_api_extended')
    module.KaggleApi = Client
    monkeypatch.setitem(sys.modules,'kaggle.api.kaggle_api_extended',module)
    monkeypatch.setenv('KAGGLE_API_TOKEN','wrong-global-token')
    k.api('research')
    other=k.CONFIG/'accounts/other/kaggle.json'
    k.write(other,{'username':'another','key':'different'})
    k.api('other')
    assert 'KAGGLE_API_TOKEN' not in captured[0]
    assert [env['KAGGLE_USERNAME'] for env in captured]==['fixtureuser','another']
    assert [env['KAGGLE_KEY'] for env in captured]==['never-print-this','different']


def test_dry_run_never_authenticates_or_launches(project, monkeypatch):
    _,settings,_=project
    monkeypatch.setattr(k,'api',lambda _:pytest.fail('dry run must not authenticate'))
    assert k.main(['deploy',str(settings),'--dry-run']) == 0
    directory=next((k.STATE/'runs').iterdir())
    notebook=json.loads((directory/'kernel.ipynb').read_text())
    compile(notebook['cells'][0]['source'],'kernel.ipynb','exec')
    metadata=json.loads((directory/'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True
    assert metadata['machine_shape']=='NvidiaTeslaT4'
    assert 'never-print-this' not in (directory/'kernel.ipynb').read_text()
    with pytest.raises(ValueError,match='no submitted run'):
        k.latest(settings)


def execute(directory, work, inputs):
    namespace: dict[str, Any] = {'__name__':'fixture'}
    code=json.loads((directory/'kernel.ipynb').read_text())['cells'][0]['source']
    # Compile definitions without the final invocation, then supply local fixture directories.
    definitions,invocation=code.rsplit('\nrun(',1)
    exec(compile(definitions,'launcher','exec'),namespace)
    import ast
    args=ast.parse('run('+invocation).body[0].value.args
    namespace['run'](*(ast.literal_eval(arg) for arg in args),work=work,inputs=inputs)


def test_snapshot_preserves_original_head_dirty_source_and_settings(project,tmp_path):
    root,settings,config=project
    (root/'train.py').write_text((root/'train.py').read_text()+'print("dirty edit")\n')
    (root/'untracked_secret.txt').write_text('not deployed')
    directory,record=k.prepare(settings,config,'research')
    work=tmp_path/'work'
    work.mkdir()
    execute(directory,work,tmp_path/'input')
    assert (work/'kdeploy-output/files/results/counter.txt').read_text()=='2'
    assert not (work/'project/untracked_secret.txt').exists()
    assert k.git(work/'project','rev-parse','HEAD').decode().strip()==record['git_commit']
    assert b'M train.py' in k.git(work/'project','status','--porcelain')
    assert not (work/'project/results').exists()  # no duplicate output tree
    assert 'dirty edit' in (work/'kdeploy-output/run.log').read_text()
    assert not (work/'project/.git/config').read_text().find('remote "origin"') >= 0


@pytest.mark.parametrize('mount', ['saved-version', 'notebooks/owner/saved-version'])
def test_continuation_restores_outputs_before_project_command(project,tmp_path,mount):
    _,settings,config=project
    first,record=k.prepare(settings,config,'research')
    inputs=tmp_path/'input'
    saved=inputs/mount
    saved.mkdir(parents=True)
    execute(first,saved,inputs)
    second,continued=k.prepare(settings,config,'research',record)
    work=tmp_path/'next-session'
    work.mkdir()
    execute(second,work,inputs)
    assert (work/'kdeploy-output/files/results/counter.txt').read_text()=='4'
    assert continued['previous']=={'id':record['id'],'kernel':record['kernel']}
    metadata=json.loads((second/'kernel-metadata.json').read_text())
    assert metadata['kernel_sources']==[record['kernel']]
    assert metadata['id'] != record['kernel']  # immutable deployment, no latest-version ambiguity


@pytest.mark.parametrize('path',['../escape','/absolute','nested/../../escape'])
def test_path_traversal_rejected(tmp_path,path):
    with pytest.raises(ValueError):
        remote.inside(tmp_path,path)


def test_restore_corruption_is_rejected(tmp_path):
    source=tmp_path/'saved'
    (source/'files').mkdir(parents=True)
    (source/'files/a').write_text('corrupt')
    (source/'manifest.json').write_text('{"a":"wrong-hash"}')
    with pytest.raises(ValueError,match='hash mismatch'):
        remote.restore(source,tmp_path/'target')


def test_failed_command_preserves_artifacts(project,tmp_path):
    _,settings,config=project
    config['command']+='; exit 7'
    directory,_=k.prepare(settings,config,'research')
    work=tmp_path/'work'
    work.mkdir()
    with pytest.raises(RuntimeError,match='status 7'):
        execute(directory,work,tmp_path/'inputs')
    assert json.loads((work/'kdeploy-output/run.json').read_text())['returncode']==7
    assert (work/'kdeploy-output/files/results/counter.txt').read_text()=='2'


def test_download_paginates_all_output_files(monkeypatch):
    tokens=[]
    def page(request):
        tokens.append(request.page_token)
        return SimpleNamespace(files=[SimpleNamespace(file_name=f'file-{len(tokens)}',url='https://fixture')],
                               next_page_token='next' if len(tokens)==1 else '')
    service=SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(list_kernel_session_output=page)))
    class Context:
        def __enter__(self): return service
        def __exit__(self,*args): pass
    assert len(k.outputs(SimpleNamespace(build_kaggle_client=Context),'owner/kernel'))==2
    assert tokens==['','next']


def test_fetch_skips_verified_and_preserves_existing_on_corruption(tmp_path,monkeypatch):
    target=tmp_path/'file'
    target.write_bytes(b'original')
    expected=hashlib.sha256(b'original').hexdigest()
    monkeypatch.setattr(k.requests,'get',lambda *args,**kwargs:pytest.fail('verified file downloaded again'))
    k.fetch('unused',target,expected)
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def raise_for_status(self): pass
        def iter_content(self,size): return [b'corrupt']
    monkeypatch.setattr(k.requests,'get',lambda *args,**kwargs:Response())
    with pytest.raises(ValueError,match='hash mismatch'):
        k.fetch('https://fixture',target,'different expected hash')
    assert target.read_bytes()==b'original'


def test_submit_and_reject_continuation_of_active_job(project,monkeypatch):
    _,settings,_=project
    calls=[]
    def push(path, **kwargs):
        calls.append(path)
        return SimpleNamespace(error='', version_number=1, ref='smoke', url='https://www.kaggle.com/code/canonicaluser/smoke')
    client=SimpleNamespace(kernels_push=push,
                           kernels_status=lambda kernel:SimpleNamespace(status='RUNNING'))
    monkeypatch.setattr(k,'api',lambda _:client)
    assert k.main(['deploy',str(settings)])==0
    _,record=k.latest(settings)
    assert record['state']=='submitted' and record['version']==1
    assert record['kernel'] == 'canonicaluser/smoke'
    assert k.main(['continue',str(settings)])==1
    assert len(calls)==1


def test_download_checks_remote_identity_and_preserves_tree(project, tmp_path, monkeypatch):
    import shutil
    _, settings, config = project
    directory, record = k.prepare(settings, config, 'research')
    saved = tmp_path / 'saved'
    saved.mkdir()
    execute(directory, saved, tmp_path / 'inputs')
    urls = {str(path.relative_to(saved)): str(path) for path in saved.rglob('*') if path.is_file()}
    monkeypatch.setattr(k, 'outputs', lambda *args: urls)
    def fetch(url, target, expected=None):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(url, target)
        if expected:
            assert k.sha(target) == expected
    monkeypatch.setattr(k, 'fetch', fetch)
    target = tmp_path / 'download'
    k.download(None, record, target)
    assert (target / 'files/results/counter.txt').read_text() == '2'
    with pytest.raises(ValueError, match='does not match'):
        k.download(None, {**record, 'id': 'wrong-run'}, tmp_path / 'other-download')


def test_settings_select_the_job_without_a_shard_flag(project, monkeypatch):
    root, _, config = project
    settings = root / 'second-settings.yaml'
    settings.write_text('{"shard": 8}\n')
    monkeypatch.setattr(k, 'api', lambda _: pytest.fail('dry run contacted Kaggle'))
    assert k.main(['deploy', str(settings), '--dry-run']) == 0
    directory = next((k.STATE / 'runs').iterdir())
    record = json.loads((directory / 'run.json').read_text())
    assert record['settings_relative'] == 'second-settings.yaml'
    assert 'shard' not in record


def test_continue_rejects_changed_settings_before_network(project, monkeypatch):
    _, settings, config = project
    directory, record = k.prepare(settings, config, 'research')
    k.write(directory / 'run.json', {**record, 'state': 'submitted'})
    settings.write_text('{"shard": 3}\n')
    monkeypatch.setattr(k, 'api', lambda _: pytest.fail('changed settings contacted Kaggle'))
    assert k.main(['continue', str(settings)]) == 1
