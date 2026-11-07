"""Kaggle-specific authentication, quota, notebook and output operations."""
import hashlib
import json
import os
import tempfile
import time
from importlib import import_module
from pathlib import Path
from typing import Any


class KaggleClient:
    def __init__(self, credential_path, credentials):
        for key in list(os.environ):
            if key.startswith('KAGGLE_'):
                del os.environ[key]
        os.environ.update(KAGGLE_CONFIG_DIR=str(credential_path.parent),
                          KAGGLE_USERNAME=credentials['username'], KAGGLE_KEY=credentials['key'])
        for source in ('kagglesdk', 'kagglesdk.kaggle_http_client', 'kaggle.api.kaggle_api_extended'):
            module: Any = import_module(source)
            module.get_access_token_from_env = lambda: (None, None)
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.client = KaggleApi()
        self.client.authenticate()

    def status(self, kernel):
        return self.client.kernels_status(kernel)

    def logs(self, kernel):
        return self.client.kernels_logs(kernel)

    def submit(self, directory, accelerator):
        return self.client.kernels_push(str(directory), acc=accelerator or None)

    def publish_secrets(self, ref, values, state_path):
        payload = json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
        digest = hashlib.sha256(payload).hexdigest()
        datasets = self.client.dataset_list(mine=True, search=ref.split('/')[1]) or []
        existing = next((dataset for dataset in datasets if dataset and dataset.ref == ref), None)
        if existing and not existing.is_private:
            raise ValueError('kdeploy secrets dataset exists but is public')
        state_path = Path(state_path)
        previous = json.loads(state_path.read_text()) if state_path.is_file() else {}
        version = existing.current_version_number if existing else None
        if existing and previous == {'sha256': digest, 'version': version}:
            return ref
        with tempfile.TemporaryDirectory(prefix='kdeploy-secrets-') as temporary:
            folder = Path(temporary)
            secret_file = folder / 'secrets.yaml'
            descriptor = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(payload)
            (folder / 'dataset-metadata.json').write_text(json.dumps({
                'title': 'kdeploy secrets', 'id': ref, 'licenses': [{'name': 'unknown'}]}))
            if existing:
                response = self.client.dataset_create_version(str(folder), 'Update kdeploy secrets', quiet=True,
                                                              convert_to_csv=False)
            else:
                response = self.client.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False)
            if response.error:
                raise RuntimeError('Kaggle rejected the private secrets dataset upload')
        deadline = time.monotonic() + 120
        while True:
            status = self.client.dataset_status(ref).lower()
            if status in ('complete', 'ready'):
                break
            if status in ('error', 'failed') or time.monotonic() >= deadline:
                raise RuntimeError('private secrets dataset did not become ready')
            time.sleep(2)
        state_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        temporary_state = state_path.with_suffix('.tmp')
        temporary_state.write_text(json.dumps({'sha256': digest, 'version': (version or 0) + 1}))
        temporary_state.replace(state_path)
        return ref

    def check_quota(self, profile, accelerator):
        if not hasattr(self.client, 'quota_view'):
            raise ValueError('Kaggle quota API is unavailable; install kaggle>=2.2.1')
        response = self.client.quota_view()
        quota = response.tpu_quota if accelerator.lower().startswith('tpu') else response.gpu_quota
        if quota is None:
            raise ValueError(f'account {profile} has no available accelerator quota information')
        if quota.total_time_allowed <= quota.time_used:
            raise ValueError(f'account {profile} has no accelerator hours remaining')

    def outputs(self, kernel):
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        owner, slug = kernel.split('/')
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug, request.page_size = owner, slug, 200
        files, seen = {}, set()
        with self.client.build_kaggle_client() as service:
            while True:
                response = service.kernels.kernels_api_client.list_kernel_session_output(request)
                files.update({item.file_name: item.url for item in response.files or []})
                token = response.next_page_token
                if not token:
                    return files
                if token in seen:
                    raise RuntimeError('Kaggle repeated an output page token')
                seen.add(token)
                request.page_token = token
