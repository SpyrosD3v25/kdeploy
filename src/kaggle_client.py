"""Kaggle-specific authentication, quota, notebook and output operations."""
import os
from importlib import import_module
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
