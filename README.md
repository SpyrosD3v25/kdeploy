# kdeploy

Deploy a Git project to Kaggle from your terminal. Automatic account selection, source snapshots, saved outputs and continuation; no project-specific training logic.

## 1. Register an account and project

```bash
mkdir -p ~/Downloads/kdeploy/research1
# Put research1's legacy Kaggle JSON key at ~/Downloads/kdeploy/research1/kaggle.json
kdeploy hydrate accs
kdeploy account list
kdeploy project add example /absolute/path/to/project \
  --command 'python train.py --settings {settings}' \
  --setup 'python -m pip install -r requirements.txt'
kdeploy project list
```

`hydrate accs` registers the existing **legacy Kaggle JSON API keys** in `~/Downloads/kdeploy/<folder>/kaggle.json`; it uses the `username` inside each key as the account name, regardless of the folder name. It does not create Kaggle accounts or copy the keys. Run it again after adding accounts. Account-specific authentication overrides global Kaggle environment variables and access-token files. Existing `account add` profiles remain supported.

Projects do not have a default account. Deploy selects the account whose latest run status was updated least recently, with unused accounts first. The selected account is checked for an active kdeploy run and, for accelerator jobs, remaining Kaggle quota. If it is busy, out of hours, or its quota cannot be checked, deployment exits with a message. Override selection with `--account NAME`. Dry runs do not contact Kaggle or change rotation. Run statuses and accounts are stored in `~/.local/state/kdeploy/kdeploy.sqlite3`; submitted runs from the previous file-based format are imported when deploying. Statuses are refreshed on `status` or when an account is selected, not by a background service.

`project add` prints the configuration path under `~/.config/kdeploy/projects/`. Edit that YAML to set `command`, `setup`, `outputs` (directories, default `[results]`), `env`, `datasets`, `kernels` and `gpu` (default `NvidiaTeslaT4`; null for CPU). Nothing is written into the registered project. Setup and command run in separate shells; persistent environment values belong in `env`. `{settings}` works in commands and environment values; the launcher also sets `KDEPLOY_SETTINGS`.

## 2. Deploy using a settings file

```bash
kdeploy deploy /absolute/path/to/project/settings.yaml --dry-run
kdeploy deploy /absolute/path/to/project/settings.yaml
kdeploy status /absolute/path/to/project/settings.yaml
kdeploy logs /absolute/path/to/project/settings.yaml
kdeploy download /absolute/path/to/project/settings.yaml
```

Use `--account research2` to explicitly select a profile. **Shard selection belongs to the project's settings YAML and its own program.** There is no `--shard` argument. For example, a project may read `shard: 2` itself; kdeploy passes that file unchanged and does not interpret it.

Deploy prepares a private notebook containing tracked working-tree files and the selected settings, including uncommitted tracked edits. Settings may live outside the repository: add `project: example` to select a registered project. Only that file is injected as `.kdeploy-settings.yaml`; it never replaces the repository’s general `settings.yaml` or uploads sibling files. The original external path stays the identity used by status, download and continue; `{settings}` and `KDEPLOY_SETTINGS` point to the injected copy on Kaggle. It preserves the original Git HEAD and index, without uploading Git credentials, remotes or history. Untracked dependencies and submodule contents require project setup; tracked symlinks are rejected. Keep large datasets outside Git and attach them through `datasets`. `--dry-run` creates inspectable local submission files without authenticating or contacting Kaggle.

Every deployment gets a distinct Kaggle notebook, preventing subsequent deployments from replacing its output version. Local run records include source/settings hashes, Git identity, project configuration, account, notebook and returned version. They live under `~/.local/state/kdeploy/runs/`. Status/logs/download accept `--run RUN_ID`; otherwise they select the latest matching run. Download preserves artifact paths, verifies SHA256 and skips files already verified. Add `--to /path` to select its destination.

## 3. Continue after a session

```bash
kdeploy continue /absolute/path/to/project/settings.yaml
```

The previous saved notebook output is attached automatically; verified artifact files are restored to the project's output directories before its command runs. Continuation uses the previous run's account because its notebook is private. Settings must remain unchanged. The project itself must save checkpoints and resume computation from them. Active runs cannot be continued concurrently.

Output directories point directly into `/kaggle/working/kdeploy-output/files/`; the launcher finalizes a manifest, run record and log when the project command exits, including normal command failures. **Use a bounded project run that exits before Kaggle's hard session limit.** A machine killed before finalization may not publish a usable output. Continuation requires an accessible, finalized saved output. Kaggle output quotas still apply, and accumulating artifacts must fit them. Account access, GPU eligibility and quotas remain Kaggle's responsibility.

This implementation uses the [official Kaggle API](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) and [notebook metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md). The installed SDK's output method omits pagination/version handling; kdeploy paginates the SDK endpoint directly and uses distinct notebooks plus run/hash verification. Private Kaggle smoke tests verified authentication, upload, execution, artifact finalization and hash-checked download. Two private GPU smoke sessions on the selected `spyrosx` account confirmed dual Tesla T4 allocation, CUDA execution, checkpoint save/load, exact CPU/CUDA RNG continuation on a fresh session, and verified downloads. Earlier CPU-only runs used an unintended global-token account; account discovery is now isolated, with a regression check for both authentication layers. Local tests cover continuation with a file-writing fixture; no model training has run.

Python 3.11+, Git and Bash are required. Dependencies are in `pyproject.toml`; `pip install -e '.[dev]'` also installs the development checks. Run `ruff check .`, `mypy .` and `python -m pytest`. The local `bin/kdeploy` launcher uses this machine's existing Python; it is added to Bash PATH.
