# Publish This Repository to GitHub

The intended remote is:

```text
https://github.com/singh-rajni/ml-foundations.git
```

## Fastest option - use the Git bundle

The delivered `ml-foundations.bundle` contains the complete commit history.

```bash
git clone ml-foundations.bundle ml-foundations
cd ml-foundations
git remote set-url origin https://github.com/singh-rajni/ml-foundations.git
git push -u origin main
```

Authenticate with GitHub CLI, a credential manager, or a personal access token when prompted.

## Use the ZIP archive

The ZIP contains committed files but not the `.git` directory.

```bash
unzip ml-foundations-complete.zip
cd ml-foundations
git init -b main
git add .
git commit -m "Add seven-chapter ML foundations curriculum"
git remote add origin https://github.com/singh-rajni/ml-foundations.git
git push -u origin main
```

## Add the files to an existing local clone

Unzip the archive to a temporary location, copy its contents into the existing clone, and preserve the clone's `.git` directory.

```bash
unzip ml-foundations-complete.zip -d /tmp/ml-foundations-package
rsync -av --exclude .git /tmp/ml-foundations-package/ml-foundations/ /path/to/existing/ml-foundations/
cd /path/to/existing/ml-foundations
python -m pytest -q
git add .
git commit -m "Add seven-chapter ML foundations curriculum"
git push
```

## If the remote already has commits and you used the bundle

Fetch and rebase the remote branch before pushing:

```bash
git fetch origin
git rebase origin/main
git push -u origin main
```

Resolve any conflict, rerun tests, and continue the rebase.

## Validate before publishing

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_reference_workflow.py
python scripts/validate_repo.py
```
