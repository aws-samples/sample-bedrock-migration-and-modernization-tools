"""Shared local-storage paths for the OFFLINE build.

The offline app keeps all state under a single data directory (DATA_DIR):

    {DATA_DIR}/db.sqlite3   -> metadata (was DynamoDB)
    {DATA_DIR}/storage/...  -> object store mirroring S3 keys (was S3)
    {DATA_DIR}/secret.key   -> Fernet key (was KMS)

DATA_DIR defaults to "<repo>/.localdata". Override with the DATA_DIR env var.
"""
import os

# This file lives at <repo>/web-ui/aws/_local.py -> parents[2] is <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.abspath(os.environ.get('DATA_DIR', os.path.join(_REPO_ROOT, '.localdata')))

STORAGE_ROOT = os.path.join(DATA_DIR, 'storage')
DB_PATH = os.path.join(DATA_DIR, 'db.sqlite3')
SECRET_KEY_PATH = os.path.join(DATA_DIR, 'secret.key')


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR
