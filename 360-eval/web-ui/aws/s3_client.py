"""OFFLINE object store — local filesystem replacement for S3.

Keeps the exact public surface of the cloud s3_client (same function names and
signatures) so app.py and _launch_local_eval work unchanged. An S3 "key" maps to
a file at {STORAGE_ROOT}/{key}. Keys use posix '/' separators, mirroring S3.
"""
import os
import shutil
from urllib.parse import quote

from aws._local import STORAGE_ROOT

# Kept for compatibility — now just a label; never used as a real bucket offline.
S3_BUCKET = os.environ.get('S3_BUCKET', '360eval-local')


def _key_to_path(key):
    """Resolve an S3-style key to an absolute path under STORAGE_ROOT (traversal-guarded)."""
    key = (key or '').lstrip('/')
    path = os.path.abspath(os.path.join(STORAGE_ROOT, *key.split('/')))
    root = os.path.abspath(STORAGE_ROOT)
    if path != root and not path.startswith(root + os.sep):
        raise ValueError(f'Illegal storage key (path traversal): {key!r}')
    return path


def _user_prefix(user_id, sub_path=''):
    prefix = f'users/{user_id}'
    if sub_path:
        prefix = f'{prefix}/{sub_path}'
    return prefix


def upload_file(user_id, sub_path, file_obj):
    """Write a file-like object to the local store. Returns the key."""
    s3_key = _user_prefix(user_id, sub_path)
    path = _key_to_path(s3_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        shutil.copyfileobj(file_obj, f)
    return s3_key


def upload_bytes(user_id, sub_path, data):
    """Write bytes or a string to the local store. Returns the key."""
    s3_key = _user_prefix(user_id, sub_path)
    path = _key_to_path(s3_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = data.encode('utf-8') if isinstance(data, str) else data
    with open(path, 'wb') as f:
        f.write(body)
    return s3_key


def download_file(s3_key, local_path):
    """Copy a stored object to a local file path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copyfile(_key_to_path(s3_key), local_path)
    return local_path


def download_bytes(s3_key):
    """Read a stored object and return its contents as bytes."""
    with open(_key_to_path(s3_key), 'rb') as f:
        return f.read()


def delete_object(s3_key):
    """Delete a single stored object (no-op if absent)."""
    try:
        os.remove(_key_to_path(s3_key))
    except FileNotFoundError:
        pass


def delete_prefix(prefix):
    """Delete every stored object whose key starts with `prefix`, then prune empty dirs."""
    for key in list_objects(prefix):
        delete_object(key)
    # Prune now-empty directories under the prefix's base dir.
    base = _key_to_path(prefix)
    base_dir = base if os.path.isdir(base) else os.path.dirname(base)
    root = os.path.abspath(STORAGE_ROOT)
    for dirpath, _dirnames, _filenames in os.walk(base_dir, topdown=False):
        if os.path.abspath(dirpath) == root:
            continue
        try:
            os.rmdir(dirpath)
        except OSError:
            pass  # not empty / already gone


def list_objects(prefix):
    """List all keys whose string value starts with `prefix` (S3 prefix semantics)."""
    keys = []
    root = os.path.abspath(STORAGE_ROOT)
    if not os.path.isdir(root):
        return keys
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            key = os.path.relpath(full, root).replace(os.sep, '/')
            if key.startswith(prefix):
                keys.append(key)
    keys.sort()
    return keys


def generate_presigned_url(s3_key, expiry=3600):
    """Offline replacement: a same-origin admin route that streams the local file.

    The only caller is the admin download route, which hands this URL to the
    browser. `expiry` is ignored (no signing offline).
    """
    return f'/api/admin/local-file?key={quote(s3_key)}'


def object_exists(s3_key):
    """Check whether a stored object exists."""
    try:
        return os.path.isfile(_key_to_path(s3_key))
    except ValueError:
        return False


def get_object_size(s3_key):
    """Return the size of a stored object in bytes (0 on error)."""
    try:
        return os.path.getsize(_key_to_path(s3_key))
    except (OSError, ValueError):
        return 0
