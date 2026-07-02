"""OFFLINE credential encryption — Fernet replacement for KMS.

Same public surface as the cloud kms_client. A symmetric key is generated on
first use and stored at {DATA_DIR}/secret.key (mode 0600). encrypt/decrypt round
-trip the per-user API keys stored in the local DB; mask is unchanged.
"""
import os
import threading

from cryptography.fernet import Fernet

from aws._local import SECRET_KEY_PATH, ensure_data_dir

# Kept for compatibility — unused offline (Fernet replaces KMS).
KMS_KEY_ID = os.environ.get('KMS_KEY_ID', '')

_fernet = None
_lock = threading.Lock()


def _get_fernet():
    global _fernet
    if _fernet is not None:
        return _fernet
    with _lock:
        if _fernet is None:
            ensure_data_dir()
            if not os.path.exists(SECRET_KEY_PATH):
                key = Fernet.generate_key()
                # Write atomically-ish, then lock down permissions.
                with open(SECRET_KEY_PATH, 'wb') as f:
                    f.write(key)
                try:
                    os.chmod(SECRET_KEY_PATH, 0o600)
                except OSError:
                    pass
            with open(SECRET_KEY_PATH, 'rb') as f:
                _fernet = Fernet(f.read())
    return _fernet


def encrypt_api_key(plaintext_key):
    """Encrypt an API key. Returns a Fernet token (urlsafe base64 text)."""
    return _get_fernet().encrypt(plaintext_key.encode('utf-8')).decode('utf-8')


def decrypt_api_key(encrypted_key_b64):
    """Decrypt a Fernet token. Returns the plaintext string."""
    return _get_fernet().decrypt(encrypted_key_b64.encode('utf-8')).decode('utf-8')


def mask_api_key(api_key):
    """Create a masked display version of an API key (e.g., sk-...abc)."""
    if len(api_key) <= 6:
        return '***'
    return f'{api_key[:3]}...{api_key[-3:]}'
