import os
import json
import base64
import functools
import logging
from flask import request, g, jsonify

logger = logging.getLogger(__name__)

# ALB OIDC headers — set by ALB after successful Federate/Midway authentication
# x-amzn-oidc-identity: the user's subject claim (e.g., alias or email)
# x-amzn-oidc-data: JWT with full user claims (email, name, etc.)
# x-amzn-oidc-accesstoken: the raw OIDC access token
USER_HEADER = os.environ.get('USER_IDENTITY_HEADER', 'x-amzn-oidc-identity')
OIDC_DATA_HEADER = 'x-amzn-oidc-data'
# Header carrying the user email. Only consumed by the LOCAL_DEV_MODE injector in
# app.py (get_current_user derives the email from the user id otherwise).
EMAIL_HEADER = os.environ.get('USER_EMAIL_HEADER', 'x-amzn-oidc-email')


def _decode_jwt_payload(token):
    """Decode the payload from a JWT without cryptographic verification.
    ALB OIDC headers are trusted (set by the ALB, not the client)."""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # Add padding
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += '=' * padding
        payload = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload)
    except Exception as e:
        logger.debug(f"Failed to decode JWT: {e}")
        return None


def get_current_user():
    """
    Extract user identity from ALB OIDC headers.

    After ALB authenticates via Federate/Midway OIDC, it sets:
    - x-amzn-oidc-identity: subject claim (user alias or email)
    - x-amzn-oidc-data: JWT with full claims (sub, email, name, etc.)

    Falls back to x-midway-user header for local dev mode.
    """
    # Check ALB OIDC identity header first
    user_id = request.headers.get(USER_HEADER)

    if user_id:
        email = f'{user_id}@amazon.com'

        # Try to get richer info from the OIDC data JWT
        oidc_data = request.headers.get(OIDC_DATA_HEADER)
        if oidc_data:
            claims = _decode_jwt_payload(oidc_data)
            if claims:
                # Extract alias from sub or email
                sub = claims.get('sub', user_id)
                email = claims.get('email', email)
                # Use the short alias if sub contains @
                if '@' in sub:
                    user_id = sub.split('@')[0]
                elif '@' not in user_id and '.' not in user_id:
                    pass  # Already a clean alias
                else:
                    user_id = sub

        return {'user_id': user_id, 'email': email}

    return None


def require_user(f):
    """
    Flask route decorator that ensures a valid user identity is present.
    Injects user_id and user_email into Flask's g object.

    In production, the ALB OIDC action handles authentication before
    the request reaches Flask — unauthenticated users are redirected
    to Federate login automatically by the ALB. So a missing identity
    header here means something is misconfigured.
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return jsonify({'error': 'Authentication required. No user identity found in request headers.'}), 401

        g.user_id = user['user_id']
        g.user_email = user['email']
        return f(*args, **kwargs)

    return decorated
