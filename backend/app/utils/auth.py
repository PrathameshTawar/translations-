"""Shared-secret guard for operator-only endpoints (benchmarks, TM admin)."""

import hmac
from functools import wraps

from flask import current_app, jsonify, request

ADMIN_HEADER = 'X-Admin-Token'


def admin_required(fn):
    """
    Require the ``X-Admin-Token`` header to match ``ADMIN_TOKEN``.

    When ``ADMIN_TOKEN`` is not configured the endpoint is disabled (403)
    rather than open, so a fresh deployment never exposes expensive or
    destructive operations by accident.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = current_app.config.get('ADMIN_TOKEN') or ''
        if not expected:
            return jsonify({'error': 'Admin endpoints are disabled: '
                                     'set ADMIN_TOKEN to enable them'}), 403
        supplied = request.headers.get(ADMIN_HEADER, '')
        if not hmac.compare_digest(supplied.encode('utf-8'), expected.encode('utf-8')):
            return jsonify({'error': f'Missing or invalid {ADMIN_HEADER} header'}), 401
        return fn(*args, **kwargs)
    return wrapper


def config_limit(key):
    """Return a callable that reads a rate-limit string from app config."""
    return lambda: current_app.config[key]
