"""Input validators for translation and auth endpoints."""
import re
from typing import Tuple, Optional

SUPPORTED_LANGUAGES = {
    'english', 'hindi', 'marathi', 'bengali',
    'gujarati', 'tamil', 'telugu', 'kannada'
}
EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')


def validate_translation_request(data: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate a translation request body.
    Returns (is_valid, error_message).
    """
    if not data:
        return False, 'Request body is required'

    text = (data.get('text') or data.get('query') or '').strip()
    if not text:
        return False, 'text field is required'
    if len(text) > 5000:
        return False, f'text exceeds 5000 character limit (got {len(text)})'

    source_lang = (data.get('source_lang') or 'english').lower().strip()
    target_lang = (data.get('target_lang') or 'hindi').lower().strip()

    if source_lang not in SUPPORTED_LANGUAGES:
        return False, f'Unsupported source_lang: {source_lang}'
    if target_lang not in SUPPORTED_LANGUAGES:
        return False, f'Unsupported target_lang: {target_lang}'
    if source_lang == target_lang:
        return False, 'source_lang and target_lang must differ'

    return True, None


def validate_register_request(data: dict) -> Tuple[bool, Optional[str]]:
    """Validate user registration data."""
    if not data:
        return False, 'Request body is required'

    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip()
    password = data.get('password') or ''

    if not name or len(name) < 2:
        return False, 'name must be at least 2 characters'
    if not email or not EMAIL_RE.match(email):
        return False, 'A valid email address is required'
    if len(password) < 6:
        return False, 'password must be at least 6 characters'

    return True, None


def validate_login_request(data: dict) -> Tuple[bool, Optional[str]]:
    """Validate login data."""
    if not data:
        return False, 'Request body is required'

    email = (data.get('email') or '').strip()
    password = data.get('password') or ''

    if not email:
        return False, 'email is required'
    if not password:
        return False, 'password is required'

    return True, None
