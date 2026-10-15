import os
from datetime import timedelta
from dotenv import load_dotenv

# Load .env early so config is available under gunicorn / wsgi too
load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret-key')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)

    # Cache
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_TTL', 86400))  # 24 hours

    # Request limits
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 2 * 1024 * 1024))  # 2 MB

    # CORS
    CORS_ORIGINS = [
        o.strip()
        for o in os.getenv('CORS_ORIGIN', 'http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173,http://127.0.0.1:3000').split(',')
        if o.strip()
    ]

    # Translator model
    NLLB_MODEL = os.getenv('NLLB_MODEL', 'facebook/nllb-200-distilled-600M')


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///translator.db')


class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')


config_by_name = dict(
    development=DevelopmentConfig,
    production=ProductionConfig,
)

# Values that must never be accepted as production secrets (defaults shipped
# in this repo or in example env files).
_PLACEHOLDER_SECRETS = {
    '', 'dev-secret-key', 'dev-jwt-secret-key', 'change-me', 'change-me-in-production',
    'changeme', 'secret',
}
_MIN_SECRET_LENGTH = 32


def validate_production_env(env=None):
    """
    Fail fast when production is started without real secrets.

    Runs when the production config is *selected* (not at import time), so
    importing this module in development or tests never requires production
    variables.
    """
    env = os.environ if env is None else env
    problems = []
    if not env.get('DATABASE_URL'):
        problems.append('DATABASE_URL must be set')
    for name in ('SECRET_KEY', 'JWT_SECRET_KEY'):
        value = env.get(name, '')
        if value.strip().lower() in _PLACEHOLDER_SECRETS:
            problems.append(f'{name} must be set to a real secret')
        elif len(value) < _MIN_SECRET_LENGTH:
            problems.append(f'{name} must be at least {_MIN_SECRET_LENGTH} characters')
    if problems:
        raise RuntimeError('Invalid production configuration: ' + '; '.join(problems)
                           + ". Generate secrets with: python -c \"import secrets; "
                           "print(secrets.token_urlsafe(48))\"")


def get_config(config_name='development'):
    try:
        config = config_by_name[config_name]
    except KeyError:
        raise RuntimeError(
            f"Unknown config: '{config_name}'. Valid options: {list(config_by_name.keys())}"
        )
    if config is ProductionConfig:
        validate_production_env()
    return config
