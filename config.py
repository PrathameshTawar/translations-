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

    # Fail fast in production — never boot with weak or missing secrets
    if not SQLALCHEMY_DATABASE_URI:
        raise RuntimeError('DATABASE_URL must be set in production')
    if not os.getenv('SECRET_KEY'):
        raise RuntimeError('SECRET_KEY must be set in production')
    if not os.getenv('JWT_SECRET_KEY'):
        raise RuntimeError('JWT_SECRET_KEY must be set in production')


config_by_name = dict(
    development=DevelopmentConfig,
    production=ProductionConfig,
)


def get_config(config_name='development'):
    try:
        return config_by_name[config_name]
    except KeyError:
        raise RuntimeError(
            f"Unknown config: '{config_name}'. Valid options: {list(config_by_name.keys())}"
        )
