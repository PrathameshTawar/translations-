import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev_key')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    DEBUG = True
    # Default to sqlite for local dev instead of postgres to make testing easier
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///translator.db')
    
class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')

config_by_name = dict(
    development=DevelopmentConfig,
    production=ProductionConfig
)
