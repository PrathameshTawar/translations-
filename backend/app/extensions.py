from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
jwt = JWTManager()

# Simple mock cache to avoid Redis requirement for local dev MVP
class SimpleCache:
    def __init__(self):
        self.store = {}
    def init_app(self, app):
        pass
    def get(self, key):
        return self.store.get(key)
    def set(self, key, value, timeout=None):
        self.store[key] = value

cache = SimpleCache()
