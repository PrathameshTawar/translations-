from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

db = SQLAlchemy()
jwt = JWTManager()

# Per-route limits are declared on the routes (strings come from config, see
# Config.RATELIMIT_*). Storage defaults to in-process memory, which is
# per-worker; set RATELIMIT_STORAGE_URI=redis://... to share limits across
# gunicorn workers.
limiter = Limiter(key_func=get_remote_address, default_limits=[])

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
