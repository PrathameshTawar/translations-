"""Config validation, admin-token guard and rate limiting."""

import pytest

STRONG = 'x' * 40


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'sqlite://')

    def _make(**overrides):
        from app import create_app
        app = create_app('development', overrides={
            'TESTING': True, 'SQLALCHEMY_DATABASE_URI': 'sqlite://', **overrides})
        return app.test_client()
    return _make


class TestProductionConfig:
    def test_config_module_imports_without_production_env(self, monkeypatch):
        import importlib
        for name in ('DATABASE_URL', 'SECRET_KEY', 'JWT_SECRET_KEY'):
            monkeypatch.delenv(name, raising=False)
        import app.config as cfg
        importlib.reload(cfg)  # used to raise in the ProductionConfig class body
        assert cfg.get_config('development') is cfg.DevelopmentConfig

    def test_production_rejects_missing_values(self):
        from app.config import validate_production_env
        with pytest.raises(RuntimeError, match='DATABASE_URL'):
            validate_production_env({})

    @pytest.mark.parametrize('secret', ['change-me-in-production', 'dev-secret-key', 'short'])
    def test_production_rejects_placeholder_or_short_secrets(self, secret):
        from app.config import validate_production_env
        with pytest.raises(RuntimeError, match='SECRET_KEY'):
            validate_production_env({'DATABASE_URL': 'sqlite://', 'SECRET_KEY': secret,
                                     'JWT_SECRET_KEY': STRONG})

    def test_production_accepts_strong_secrets(self):
        from app.config import validate_production_env
        validate_production_env({'DATABASE_URL': 'sqlite://', 'SECRET_KEY': STRONG,
                                 'JWT_SECRET_KEY': STRONG})

    def test_short_admin_token_rejected(self):
        from app.config import validate_production_env
        with pytest.raises(RuntimeError, match='ADMIN_TOKEN'):
            validate_production_env({'DATABASE_URL': 'sqlite://', 'SECRET_KEY': STRONG,
                                     'JWT_SECRET_KEY': STRONG, 'ADMIN_TOKEN': 'abc'})


class TestAdminGuard:
    def test_disabled_when_token_unset(self, make_client):
        client = make_client(ADMIN_TOKEN='', RATELIMIT_ENABLED=False)
        assert client.post('/api/v1/evaluate/run', json={}).status_code == 403
        assert client.get('/api/v1/tm/entries').status_code == 403

    def test_wrong_token_is_401(self, make_client):
        client = make_client(ADMIN_TOKEN=STRONG, RATELIMIT_ENABLED=False)
        r = client.post('/api/v1/tm/clear', headers={'X-Admin-Token': 'nope'})
        assert r.status_code == 401

    def test_valid_token_passes_guard(self, make_client):
        client = make_client(ADMIN_TOKEN=STRONG, RATELIMIT_ENABLED=False)
        r = client.get('/api/v1/tm/entries', headers={'X-Admin-Token': STRONG})
        assert r.status_code == 200

    def test_evaluate_run_rejects_concurrent_runs(self, make_client):
        import app.routes.evaluate as ev
        client = make_client(ADMIN_TOKEN=STRONG, RATELIMIT_ENABLED=False)
        assert ev._run_lock.acquire(blocking=False)
        try:
            r = client.post('/api/v1/evaluate/run', json={},
                            headers={'X-Admin-Token': STRONG})
            assert r.status_code == 409
        finally:
            ev._run_lock.release()


class TestRateLimits:
    def test_evaluate_run_is_rate_limited(self, make_client):
        from app.extensions import limiter
        client = make_client(ADMIN_TOKEN='', RATELIMIT_EVALUATE='2 per hour')
        limiter.reset()
        codes = [client.post('/api/v1/evaluate/run', json={}).status_code
                 for _ in range(3)]
        assert codes == [403, 403, 429]

    def test_login_is_rate_limited(self, make_client):
        from app.extensions import limiter
        client = make_client(RATELIMIT_AUTH='3 per minute')
        limiter.reset()
        body = {'email': 'a@example.com', 'password': 'wrongpass1'}
        codes = [client.post('/api/v1/auth/login', json=body).status_code
                 for _ in range(4)]
        assert codes[-1] == 429
        assert 429 not in codes[:3]
