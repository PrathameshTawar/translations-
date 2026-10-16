import logging
from flask import Flask, jsonify
from flask_cors import CORS
from app.config import get_config
from app.extensions import db, cache, jwt, limiter
from app.utils.logger import setup_logging


def create_app(config_name='development', overrides=None):
    """
    Build the Flask app.

    Args:
        config_name: 'development' or 'production'.
        overrides: config values applied before extensions initialise
                   (tests use it to e.g. disable rate limiting).
    """
    setup_logging()

    app = Flask(__name__)
    app.config.from_object(get_config(config_name))
    if overrides:
        app.config.update(overrides)

    if app.config.get('BEHIND_PROXY'):
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

    # Initialize extensions
    db.init_app(app)
    cache.init_app(app)
    jwt.init_app(app)
    limiter.init_app(app)

    # CORS — from config (defaults to local dev origins)
    CORS(app, resources={
        r'/api/*': {
            'origins': app.config['CORS_ORIGINS'],
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allow_headers': ['Content-Type', 'Authorization'],
            'supports_credentials': True,
        }
    })

    # Register blueprints
    from app.routes.translate import translate_bp
    from app.routes.auth import auth_bp
    from app.routes.history import history_bp
    from app.routes.evaluate import evaluate_bp
    from app.routes.translation_memory import tm_bp
    from app.routes.qa import qa_bp

    app.register_blueprint(translate_bp, url_prefix='/api/v1/translate')
    app.register_blueprint(auth_bp,      url_prefix='/api/v1/auth')
    app.register_blueprint(history_bp,   url_prefix='/api/v1/history')
    app.register_blueprint(evaluate_bp,  url_prefix='/api/v1/evaluate')
    app.register_blueprint(tm_bp,        url_prefix='/api/v1/tm')
    app.register_blueprint(qa_bp,        url_prefix='/api/v1/qa')

    # Health check
    @app.route('/api/health')
    def health():
        return jsonify({'status': 'ok', 'version': '1.0.0'}), 200

    # Error handlers
    register_error_handlers(app)

    # Create database tables
    with app.app_context():
        db.create_all()

    return app


def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'error': 'Bad request', 'detail': str(e) if app.debug else None}), 400

    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({'error': 'Unauthorized'}), 401

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({'error': 'Forbidden'}), 403

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return jsonify({'error': 'Rate limit exceeded'}), 429

    @app.errorhandler(500)
    def internal_error(e):
        app.logger.exception('Unhandled server error')
        return jsonify({'error': 'Internal server error'}), 500
